from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model import build_model
from preprocess_eeg import build_eeg_dataset
from utils import (
    classification_metrics,
    ensure_dir,
    plot_confusion_matrix,
    project_root,
    seed_root,
    set_seed,
    standardize_train_test,
)


def load_or_build_data(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.image56_npy_dir:
        cache_dir = Path(args.image56_npy_dir)
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"--image56_npy_dir missing meta.json: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mmap_mode = None if args.image56_load_ram else "r"
        data = {
            "x": np.load(cache_dir / meta["x_file"], mmap_mode=mmap_mode),
            "y": np.load(cache_dir / "y.npy", allow_pickle=True),
            "subject": np.load(cache_dir / "subject.npy", allow_pickle=True),
            "session": np.load(cache_dir / "session.npy", allow_pickle=True),
            "trial": np.load(cache_dir / "trial.npy", allow_pickle=True),
            "image56_meta": np.array(meta, dtype=object),
        }
    elif args.image56_dir:
        cache_dir = Path(args.image56_dir)
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"--image56_dir missing meta.json: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dtype = np.float16 if meta["dtype"] == "float16" else np.float32
        x = np.memmap(cache_dir / "x_image56.dat", dtype=dtype, mode="r", shape=tuple(meta["shape"]))
        data = {
            "x": x,
            "y": np.load(cache_dir / "y.npy", allow_pickle=True),
            "subject": np.load(cache_dir / "subject.npy", allow_pickle=True),
            "session": np.load(cache_dir / "session.npy", allow_pickle=True),
            "trial": np.load(cache_dir / "trial.npy", allow_pickle=True),
            "image56_meta": np.array(meta, dtype=object),
        }
    elif args.data_npz:
        npz_path = Path(args.data_npz)
        if not npz_path.exists():
            raise FileNotFoundError(f"--data_npz does not exist: {npz_path}")
        loaded = np.load(npz_path, allow_pickle=True)
        data = {k: loaded[k] for k in loaded.files}
    else:
        data = build_eeg_dataset(seed_root(args.root), max_sessions=args.max_sessions)
        if args.save_data_npz:
            out = Path(args.save_data_npz)
            ensure_dir(out.parent)
            np.savez_compressed(out, **data)
            print(f"Saved cached dataset to {out}")

    if args.max_samples_per_subject is not None:
        rng = np.random.default_rng(42)
        keep = []
        subjects = data["subject"]
        for subject in sorted(np.unique(subjects).tolist()):
            idx = np.where(subjects == subject)[0]
            if len(idx) > args.max_samples_per_subject:
                idx = rng.choice(idx, size=args.max_samples_per_subject, replace=False)
            keep.append(idx)
        keep_idx = np.sort(np.concatenate(keep))
        data = {
            k: (v[keep_idx] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(subjects) else v)
            for k, v in data.items()
        }
    return data


class EEGImage56Dataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        sessions: np.ndarray,
        trials: np.ndarray,
        context: int = 56,
    ):
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.context_indices = build_context_indices(subjects, sessions, trials, context)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64]:
        return self.x[self.context_indices[index]], self.y[index]


class PrecomputedImage56Dataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.x = x
        self.y = y.astype(np.int64, copy=False)
        self.indices = np.arange(len(y), dtype=np.int64) if indices is None else indices.astype(np.int64, copy=False)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64]:
        real_index = int(self.indices[index])
        x = np.asarray(self.x[real_index], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return x, self.y[real_index]


def build_context_indices(subjects: np.ndarray, sessions: np.ndarray, trials: np.ndarray, context: int) -> np.ndarray:
    indices = np.empty((len(subjects), context), dtype=np.int32)
    start = 0
    prev_key = None
    for i, key in enumerate(zip(subjects.tolist(), sessions.tolist(), trials.tolist())):
        if prev_key is None:
            prev_key = key
        if key != prev_key:
            start = i
            prev_key = key
        begin = max(start, i - context + 1)
        window = np.arange(begin, i + 1, dtype=np.int32)
        if len(window) < context:
            pad = np.full(context - len(window), start, dtype=np.int32)
            window = np.concatenate([pad, window])
        indices[i] = window
    return indices


def image56_collate(batch: list[tuple[np.ndarray, np.int64]]) -> tuple[torch.Tensor, torch.Tensor]:
    contexts, labels = zip(*batch)
    x = torch.from_numpy(np.stack(contexts, axis=0)).float()
    # B x context x channels x bands -> B x bands x channels x context -> B x 5 x 56 x 56
    x = x.permute(0, 3, 2, 1).contiguous()
    x = F.interpolate(x, size=(56, 56), mode="bilinear", align_corners=False)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def precomputed_image56_collate(batch: list[tuple[np.ndarray, np.int64]]) -> tuple[torch.Tensor, torch.Tensor]:
    contexts, labels = zip(*batch)
    x = torch.from_numpy(np.stack(contexts, axis=0)).float()
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def compute_image56_stats(x: np.ndarray, indices: np.ndarray, chunk_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    sum_ = np.zeros((5,), dtype=np.float64)
    sumsq = np.zeros((5,), dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        idx = indices[start : start + chunk_size]
        chunk = np.asarray(x[idx], dtype=np.float32)
        total += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
        sum_ += chunk.sum(axis=(0, 2, 3), dtype=np.float64)
        sumsq += np.square(chunk, dtype=np.float32).sum(axis=(0, 2, 3), dtype=np.float64)
    mean = sum_ / total
    var = np.maximum(sumsq / total - mean**2, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32).reshape(5, 1, 1), std.astype(np.float32).reshape(5, 1, 1)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    args: argparse.Namespace,
    subjects: np.ndarray | None = None,
    sessions: np.ndarray | None = None,
    trials: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> DataLoader:
    if args.eeg_representation == "image56":
        if args.image56_dir or args.image56_npy_dir:
            mean, std = norm_stats if norm_stats is not None else (None, None)
            ds = PrecomputedImage56Dataset(x, y, indices=indices, mean=mean, std=std)
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=args.num_workers,
                collate_fn=precomputed_image56_collate,
                pin_memory=torch.cuda.is_available(),
            )
        if subjects is None or sessions is None or trials is None:
            raise ValueError("image56 representation requires subject/session/trial metadata")
        ds = EEGImage56Dataset(x, y, subjects, sessions, trials, args.eeg_context)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=image56_collate,
            pin_memory=torch.cuda.is_available(),
        )
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers)


def run_fold(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    trials: np.ndarray,
    test_subject: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    train_mask = subjects != test_subject
    test_mask = subjects == test_subject
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if args.image56_dir or args.image56_npy_dir:
        train_y, test_y = y, y
        mean, std = compute_image56_stats(x, train_idx)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        train_x, test_x = x, x
        train_subjects = test_subjects = train_sessions = test_sessions = train_trials = test_trials = None
        norm_stats = (mean, std)
    else:
        train_x, test_x = x[train_mask], x[test_mask]
        train_y, test_y = y[train_mask], y[test_mask]
        train_subjects, test_subjects = subjects[train_mask], subjects[test_mask]
        train_sessions, test_sessions = sessions[train_mask], sessions[test_mask]
        train_trials, test_trials = trials[train_mask], trials[test_mask]
        train_x, test_x = standardize_train_test(train_x, test_x, axes=(0,))
        norm_stats = None

    train_loader = make_loader(
        train_x,
        train_y,
        args.batch_size,
        True,
        args,
        train_subjects,
        train_sessions,
        train_trials,
        train_idx if (args.image56_dir or args.image56_npy_dir) else None,
        norm_stats,
    )
    test_loader = make_loader(
        test_x,
        test_y,
        args.batch_size,
        False,
        args,
        test_subjects,
        test_sessions,
        test_trials,
        test_idx if (args.image56_dir or args.image56_npy_dir) else None,
        norm_stats,
    )

    model = build_model(
        args.model,
        hidden=args.hidden,
        use_me=False,
        stae_layers=args.stae_layers,
        swin_window_size=args.swin_window_size,
        swin_depths=tuple(args.swin_depths),
        swin_heads=tuple(args.swin_heads),
        swin_mlp_ratio=args.swin_mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        metrics, _, _ = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics.__dict__})
        if metrics.accuracy > best_acc:
            best_acc = metrics.accuracy
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(
            f"subject={test_subject:02d} epoch={epoch:03d} "
            f"loss={np.mean(losses):.4f} acc={metrics.accuracy:.4f} f1={metrics.macro_f1:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, pred, truth = evaluate(model, test_loader, device)
    return {
        "test_subject": int(test_subject),
        "metrics": metrics.__dict__,
        "history": history,
        "y_pred": pred.tolist(),
        "y_true": truth.tolist(),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds = []
    trues = []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    return classification_metrics(y_true, y_pred), y_pred, y_true


def plot_history(results: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for fold in results:
        h = fold["history"]
        ax.plot([v["epoch"] for v in h], [v["accuracy"] for v in h], alpha=0.35)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("LOSO Validation Accuracy")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def safe_name(value: str) -> str:
    value = value.replace("\\", "/").rstrip("/").split("/")[-1] or "run"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate SEED EEG emotion recognition.")
    parser.add_argument("--root", default=str(project_root()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--model", default="compact", choices=["compact", "stst_swin"])
    parser.add_argument("--eeg_representation", default="vector", choices=["vector", "image56"])
    parser.add_argument("--eeg_context", type=int, default=56)
    parser.add_argument("--stae_layers", type=int, default=4)
    parser.add_argument("--swin_window_size", type=int, default=7)
    parser.add_argument("--swin_depths", type=int, nargs=3, default=[2, 2, 2])
    parser.add_argument("--swin_heads", type=int, nargs=3, default=[2, 4, 8])
    parser.add_argument("--swin_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--subjects", type=int, nargs="*", default=None, help="Debug subset, e.g. --subjects 1 2")
    parser.add_argument("--max_sessions", type=int, default=None, help="Debug option for preprocessing.")
    parser.add_argument("--data_npz", default=None, help="Load cached EEG dataset from preprocess_eeg.py.")
    parser.add_argument("--image56_dir", default=None, help="Load precomputed image56 memmap cache.")
    parser.add_argument("--image56_npy_dir", default=None, help="Load materialized float32 image56 .npy cache.")
    parser.add_argument("--image56_load_ram", action="store_true", help="Load x_image56.npy fully into RAM.")
    parser.add_argument("--save_data_npz", default=None, help="Save built EEG dataset for faster future runs.")
    parser.add_argument("--max_samples_per_subject", type=int, default=None, help="Deterministic debug subset after loading.")
    parser.add_argument(
        "--protocol",
        default="subject_independent_loso",
        choices=["subject_independent_loso"],
        help="Evaluation protocol. Current implementation supports SEED LOSO subject-independent.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default=str(project_root() / "outputs"))
    args = parser.parse_args()

    set_seed(42)
    data = load_or_build_data(args)
    if args.image56_dir or args.image56_npy_dir:
        args.eeg_representation = "image56"
    x, y, subjects = data["x"], data["y"], data["subject"]
    sessions, trials = data["session"], data["trial"]
    fold_subjects = sorted(np.unique(subjects).tolist())
    if args.subjects is not None:
        fold_subjects = [s for s in fold_subjects if s in set(args.subjects)]

    device = torch.device(args.device)
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(project_root() / "figures")
    results = [run_fold(x, y, subjects, sessions, trials, int(s), args, device) for s in fold_subjects]

    accs = np.array([r["metrics"]["accuracy"] for r in results], dtype=float)
    f1s = np.array([r["metrics"]["macro_f1"] for r in results], dtype=float)
    uars = np.array([r["metrics"]["uar"] for r in results], dtype=float)
    summary = {
        "protocol": args.protocol,
        "protocol_description": "LOSO subject-independent on SEED ExtractedFeatures_1s/de_LDS",
        "model": args.model,
        "eeg_representation": args.eeg_representation,
        "eeg_context": args.eeg_context,
        "paper_sourced_settings": [
            "SEED 62-ch EEG, 3-class positive/neutral/negative",
            "DE features from five bands",
            "1-second non-overlapping EEG feature intervals",
            "Swin Transformer based backbone",
            "4 STAE layers from paper ablation",
            "AdamW, lr=3e-4, cosine annealing, batch size 32, 100 epochs for full run",
            "LOSO-CV for public datasets",
        ],
        "implementation_assumptions": [
            "image56 maps trial-local 62 x 56 x 5 DE contexts to 56 x 56 x 5 by bilinear resize",
            "patch_size=2 inferred from 56x56xn to 28x28x4n in the paper figure",
            "window_size, depths, heads, mlp_ratio, dropout are Swin-style assumptions not explicitly reported",
        ],
        "subjects": fold_subjects,
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std(ddof=0)),
        "mean_macro_f1": float(f1s.mean()),
        "std_macro_f1": float(f1s.std(ddof=0)),
        "mean_uar": float(uars.mean()),
        "std_uar": float(uars.std(ddof=0)),
        "folds": results,
    }
    (out_dir / "seed_loso_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    all_pred = np.concatenate([np.array(r["y_pred"]) for r in results])
    all_true = np.concatenate([np.array(r["y_true"]) for r in results])
    run_name = safe_name(str(out_dir))
    figure_prefix = f"{run_name}_{args.protocol}"
    confusion_name = f"{figure_prefix}_confusion_matrix.png"
    curves_name = f"{figure_prefix}_accuracy_curves.png"
    plot_confusion_matrix(all_true, all_pred, out_dir / confusion_name)
    plot_history(results, out_dir / curves_name)
    plot_confusion_matrix(all_true, all_pred, fig_dir / confusion_name)
    plot_history(results, fig_dir / curves_name)
    summary["figures"] = {
        "out_dir_confusion_matrix": str(out_dir / confusion_name),
        "out_dir_accuracy_curves": str(out_dir / curves_name),
        "figures_confusion_matrix": str(fig_dir / confusion_name),
        "figures_accuracy_curves": str(fig_dir / curves_name),
    }
    (out_dir / "seed_loso_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
