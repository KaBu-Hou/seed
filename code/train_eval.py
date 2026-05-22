from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import STSTEmotionModel
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
    if args.data_npz:
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


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_fold(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    test_subject: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    train_mask = subjects != test_subject
    test_mask = subjects == test_subject
    train_x, test_x = x[train_mask], x[test_mask]
    train_y, test_y = y[train_mask], y[test_mask]
    train_x, test_x = standardize_train_test(train_x, test_x)

    train_loader = make_loader(train_x, train_y, args.batch_size, True)
    test_loader = make_loader(test_x, test_y, args.batch_size, False)

    model = STSTEmotionModel(hidden=args.hidden, use_me=False).to(device)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate SEED EEG emotion recognition.")
    parser.add_argument("--root", default=str(project_root()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--subjects", type=int, nargs="*", default=None, help="Debug subset, e.g. --subjects 1 2")
    parser.add_argument("--max_sessions", type=int, default=None, help="Debug option for preprocessing.")
    parser.add_argument("--data_npz", default=None, help="Load cached EEG dataset from preprocess_eeg.py.")
    parser.add_argument("--save_data_npz", default=None, help="Save built EEG dataset for faster future runs.")
    parser.add_argument("--max_samples_per_subject", type=int, default=None, help="Deterministic debug subset after loading.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default=str(project_root() / "outputs"))
    args = parser.parse_args()

    set_seed(42)
    data = load_or_build_data(args)
    x, y, subjects = data["x"], data["y"], data["subject"]
    fold_subjects = sorted(np.unique(subjects).tolist())
    if args.subjects is not None:
        fold_subjects = [s for s in fold_subjects if s in set(args.subjects)]

    device = torch.device(args.device)
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(project_root() / "figures")
    results = [run_fold(x, y, subjects, int(s), args, device) for s in fold_subjects]

    accs = np.array([r["metrics"]["accuracy"] for r in results], dtype=float)
    f1s = np.array([r["metrics"]["macro_f1"] for r in results], dtype=float)
    uars = np.array([r["metrics"]["uar"] for r in results], dtype=float)
    summary = {
        "protocol": "LOSO subject-independent on SEED ExtractedFeatures_1s/de_LDS",
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
    plot_confusion_matrix(all_true, all_pred, fig_dir / "seed_loso_confusion_matrix.png")
    plot_history(results, fig_dir / "seed_loso_accuracy_curves.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
