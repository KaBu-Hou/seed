from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from precompute_image56_npy import map_chunk_to_image56
from train_eval import build_context_indices
from utils import ensure_dir, project_root, save_json


def summarize_array(name: str, arr: np.ndarray) -> dict:
    finite = np.isfinite(arr)
    summary = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_ratio": float(finite.mean()),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }
    if finite.any():
        vals = arr[finite].astype(np.float64, copy=False)
        summary.update(
            {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "p01": float(np.percentile(vals, 1)),
                "p50": float(np.percentile(vals, 50)),
                "p99": float(np.percentile(vals, 99)),
            }
        )
    return summary


def counts(values: np.ndarray) -> dict[str, int]:
    unique, cnt = np.unique(values, return_counts=True)
    return {str(k): int(v) for k, v in zip(unique.tolist(), cnt.tolist())}


def plot_sample_grid(x: np.ndarray, indices: np.ndarray, out_path: Path) -> None:
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]
    fig, axes = plt.subplots(len(indices), 5, figsize=(12, max(2.2, 2.2 * len(indices))))
    if len(indices) == 1:
        axes = np.expand_dims(axes, 0)
    for row, idx in enumerate(indices):
        sample = np.asarray(x[int(idx)], dtype=np.float32)
        for band in range(5):
            ax = axes[row, band]
            ax.imshow(sample[band], cmap="viridis", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(band_names[band])
            if band == 0:
                ax.set_ylabel(f"idx {int(idx)}")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate materialized image56 EEG cache quality.")
    parser.add_argument("--source_npz", default=str(project_root() / "processed" / "eeg_de_lds.npz"))
    parser.add_argument("--image56_dir", default=str(project_root() / "processed" / "eeg_de_lds_image56_npy"))
    parser.add_argument("--num_compare", type=int, default=64)
    parser.add_argument("--num_visual", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=str(project_root() / "processed" / "image56_quality"))
    args = parser.parse_args()

    cache_dir = Path(args.image56_dir)
    out_dir = ensure_dir(args.out_dir)
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    x_img = np.load(cache_dir / meta["x_file"], mmap_mode="r")
    y_img = np.load(cache_dir / "y.npy", allow_pickle=True)
    subject_img = np.load(cache_dir / "subject.npy", allow_pickle=True)
    session_img = np.load(cache_dir / "session.npy", allow_pickle=True)
    trial_img = np.load(cache_dir / "trial.npy", allow_pickle=True)

    src = np.load(args.source_npz, allow_pickle=True)
    x_src = src["x"].astype(np.float32, copy=False)
    y_src = src["y"]
    subject_src = src["subject"]
    session_src = src["session"]
    trial_src = src["trial"]

    rng = np.random.default_rng(args.seed)
    n = len(y_src)
    compare_idx = np.sort(rng.choice(n, size=min(args.num_compare, n), replace=False))
    context_indices = build_context_indices(subject_src, session_src, trial_src, int(meta["context"]))
    expected = map_chunk_to_image56(x_src, context_indices[compare_idx])
    actual = np.asarray(x_img[compare_idx], dtype=np.float32)
    abs_diff = np.abs(expected - actual)

    report = {
        "cache_dir": str(cache_dir),
        "source_npz": str(args.source_npz),
        "meta": meta,
        "shape_ok": list(x_img.shape) == meta["shape"] and tuple(x_img.shape[1:]) == (5, 56, 56),
        "lengths_ok": len(x_img) == len(y_img) == len(subject_img) == len(session_img) == len(trial_img),
        "labels_match_source": bool(np.array_equal(y_img, y_src)),
        "subjects_match_source": bool(np.array_equal(subject_img, subject_src)),
        "sessions_match_source": bool(np.array_equal(session_img.astype(str), session_src.astype(str))),
        "trials_match_source": bool(np.array_equal(trial_img, trial_src)),
        "source_summary": summarize_array("source_x", x_src),
        "cache_sample_summary": summarize_array("cache_x_sample", np.asarray(x_img[compare_idx], dtype=np.float32)),
        "label_counts": counts(y_img),
        "subject_counts": counts(subject_img),
        "trial_counts": counts(trial_img),
        "dynamic_compare": {
            "num_samples": int(len(compare_idx)),
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
            "p99_abs_diff": float(np.percentile(abs_diff, 99)),
            "allclose_atol_1e-5": bool(np.allclose(expected, actual, atol=1e-5, rtol=1e-5)),
        },
    }
    report["passed"] = bool(
        report["shape_ok"]
        and report["lengths_ok"]
        and report["labels_match_source"]
        and report["subjects_match_source"]
        and report["sessions_match_source"]
        and report["trials_match_source"]
        and report["cache_sample_summary"]["finite_ratio"] == 1.0
        and report["dynamic_compare"]["allclose_atol_1e-5"]
    )

    save_json(out_dir / "image56_quality_report.json", report)
    visual_idx = np.sort(rng.choice(n, size=min(args.num_visual, n), replace=False))
    plot_sample_grid(x_img, visual_idx, out_dir / "image56_sample_grid.png")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {out_dir / 'image56_quality_report.json'}")
    print(f"Saved visual grid: {out_dir / 'image56_sample_grid.png'}")


if __name__ == "__main__":
    main()
