from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

from utils import (
    ensure_dir,
    feature_files,
    label_indices,
    project_root,
    seed_root,
    session_from_feature_file,
    subject_from_feature_file,
)


def load_trial_feature(mat: dict, feature: str, smooth: str, trial_index: int) -> np.ndarray:
    key = f"{feature}_{smooth}{trial_index + 1}"
    if key not in mat:
        raise KeyError(f"Missing {key}. Available examples: {[k for k in mat if not k.startswith('__')][:10]}")
    arr = mat[key].astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected channel x time x band array for {key}, got {arr.shape}")
    return np.transpose(arr, (1, 0, 2))


def build_eeg_dataset(
    sjtu_root: str | Path,
    feature: str = "de",
    smooth: str = "LDS",
    max_sessions: int | None = None,
) -> dict[str, np.ndarray]:
    files = feature_files(sjtu_root)
    if max_sessions is not None:
        files = files[:max_sessions]
    labels_by_trial = label_indices()

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    sessions: list[np.ndarray] = []
    trials: list[np.ndarray] = []

    for file_path in files:
        mat = sio.loadmat(file_path)
        subject = subject_from_feature_file(file_path)
        session = session_from_feature_file(file_path)
        for trial_idx, label in enumerate(labels_by_trial):
            trial_x = load_trial_feature(mat, feature, smooth, trial_idx)
            n = trial_x.shape[0]
            xs.append(trial_x)
            ys.append(np.full(n, label, dtype=np.int64))
            subjects.append(np.full(n, subject, dtype=np.int64))
            sessions.append(np.array([session] * n, dtype=object))
            trials.append(np.full(n, trial_idx + 1, dtype=np.int64))

    return {
        "x": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "subject": np.concatenate(subjects, axis=0),
        "session": np.concatenate(sessions, axis=0),
        "trial": np.concatenate(trials, axis=0),
        "label_names": np.array(["negative", "neutral", "positive"], dtype=object),
        "feature": np.array(feature),
        "smooth": np.array(smooth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 1-second SEED EEG feature tensors.")
    parser.add_argument("--root", default=str(project_root()))
    parser.add_argument("--feature", default="de", choices=["de", "psd", "dasm", "rasm", "asm", "dcau"])
    parser.add_argument("--smooth", default="LDS", choices=["LDS", "movingAve"])
    parser.add_argument("--max_sessions", type=int, default=None, help="Debug option.")
    parser.add_argument("--out", default=str(project_root() / "processed" / "eeg_de_lds.npz"))
    args = parser.parse_args()

    sjtu = seed_root(args.root)
    data = build_eeg_dataset(sjtu, args.feature, args.smooth, args.max_sessions)
    out = Path(args.out)
    ensure_dir(out.parent)
    np.savez_compressed(out, **data)
    print(f"Saved {out}")
    print(f"x={data['x'].shape}, y={data['y'].shape}, subjects={sorted(set(data['subject'].tolist()))}")


if __name__ == "__main__":
    main()
