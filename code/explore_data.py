from __future__ import annotations

import argparse
from pathlib import Path

import scipy.io as sio

from utils import compact_tree, ensure_dir, feature_files, project_root, save_json, seed_root


def inspect_feature_file(path: Path) -> dict:
    mat = sio.loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    selected = {}
    for key in ["de_LDS1", "de_movingAve1", "psd_LDS1", "dasm_LDS1", "asm_LDS1"]:
        if key in mat:
            arr = mat[key]
            selected[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return {"file": str(path), "num_arrays": len(keys), "selected_arrays": selected}


def inspect_preprocessed_file(path: Path) -> dict:
    mat = sio.loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    shapes = {k: list(mat[k].shape) for k in keys[:15]}
    return {"file": str(path), "num_arrays": len(keys), "trial_shapes": shapes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the local SEED/SJTU dataset.")
    parser.add_argument("--root", default=str(project_root()), help="Project root containing SEED/SJTU.")
    parser.add_argument("--out_dir", default=str(project_root() / "processed"))
    args = parser.parse_args()

    sjtu = seed_root(args.root)
    out_dir = ensure_dir(args.out_dir)
    tree = compact_tree(sjtu, max_files_per_dir=10)
    (out_dir / "seed_directory_tree.txt").write_text(tree, encoding="utf-8")

    feat_files = feature_files(sjtu)
    pre_dir = sjtu / "Preprocessed_EEG"
    pre_files = sorted([p for p in pre_dir.glob("*.mat") if p.name.lower() != "label.mat"])
    stim_files = sorted((sjtu / "Stimuli").rglob("*.*"))
    video_files = [p for p in stim_files if p.suffix.lower() in {".mp4", ".mkv", ".rmvb", ".avi"}]

    summary = {
        "sjtu_root": str(sjtu),
        "extracted_feature_files": len(feat_files),
        "preprocessed_eeg_files": len(pre_files),
        "stimulus_video_files": len(video_files),
        "feature_example": inspect_feature_file(feat_files[0]) if feat_files else None,
        "preprocessed_example": inspect_preprocessed_file(pre_files[0]) if pre_files else None,
        "note": "Stimuli are emotion-inducing film clips, not participant facial response videos.",
    }
    save_json(out_dir / "seed_summary.json", summary)
    print(tree)
    print("\nSaved:")
    print(out_dir / "seed_directory_tree.txt")
    print(out_dir / "seed_summary.json")


if __name__ == "__main__":
    main()
