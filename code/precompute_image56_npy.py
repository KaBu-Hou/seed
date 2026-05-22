from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_eval import build_context_indices
from utils import ensure_dir, project_root


def map_chunk_to_image56(x: np.ndarray, context_indices: np.ndarray) -> np.ndarray:
    contexts = x[context_indices].astype(np.float32, copy=False)
    tensor = torch.from_numpy(contexts)
    # B x context x channels x bands -> B x bands x channels x context.
    tensor = tensor.permute(0, 3, 2, 1).contiguous()
    tensor = F.interpolate(tensor, size=(56, 56), mode="bilinear", align_corners=False)
    return tensor.cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize all SEED EEG DE samples as float32 image56 .npy tensors for faster training."
    )
    parser.add_argument("--data_npz", default=str(project_root() / "processed" / "eeg_de_lds.npz"))
    parser.add_argument("--out_dir", default=str(project_root() / "processed" / "eeg_de_lds_image56_npy"))
    parser.add_argument("--context", type=int, default=56)
    parser.add_argument("--chunk_size", type=int, default=1024)
    args = parser.parse_args()

    data = np.load(args.data_npz, allow_pickle=True)
    x = data["x"].astype(np.float32, copy=False)
    y = data["y"].astype(np.int64, copy=False)
    subjects = data["subject"]
    sessions = data["session"]
    trials = data["trial"]

    out_dir = ensure_dir(args.out_dir)
    out_x = out_dir / "x_image56.npy"
    context_indices = build_context_indices(subjects, sessions, trials, args.context)
    x_out = np.lib.format.open_memmap(out_x, mode="w+", dtype=np.float32, shape=(len(y), 5, 56, 56))

    for start in range(0, len(y), args.chunk_size):
        end = min(start + args.chunk_size, len(y))
        x_out[start:end] = map_chunk_to_image56(x, context_indices[start:end])
        x_out.flush()
        print(f"saved {end}/{len(y)}")

    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "subject.npy", subjects)
    np.save(out_dir / "session.npy", sessions)
    np.save(out_dir / "trial.npy", trials)
    meta = {
        "format": "eeg_image56_npy",
        "source": str(args.data_npz),
        "x_file": "x_image56.npy",
        "shape": [int(len(y)), 5, 56, 56],
        "dtype": "float32",
        "context": int(args.context),
        "mapping": "trial-local 62 x context x 5 DE windows -> B x 5 x 56 x 56 bilinear resize",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved materialized image56 dataset to {out_dir}")


if __name__ == "__main__":
    main()
