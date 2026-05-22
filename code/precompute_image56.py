from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_eval import build_context_indices
from utils import ensure_dir, project_root


def make_image56_chunk(x: np.ndarray, context_indices: np.ndarray) -> np.ndarray:
    contexts = x[context_indices]
    tensor = torch.from_numpy(contexts).float()
    tensor = tensor.permute(0, 3, 2, 1).contiguous()
    tensor = F.interpolate(tensor, size=(56, 56), mode="bilinear", align_corners=False)
    return tensor.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute SEED EEG image56 tensors as a memmap cache.")
    parser.add_argument("--data_npz", default=str(project_root() / "processed" / "eeg_de_lds.npz"))
    parser.add_argument("--out_dir", default=str(project_root() / "processed" / "eeg_de_lds_image56"))
    parser.add_argument("--context", type=int, default=56)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    src = np.load(args.data_npz, allow_pickle=True)
    x = src["x"].astype(np.float32, copy=False)
    y = src["y"].astype(np.int64, copy=False)
    subjects = src["subject"]
    sessions = src["session"]
    trials = src["trial"]
    out_dir = ensure_dir(args.out_dir)

    context_indices = build_context_indices(subjects, sessions, trials, args.context)
    dtype = np.float16 if args.dtype == "float16" else np.float32
    image_path = out_dir / "x_image56.dat"
    mmap = np.memmap(image_path, dtype=dtype, mode="w+", shape=(len(y), 5, 56, 56))

    for start in range(0, len(y), args.chunk_size):
        end = min(start + args.chunk_size, len(y))
        chunk = make_image56_chunk(x, context_indices[start:end])
        mmap[start:end] = chunk.astype(dtype, copy=False)
        if start == 0 or end == len(y) or (start // args.chunk_size) % 25 == 0:
            print(f"precomputed {end}/{len(y)}")
    mmap.flush()

    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "subject.npy", subjects)
    np.save(out_dir / "session.npy", sessions)
    np.save(out_dir / "trial.npy", trials)
    meta = {
        "format": "eeg_image56_memmap",
        "source": str(args.data_npz),
        "image_path": str(image_path),
        "shape": [int(len(y)), 5, 56, 56],
        "dtype": args.dtype,
        "context": args.context,
        "mapping": "trial-local 62 x context x 5 DE windows -> B x 5 x 56 x 56 bilinear resize",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved image56 cache to {out_dir}")


if __name__ == "__main__":
    main()
