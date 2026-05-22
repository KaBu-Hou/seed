from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from utils import ensure_dir, project_root, seed_root

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


VIDEO_EXTS = {".mp4", ".mkv", ".rmvb", ".avi", ".mov"}


def five_frame_tensor(video_path: str | Path, size: int = 56) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = 5
    indices = np.linspace(0, max(frame_count - 1, 0), 5).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            if frames:
                frames.append(frames[-1].copy())
                continue
            frames.append(np.zeros((size, size), dtype=np.float32))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        arr = gray.astype(np.float32) / 255.0
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        frames.append(arr)
    cap.release()
    return np.stack(frames, axis=0).astype(np.float32)


def build_stimulus_proxy(sjtu_root: str | Path) -> dict[str, np.ndarray]:
    stim_dir = Path(sjtu_root) / "Stimuli"
    paths = sorted([p for p in stim_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS])
    tensors = []
    labels = []
    names = []
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    for path in paths:
        emotion = path.parent.name.lower()
        if emotion not in label_map:
            continue
        try:
            tensors.append(five_frame_tensor(path))
            labels.append(label_map[emotion])
            names.append(str(path))
        except RuntimeError as exc:
            print(exc)
    return {
        "x": np.stack(tensors, axis=0) if tensors else np.empty((0, 5, 56, 56), dtype=np.float32),
        "y": np.array(labels, dtype=np.int64),
        "path": np.array(names, dtype=object),
        "mode": np.array("stimulus_proxy"),
        "note": np.array(
            "These are SEED film stimulus frames, not participant micro-expression recordings."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 5-frame 56x56 tensors. Current SEED files provide stimulus clips only."
    )
    parser.add_argument("--root", default=str(project_root()))
    parser.add_argument("--out", default=str(project_root() / "processed" / "me_stimulus_proxy.npz"))
    args = parser.parse_args()

    data = build_stimulus_proxy(seed_root(args.root))
    out = Path(args.out)
    ensure_dir(out.parent)
    np.savez_compressed(out, **data)
    print(f"Saved {out}")
    print(f"x={data['x'].shape}, y={data['y'].shape}")
    print(str(data["note"]))


if __name__ == "__main__":
    main()
