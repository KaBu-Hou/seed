from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SEED_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])
LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def seed_root(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else project_root()
    sjtu = base / "SEED" / "SJTU"
    if not sjtu.exists():
        raise FileNotFoundError(f"Expected SEED/SJTU under {base}")
    return sjtu


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def label_indices() -> np.ndarray:
    return np.array([LABEL_TO_INDEX[int(v)] for v in SEED_LABELS], dtype=np.int64)


def subject_from_feature_file(path: str | Path) -> int:
    return int(Path(path).stem.split("_")[0])


def session_from_feature_file(path: str | Path) -> str:
    parts = Path(path).stem.split("_", 1)
    return parts[1] if len(parts) > 1 else ""


def feature_files(sjtu_root: str | Path) -> list[Path]:
    feat_dir = Path(sjtu_root) / "ExtractedFeatures_1s"
    return sorted(
        [p for p in feat_dir.glob("*.mat") if p.name.lower() != "label.mat"],
        key=lambda p: (subject_from_feature_file(p), session_from_feature_file(p)),
    )


def compact_tree(root: str | Path, max_files_per_dir: int = 8) -> str:
    root = Path(root)
    lines: list[str] = []

    def walk(path: Path, prefix: str = "") -> None:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        dirs = [p for p in entries if p.is_dir()]
        files = [p for p in entries if p.is_file()]
        for d in dirs:
            lines.append(f"{prefix}{d.name}/")
            walk(d, prefix + "  ")
        shown = files[:max_files_per_dir]
        for f in shown:
            size_mb = f.stat().st_size / (1024 * 1024)
            lines.append(f"{prefix}{f.name} ({size_mb:.2f} MB)")
        if len(files) > len(shown):
            lines.append(f"{prefix}... {len(files) - len(shown)} more files")

    lines.append(f"{root.name}/")
    walk(root, "  ")
    return "\n".join(lines)


def save_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def standardize_train_test(
    train_x: np.ndarray, test_x: np.ndarray, axes: Iterable[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if axes is None:
        axes = tuple(range(train_x.ndim - 1))
    train_stats = train_x.astype(np.float32, copy=False)
    mean = train_stats.mean(axis=tuple(axes), keepdims=True, dtype=np.float32)
    std = train_stats.std(axis=tuple(axes), keepdims=True, dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    macro_f1: float
    uar: float


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        uar=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    labels = [INDEX_TO_LABEL[i] for i in range(3)]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("SEED Emotion Confusion Matrix")
    fig.tight_layout()
    ensure_dir(Path(out_path).parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
