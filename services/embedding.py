from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_image_paths(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Input directory not found: {root}")

    image_paths = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    image_paths = sorted(image_paths)
    if not image_paths:
        raise ValueError(f"No supported images found in {root}")
    return image_paths


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def extract_dinov2_features_batch(
    image_paths: list[Path],
    model_name: str,
    batch_size: int,
    preprocess_fn: Callable[[Image.Image, str], Image.Image] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    """Extract DINOv2 embeddings from local images in batches."""
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), []

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding device: {device} (torch CUDA available={torch.cuda.is_available()})")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    features: list[np.ndarray] = []
    valid_paths: list[Path] = []

    for batch_paths in tqdm(_batched(image_paths, batch_size), desc="Embedding"):
        tensors: list[torch.Tensor] = []
        ok_paths: list[Path] = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                if preprocess_fn is not None:
                    image = preprocess_fn(image, str(path))
                tensors.append(transform(image))
                ok_paths.append(path)
            except Exception as exc:
                print(f"Skipping unreadable image: {path} ({exc})")

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            output = model(batch).cpu().numpy().astype(np.float32)

        features.append(output)
        valid_paths.extend(ok_paths)

    if not features:
        return np.empty((0, 0), dtype=np.float32), []

    stacked = np.concatenate(features, axis=0)
    return l2_normalize(stacked), valid_paths
