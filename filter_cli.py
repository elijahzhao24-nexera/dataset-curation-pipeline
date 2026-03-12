#!/usr/bin/env python3
"""
Select images from an input directory using embeddings:
1) k diverse images (farthest-point sampling)
2) k similar images to a candidate directory (mean candidate embedding)

Example:
  python image_selector_cli.py ^
    --input-dir .\Input ^
    --candidate-dir .\Candidates ^
    --output-dir .\Output ^
    --k-diverse 50 ^
    --k-similar 20
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import models


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select k diverse images and k similar-to-candidates images."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing source images.")
    parser.add_argument(
        "--candidate-dir",
        default=None,
        help="Directory containing candidate images used as a similarity query.",
    )
    parser.add_argument("--output-dir", required=True, help="Output root directory.")
    parser.add_argument("--k-diverse", type=int, default=0, help="Number of diverse images to select.")
    parser.add_argument(
        "--k-similar",
        type=int,
        default=0,
        help="Number of images similar to candidates to select.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for diverse selection.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Embedding device. 'auto' uses CUDA if available.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow same image to be selected in both diverse and similar sets.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan top-level image files in directories.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device_arg


def list_images(folder: Path, recursive: bool = True) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Directory not found: {folder}")
    pattern = "**/*" if recursive else "*"
    paths = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def chunked(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_feature_model(device: str):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    backbone = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*(list(backbone.children())[:-1]))
    model.to(device).eval()
    preprocess = weights.transforms()
    return model, preprocess


@torch.no_grad()
def embed_images(
    image_paths: list[Path],
    model: torch.nn.Module,
    preprocess,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, list[Path]]:
    if not image_paths:
        return np.empty((0, 2048), dtype=np.float32), []

    features: list[np.ndarray] = []
    valid_paths: list[Path] = []

    for batch_paths in chunked(image_paths, batch_size):
        tensors = []
        ok_paths = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(preprocess(img))
                ok_paths.append(path)
            except Exception as exc:
                print(f"Skipping unreadable image: {path} ({exc})")

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        output = model(batch).flatten(1).cpu().numpy().astype(np.float32)
        features.append(output)
        valid_paths.extend(ok_paths)

    if not features:
        return np.empty((0, 2048), dtype=np.float32), []

    all_features = np.concatenate(features, axis=0)
    return l2_normalize(all_features), valid_paths


def farthest_point_sampling_cosine(features: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = features.shape[0]
    if k <= 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)
    selected = np.empty(k, dtype=int)
    selected[0] = int(rng.integers(0, n))
    min_dist = np.full(n, np.inf, dtype=np.float32)

    for i in range(1, k):
        last = features[selected[i - 1]]
        distances = 1.0 - (features @ last)
        min_dist = np.minimum(min_dist, distances)
        min_dist[selected[:i]] = -np.inf
        selected[i] = int(np.argmax(min_dist))

    return selected


def top_k_similar_to_candidates(
    base_features: np.ndarray,
    candidate_features: np.ndarray,
    k: int,
    excluded_indices: set[int] | None = None,
) -> list[tuple[int, float]]:
    if k <= 0 or len(base_features) == 0 or len(candidate_features) == 0:
        return []

    query = candidate_features.mean(axis=0)
    query = query / (np.linalg.norm(query) + 1e-12)
    scores = base_features @ query
    ranked = np.argsort(-scores)

    excluded_indices = excluded_indices or set()
    selected: list[tuple[int, float]] = []
    for idx in ranked:
        i = int(idx)
        if i in excluded_indices:
            continue
        selected.append((i, float(scores[i])))
        if len(selected) >= k:
            break
    return selected


def copy_selection(
    selected_indices: list[int],
    source_paths: list[Path],
    output_subdir: Path,
    subset_name: str,
    score_by_index: dict[int, float] | None = None,
) -> list[dict[str, str]]:
    output_subdir.mkdir(parents=True, exist_ok=True)
    score_by_index = score_by_index or {}
    rows: list[dict[str, str]] = []

    for rank, idx in enumerate(selected_indices, start=1):
        src = source_paths[idx]
        dst_name = f"{rank:04d}_{src.name}"
        dst = output_subdir / dst_name
        shutil.copy2(src, dst)
        rows.append(
            {
                "subset": subset_name,
                "rank": str(rank),
                "score": f"{score_by_index.get(idx, '')}",
                "source_path": str(src),
                "output_path": str(dst),
            }
        )
    return rows


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    if not rows:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subset", "rank", "score", "source_path", "output_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    if args.k_diverse < 0 or args.k_similar < 0:
        raise ValueError("--k-diverse and --k-similar must be >= 0")
    if args.k_diverse == 0 and args.k_similar == 0:
        raise ValueError("Set at least one of --k-diverse or --k-similar.")
    if args.k_similar > 0 and not args.candidate_dir:
        raise ValueError("--candidate-dir is required when --k-similar > 0.")

    recursive = not args.non_recursive
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    candidate_dir = Path(args.candidate_dir) if args.candidate_dir else None
    device = resolve_device(args.device)

    print(f"Using device: {device}")
    print("Loading model...")
    model, preprocess = build_feature_model(device)

    print(f"Scanning input images: {input_dir}")
    input_paths = list_images(input_dir, recursive=recursive)
    if not input_paths:
        raise ValueError("No input images found.")
    print(f"Found {len(input_paths)} input images.")

    print("Embedding input images...")
    input_features, valid_input_paths = embed_images(
        input_paths, model, preprocess, device, args.batch_size
    )
    if len(valid_input_paths) == 0:
        raise ValueError("No valid input images could be embedded.")
    print(f"Embedded {len(valid_input_paths)} input images.")

    all_rows: list[dict[str, str]] = []

    diverse_k = min(args.k_diverse, len(valid_input_paths))
    diverse_idx = farthest_point_sampling_cosine(input_features, diverse_k, args.seed)
    if diverse_k > 0:
        print(f"Selecting {diverse_k} diverse images...")
        all_rows.extend(
            copy_selection(
                selected_indices=[int(i) for i in diverse_idx],
                source_paths=valid_input_paths,
                output_subdir=output_dir / "diverse",
                subset_name="diverse",
            )
        )

    similar_idx: list[int] = []
    similar_score_by_index: dict[int, float] = {}
    if args.k_similar > 0:
        print(f"Scanning candidate images: {candidate_dir}")
        candidate_paths = list_images(candidate_dir, recursive=recursive)
        if not candidate_paths:
            raise ValueError("No candidate images found.")
        print(f"Found {len(candidate_paths)} candidate images.")

        print("Embedding candidate images...")
        candidate_features, valid_candidate_paths = embed_images(
            candidate_paths, model, preprocess, device, args.batch_size
        )
        if len(valid_candidate_paths) == 0:
            raise ValueError("No valid candidate images could be embedded.")
        print(f"Embedded {len(valid_candidate_paths)} candidate images.")

        excluded = set(int(i) for i in diverse_idx) if not args.allow_overlap else set()
        selected = top_k_similar_to_candidates(
            base_features=input_features,
            candidate_features=candidate_features,
            k=args.k_similar,
            excluded_indices=excluded,
        )
        similar_idx = [idx for idx, _score in selected]
        similar_score_by_index = {idx: score for idx, score in selected}
        print(f"Selecting {len(similar_idx)} similar images...")
        all_rows.extend(
            copy_selection(
                selected_indices=similar_idx,
                source_paths=valid_input_paths,
                output_subdir=output_dir / "similar_to_candidates",
                subset_name="similar_to_candidates",
                score_by_index=similar_score_by_index,
            )
        )

    manifest = output_dir / "selection_manifest.csv"
    write_manifest(all_rows, manifest)

    print("")
    print("Done.")
    print(f"Diverse selected: {len(diverse_idx)}")
    print(f"Similar selected: {len(similar_idx)}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)