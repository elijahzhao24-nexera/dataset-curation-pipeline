from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from context import AppContext
from db import find_nearest_by_prefix
from services.embedding import extract_dinov2_features_batch, load_image_paths
from services.roboflow import build_roboflow_preprocessor


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def retrieve_similar_images(
    ctx: AppContext,
    candidate_paths: list[Path],
    input_prefix: str,
    k: int,
) -> list[tuple[int, str, str, float]]:
    preprocess_fn = None
    if ctx.cfg.use_bin_mask_for_embedding:
        preprocess_fn = build_roboflow_preprocessor(
            roboflow_model_id=ctx.cfg.roboflow_model_id or "",
            roboflow_api_key=ctx.cfg.roboflow_api_key or "",
            class_name=ctx.cfg.roboflow_bin_class,
            pad=ctx.cfg.roboflow_pad,
            bg=ctx.cfg.roboflow_bg,
        )
        print("Roboflow preprocessing: enabled")
    else:
        print("Roboflow preprocessing: disabled")

    candidate_features, valid_candidate_paths = extract_dinov2_features_batch(
        image_paths=candidate_paths,
        model_name=ctx.cfg.dinov2_model,
        batch_size=ctx.cfg.batch_size,
        preprocess_fn=preprocess_fn,
    )

    if len(valid_candidate_paths) == 0:
        return []

    candidate_features = _l2_normalize(candidate_features)
    query = candidate_features.mean(axis=0)
    query = query / (np.linalg.norm(query) + 1e-12)

    with ctx.db.connection() as conn:
        with conn.cursor() as cur:
            rows = find_nearest_by_prefix(
                cur,
                emb_norm=query,
                bucket=ctx.cfg.bucket_name,
                prefix=input_prefix,
                k=k,
            )

    return rows


def download_matches(
    s3_client,
    matches: list[tuple[int, str, str, float]],
    output_dir: str,
) -> int:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for image_id, bucket, key, distance in matches:
        filename = Path(key).name
        if not filename:
            print(f"Warning: Could not extract filename from key: {key}. Skipping.")
            continue

        local_path = _build_unique_output_path(output_path, filename, int(image_id))
        print(
            f"Downloading id={image_id} dist={distance:.4f} s3://{bucket}/{key} -> {local_path}"
        )
        s3_client.download_file(bucket, key, str(local_path))
        downloaded += 1
    return downloaded


def _build_unique_output_path(output_dir: Path, filename: str, image_id: int) -> Path:
    candidate = output_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    dedup_candidate = output_dir / f"{stem}_{image_id}{suffix}"
    counter = 1
    while dedup_candidate.exists():
        dedup_candidate = output_dir / f"{stem}_{image_id}_{counter}{suffix}"
        counter += 1
    return dedup_candidate


def similar(ctx: AppContext, args: argparse.Namespace) -> int:
    candidate_dir = Path(args.candidates_folder)
    input_prefix = args.input_folder.strip("/")
    output_dir = args.output_folder
    k = args.k

    print("=" * 60)
    print("Select similar images")
    print("author: Elijah Zhao")
    print(f"Candidates directory (local): {candidate_dir}")
    print(f"Input folder (S3 prefix): {input_prefix}")
    print(f"Output directory (local): {output_dir}")
    print(f"S3 bucket: {ctx.cfg.bucket_name}")
    print("=" * 60)

    candidate_paths = load_image_paths(candidate_dir)
    matches = retrieve_similar_images(
        ctx=ctx,
        candidate_paths=candidate_paths,
        input_prefix=input_prefix,
        k=k,
    )

    if not matches:
        print("No similar images found.")
        return 1

    for row in matches:
        image_id, bucket, key, distance = row
        print(image_id, f"s3://{bucket}/{key}", "dist=", distance, "sim≈", (1.0 - float(distance)))

    downloaded = download_matches(ctx.s3, matches, output_dir=output_dir)
    print(f"Downloaded matches: {downloaded}")
    return 0
