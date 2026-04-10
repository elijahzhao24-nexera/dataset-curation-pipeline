from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from context import AppContext
from db import fetch_vectors_by_prefix


def _coerce_embedding(raw_embedding) -> np.ndarray:
    if isinstance(raw_embedding, np.ndarray):
        return raw_embedding.astype(np.float32, copy=False).ravel()
    if isinstance(raw_embedding, str):
        text = raw_embedding.strip().strip("[]")
        if not text:
            return np.empty((0,), dtype=np.float32)
        return np.fromstring(text, sep=",", dtype=np.float32)
    return np.asarray(raw_embedding, dtype=np.float32).ravel()


def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    if k >= len(points):
        return np.arange(len(points), dtype=int)

    farthest_indices = np.zeros(k, dtype=int)
    farthest_indices[0] = random.randint(0, len(points) - 1)
    distances = np.full(len(points), np.inf, dtype=np.float32)

    for i in range(1, k):
        last_chosen = points[farthest_indices[i - 1]]
        new_distances = np.sum((points - last_chosen) ** 2, axis=1)
        distances = np.minimum(distances, new_distances)
        farthest_indices[i] = int(np.argmax(distances))

    return farthest_indices


def download_s3_uri(
    s3_client,
    bucket: str,
    output_dir: str,
    selected_ids: list[list[str | int]],
) -> int:
    """Download selected [id, s3_uri] directly into output_dir (flat layout)."""
    base_directory = Path(output_dir)
    base_directory.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for image_id, s3_uri in selected_ids:
        try:
            key_parts = str(s3_uri).split(f"s3://{bucket}/")
            if len(key_parts) < 2:
                print(f"Warning: Could not parse S3 URI: {s3_uri}. Skipping.")
                continue

            key = key_parts[1]
            filename = Path(key).name
            if not filename:
                print(f"Warning: Could not extract filename from key: {key}. Skipping.")
                continue

            local_path = _build_unique_output_path(base_directory, filename, int(image_id))
            print(f"Downloading {s3_uri} -> {local_path}")
            s3_client.download_file(bucket, key, str(local_path))
            downloaded += 1
        except Exception as exc:
            print(f"Error downloading {s3_uri}: {exc}")

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


def diverse(ctx: AppContext, args: argparse.Namespace) -> int:
    input_prefix = args.input_folder.strip("/")
    output_dir = args.output_folder
    k_samples = args.k

    print("=" * 60)
    print("Select diverse images")
    print("author: Elijah Zhao")
    print(f"Input folder (S3 prefix): {input_prefix}")
    print(f"Output directory (local): {output_dir}")
    print(f"S3 bucket: {ctx.cfg.bucket_name}")
    print("=" * 60)

    with ctx.db.connection() as conn:
        with conn.cursor() as cur:
            rows = fetch_vectors_by_prefix(
                cur,
                bucket=ctx.cfg.bucket_name,
                prefix=input_prefix,
            )

    if not rows:
        print("No vectors found for input prefix.")
        return 1

    ids: list[int] = []
    s3_uris: list[str] = []
    vectors: list[np.ndarray] = []
    bad_rows = 0

    for row in rows:
        row_id, bucket, s3_key, embedding = row
        emb = _coerce_embedding(embedding)
        if emb.size == 0:
            bad_rows += 1
            continue
        ids.append(int(row_id))
        s3_uris.append(f"s3://{bucket}/{s3_key}")
        vectors.append(emb)

    if not vectors:
        print("No valid embeddings found for input prefix.")
        return 1

    feature_matrix = np.stack(vectors, axis=0)
    if len(feature_matrix) < k_samples:
        print(
            f"Warning: Not enough vectors ({len(feature_matrix)}) for {k_samples} samples. ")
        indices = np.arange(len(feature_matrix), dtype=int)
    else:
        indices = farthest_point_sampling(feature_matrix, k_samples)

    selected_ids = [[ids[int(i)], s3_uris[int(i)]] for i in indices]

    print(f"Total candidates: {len(feature_matrix)}")
    print(f"Rows skipped (bad embeddings): {bad_rows}")
    print(f"Selected: {len(selected_ids)}")

    downloaded = download_s3_uri(
        s3_client=ctx.s3,
        bucket=ctx.cfg.bucket_name,
        output_dir=output_dir,
        selected_ids=selected_ids,
    )

    print("Diverse selection complete.")
    print(f"Downloaded objects: {downloaded}")
    return 0
