from __future__ import annotations

import argparse
from pathlib import Path

from context import AppContext
from db import count_vectors, find_nearest, insert_vector
from services.embedding import extract_dinov2_features_batch, load_image_paths
from services.roboflow import build_roboflow_preprocessor
from services.storage import make_s3_key, upload_to_s3


def ingest(ctx: AppContext, args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)

    print("=" * 60)
    print("Dataset ingest")
    print("author: Elijah Zhao")
    print(f"Input directory: {input_dir}")
    print(f"S3 bucket: {ctx.cfg.bucket_name}")
    print("=" * 60)

    image_paths = load_image_paths(input_dir)
    print(f"Found {len(image_paths)} image files.")

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

    features, valid_paths = extract_dinov2_features_batch(
        image_paths=image_paths,
        model_name=ctx.cfg.dinov2_model,
        batch_size=ctx.cfg.batch_size,
        preprocess_fn=preprocess_fn,
    )

    if len(valid_paths) == 0:
        raise RuntimeError("No valid images were embedded.")

    print(f"Embedded {len(valid_paths)} images.")
    print(f"COSINE_THRESHOLD={ctx.cfg.cosine_sim_threshold}")

    kept = 0
    skipped = 0
    errors = 0
    debug_every = 25

    with ctx.db.connection() as conn:
        with conn.cursor() as cur:
            for i, (image_path, vector_embedding) in enumerate(zip(valid_paths, features), start=1):
                try:
                    nearest_rows = find_nearest(cur, vector_embedding, k=1)
                    if nearest_rows:
                        nn_id, _nn_bucket, _nn_key, cosine_distance = nearest_rows[0]
                        cosine_similarity = 1.0 - float(cosine_distance)
                    else:
                        nn_id = None
                        cosine_similarity = -1.0

                    is_duplicate = (
                        nn_id is not None and cosine_similarity >= ctx.cfg.cosine_sim_threshold
                    )
                    if is_duplicate:
                        skipped += 1
                        continue

                    s3_key = make_s3_key(image_path)
                    upload_to_s3(ctx.s3, image_path, ctx.cfg.bucket_name, s3_key)
                    insert_vector(
                        cur,
                        bucket=ctx.cfg.bucket_name,
                        key=s3_key,
                        embedding_version=ctx.cfg.dinov2_model,
                        emb_norm=vector_embedding,
                    )
                    kept += 1

                    if i % debug_every == 0:
                        print(
                            f"[{i}/{len(valid_paths)}] kept={kept} skipped={skipped} errors={errors}"
                        )
                except Exception as exc:
                    errors += 1
                    print(f"ERROR [{i}] {image_path}: {exc}")

            db_count = count_vectors(cur)

    print("")
    print("Ingest complete.")
    print(f"Kept: {kept}")
    print(f"Skipped duplicates: {skipped}")
    print(f"Errors: {errors}")
    print(f"DB row count: {db_count}")
    return 0 if errors == 0 else 1
