from __future__ import annotations

from pathlib import Path


def upload_to_s3(s3_client, local_path: str | Path, bucket: str, key: str) -> None:
    s3_client.upload_file(str(local_path), bucket, key)


def make_s3_key(local_path: str | Path) -> str:
    name = Path(local_path).name
    return name


def copy_s3_object(
    s3_client,
    src_bucket: str,
    src_key: str,
    dst_bucket: str,
    dst_key: str,
) -> None:
    s3_client.copy_object(
        Bucket=dst_bucket,
        Key=dst_key,
        CopySource={"Bucket": src_bucket, "Key": src_key},
    )
