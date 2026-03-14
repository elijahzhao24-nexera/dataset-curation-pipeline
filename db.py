from __future__ import annotations

from typing import Any

import numpy as np


NEAREST_K_SQL = """
SELECT id, s3_bucket, s3_key, (embedding <=> %s::vector) AS cosine_distance
FROM public.image_vectors
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

NEAREST_K_BY_PREFIX_SQL = """
SELECT id, s3_bucket, s3_key, (embedding <=> %s::vector) AS cosine_distance
FROM public.image_vectors
WHERE s3_bucket = %s
  AND s3_key LIKE %s
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

INSERT_SQL = """
INSERT INTO public.image_vectors (s3_bucket, s3_key, embedding_version, embedding)
VALUES (%s, %s, %s, %s::vector)
ON CONFLICT (s3_bucket, s3_key)
DO UPDATE SET
  embedding_version = EXCLUDED.embedding_version,
  embedding = EXCLUDED.embedding,
  active = true;
"""

FETCH_VECTORS_BY_PREFIX_SQL = """
SELECT id, s3_bucket, s3_key, embedding
FROM public.image_vectors
WHERE s3_bucket = %s
  AND s3_key LIKE %s
ORDER BY id;
"""


def to_pgvector_literal(x: np.ndarray) -> str:
    vector = np.asarray(x, dtype=np.float32).ravel()
    return "[" + ",".join(map(str, vector.tolist())) + "]"


def find_nearest(cursor: Any, emb_norm: np.ndarray, k: int = 1) -> list[tuple[Any, ...]]:
    vector_literal = to_pgvector_literal(emb_norm)
    cursor.execute(NEAREST_K_SQL, (vector_literal, vector_literal, k))
    return cursor.fetchall()


def find_nearest_by_prefix(
    cursor: Any,
    emb_norm: np.ndarray,
    bucket: str,
    prefix: str,
    k: int,
) -> list[tuple[Any, ...]]:
    vector_literal = to_pgvector_literal(emb_norm)
    normalized_prefix = prefix.strip("/")
    like_pattern = "%" if not normalized_prefix else f"{normalized_prefix}/%"
    cursor.execute(
        NEAREST_K_BY_PREFIX_SQL,
        (vector_literal, bucket, like_pattern, vector_literal, k),
    )
    return cursor.fetchall()


def insert_vector(
    cursor: Any,
    bucket: str,
    key: str,
    embedding_version: str,
    emb_norm: np.ndarray,
) -> None:
    vector_literal = to_pgvector_literal(emb_norm)
    cursor.execute(INSERT_SQL, (bucket, key, embedding_version, vector_literal))


def count_vectors(cursor: Any) -> int:
    cursor.execute("SELECT COUNT(*) FROM public.image_vectors;")
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def fetch_vectors_by_prefix(
    cursor: Any,
    bucket: str,
    prefix: str,
) -> list[tuple[Any, ...]]:
    normalized_prefix = prefix.strip("/")
    like_pattern = "%" if not normalized_prefix else f"{normalized_prefix}/%"
    cursor.execute(FETCH_VECTORS_BY_PREFIX_SQL, (bucket, like_pattern))
    return cursor.fetchall()
