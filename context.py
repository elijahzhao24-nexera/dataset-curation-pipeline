from __future__ import annotations

from typing import Any

from config import Config


class AppContext:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._db = None
        self._s3 = None

    @property
    def db(self) -> Any:
        if self._db is None:
            self._db = make_db_pool(
                db_url=self.cfg.db_url,
                min_size=self.cfg.db_pool_min_size,
                max_size=self.cfg.db_pool_max_size,
            )
        return self._db

    @property
    def s3(self) -> Any:
        if self._s3 is None:
            self._s3 = make_s3_client(
                aws_region=self.cfg.aws_region,
                aws_access_key_id=self.cfg.aws_access_key_id,
                aws_secret_access_key=self.cfg.aws_secret_access_key,
                aws_session_token=self.cfg.aws_session_token,
            )
        return self._s3

    def close(self) -> None:
        if self._db is not None and hasattr(self._db, "close"):
            self._db.close()

    def __enter__(self) -> "AppContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def make_db_pool(db_url: str, min_size: int, max_size: int) -> Any:
    try:
        from psycopg_pool import ConnectionPool
    except ImportError as exc:
        raise RuntimeError(
            "psycopg_pool is required for DB access. Install with: pip install psycopg[binary] psycopg_pool"
        ) from exc

    return ConnectionPool(
        conninfo=db_url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": True},
    )


def make_s3_client(
    aws_region: str,
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
    aws_session_token: str | None,
) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 access. Install with: pip install boto3") from exc

    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=aws_region,
    )
    return session.client("s3")
