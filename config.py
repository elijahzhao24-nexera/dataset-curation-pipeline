from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus


@dataclass(frozen=True)
class Config:
    db_url: str
    bucket_name: str
    aws_region: str
    cosine_sim_threshold: float
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    db_pool_min_size: int = 1
    db_pool_max_size: int = 5
    dinov2_model: str = "dinov2_vitb14"
    batch_size: int = 32
    use_bin_mask_for_embedding: bool = False
    roboflow_model_id: str | None = None
    roboflow_api_key: str | None = None
    roboflow_bin_class: str = "tote-bin"
    roboflow_bg: int = 0
    roboflow_pad: int = 10

    @classmethod
    def from_env(cls, dotenv_path: str | Path = ".env") -> "Config":
        _load_dotenv(dotenv_path)

        db_url = _resolve_db_url()
        bucket_name = _require_first("AWS_S3_BUCKET", "BUCKET_NAME")

        cfg = cls(
            db_url=db_url,
            bucket_name=bucket_name,
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            cosine_sim_threshold=float(_first_non_empty("COSINE_THRESHOLD", "COSINE_SIM_THRESHOLD") or "0.98"),
            aws_access_key_id=_first_non_empty("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=_first_non_empty("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=_first_non_empty("AWS_SESSION_TOKEN"),
            db_pool_min_size=int(os.getenv("DB_POOL_MIN_SIZE", "1")),
            db_pool_max_size=int(os.getenv("DB_POOL_MAX_SIZE", "5")),
            dinov2_model=os.getenv("DINOV2_MODEL", "dinov2_vitb14"),
            batch_size=int(os.getenv("BATCH_SIZE", "32")),
            use_bin_mask_for_embedding=_parse_bool(os.getenv("USE_BIN_MASK_FOR_EMBEDDING", "false")),
            roboflow_model_id=_first_non_empty("ROBOFLOW_MODEL_ID"),
            roboflow_api_key=_first_non_empty("ROBOFLOW_API_KEY"),
            roboflow_bin_class=os.getenv("ROBOFLOW_BIN_CLASS", "tote-bin"),
            roboflow_bg=int(os.getenv("ROBOFLOW_BG", "0")),
            roboflow_pad=int(os.getenv("ROBOFLOW_PAD", "10")),
        )

        if cfg.db_pool_min_size > cfg.db_pool_max_size:
            raise RuntimeError("DB_POOL_MIN_SIZE cannot be greater than DB_POOL_MAX_SIZE.")
        if cfg.use_bin_mask_for_embedding:
            if not cfg.roboflow_model_id:
                raise RuntimeError(
                    "ROBOFLOW_MODEL_ID is required when USE_BIN_MASK_FOR_EMBEDDING=true."
                )
            if not cfg.roboflow_api_key:
                raise RuntimeError(
                    "ROBOFLOW_API_KEY is required when USE_BIN_MASK_FOR_EMBEDDING=true."
                )

        return cfg


def _first_non_empty(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return None


def _require_first(*names: str) -> str:
    value = _first_non_empty(*names)
    if value is None:
        raise RuntimeError(f"Missing required environment variable. Expected one of: {', '.join(names)}")
    return value


def _resolve_db_url() -> str:
    direct = _first_non_empty("DB_URL")
    if direct:
        return direct

    host = _first_non_empty("PGHOST")
    port = _first_non_empty("PGPORT")
    dbname = _first_non_empty("PGDATABASE")
    user = _first_non_empty("PGUSER")
    password = _first_non_empty("PGPASSWORD")

    values = {
        "PGHOST": host,
        "PGPORT": port,
        "PGDATABASE": dbname,
        "PGUSER": user,
        "PGPASSWORD": password,
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "DB config missing. Set DB_URL or provide PG* vars. Missing: " + ", ".join(missing)
        )

    sslmode = _first_non_empty("PGSSLMODE")
    auth = f"{quote_plus(user)}:{quote_plus(password)}"
    base = f"postgresql://{auth}@{host}:{port}/{dbname}"
    if sslmode:
        return f"{base}?sslmode={quote_plus(sslmode)}"
    return base


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_dotenv(dotenv_path: str | Path) -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
