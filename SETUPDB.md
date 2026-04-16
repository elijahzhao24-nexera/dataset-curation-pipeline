# DB + S3 Access Setup (Org Users)

Use this file for normal team onboarding.
## TLDR

- DB password (`PGPASSWORD`) should come from NordPass (or your team password manager).
- Current production auth mode is per-user IAM access keys (either use shared user from nord pass, or request a custom one from manager).
- SSO remains a future migration option.

## 1) Before You Start

Get these values from your admin/handoff docs:

- `AWS_S3_BUCKET`
- `AWS_REGION`
- `PGPORT`
- `PGDATABASE`
- `PGUSER` (usually `postgres`)
- `PGPASSWORD` (from NordPass/password sharer)
- `PGHOST` 

Optional Roboflow ID and keys:
- `ROBOFLOW_MODEL_ID`
- `ROBOFLOW_API_KEY`

## 2) Current Mode: IAM Access Keys

Use your own IAM user key and secret from the IAM group. (get admin to make one for you, or use default one in nordpass)

```env
AWS_S3_BUCKET=<prod_bucket_name>
AWS_REGION=<aws_region>

AWS_ACCESS_KEY_ID=<your_key>
AWS_SECRET_ACCESS_KEY=<your_secret>
AWS_SESSION_TOKEN=<leave blank>
AWS_PROFILE=<leave blank>

PGHOST=<rds_endpoint>
PGPORT=5432
PGDATABASE=dataset_curation
PGUSER=app_user
PGPASSWORD=<from_nordpass>
PGSSLMODE=require
```

## 3) Connectivity Checks (Works for Both Modes)

These checks work with either:

- access keys in env, or
- valid SSO login + `AWS_PROFILE`

S3 check:

```powershell
python -c "import os,boto3; from config import Config; cfg=Config.from_env(); p=(os.getenv('AWS_PROFILE') or '').strip() or None; s=boto3.session.Session(profile_name=p, region_name=cfg.aws_region).client('s3'); s.head_bucket(Bucket=cfg.bucket_name); print('S3 OK')"
```

DB check:

```powershell
python -c "import psycopg; from config import Config; cfg=Config.from_env(); c=psycopg.connect(cfg.db_url); cur=c.cursor(); cur.execute('SELECT 1'); print('DB OK', cur.fetchone()[0]); c.close()"
```

CLI check:

```powershell
python cli.py --help
```

## 4) Optional Future: Migrate to SSO

When SSO is ready:

1. Run `aws configure sso`.
2. Run `aws sso login --profile <ORG_SSO_PROFILE>`.
3. Update `.env` to set `AWS_PROFILE` and clear `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`.
4. Re-run connectivity checks.
