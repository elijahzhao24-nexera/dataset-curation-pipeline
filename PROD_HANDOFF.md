# Production Handoff Runbook

This runbook is for continuity when ownership changes.

- Infrastructure is manual in this phase.
- Production bucket and DB are provisioned once, then reused.
- Human access target is role-based through AWS IAM Identity Center (SSO).
- Interim mode before SSO is ready: per-user access keys are allowed temporarily (no shared keys).

## 1) Handoff Package (What to Share)

Share these operational values with maintainers:

- `AWS_S3_BUCKET`
- `AWS_REGION`
- `PGHOST`
- `PGPORT`
- `PGDATABASE`
- `PGUSER`
- `PGSSLMODE`
- AWS account ID
- Expected AWS profile name format (for example `dataset-curation-prod`)

## 2) Secret Storage and Ownership

Store production DB password (`PGPASSWORD`) in your company password vault.

## 3) Offboarding Checklist 

1. Confirm at least one maintainer have SSO access to prod role/permission set.
2. Confirm at least one maintainer can access DB password in password vault.
3. Transfer SSO/IAM Identity Center admin ownership to named maintainer(s).
4. Rotate DB password and update password-vault entry.
5. Verify maintainers can run:
   - `python cli.py --help`
   - S3 connectivity check from [SETUPDB.md](SETUPDB.md)
   - DB connectivity check from [SETUPDB.md](SETUPDB.md)
6. Remove your prod access after verification passes.

## 4) Interim Mode (Until SSO Is Ready)

This is temporary:

- each teammate uses their own access keys with least required permissions
- DB password is retrieved from NordPass/password manager
- do not use one shared IAM key across the team

After SSO goes live, migrate using [SETUPDB.md](SETUPDB.md) and retire old long-lived keys.
