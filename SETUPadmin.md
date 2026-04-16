# Production Admin Setup (One-Time)

Use this file for admin-only tasks. Most teammates should use [SETUPDB.md](SETUPDB.md) instead.

## 1) When to Run This

Run this only if:

- production bucket/DB are not created yet
- you are rotating DB credentials
- you are migrating auth from access keys to SSO

If prod infra already exists, teammates do not re-run this.

## 2) Current Production Auth Model

Current setup is a single org AWS account with per-user IAM access keys (no shared key).

- Create one IAM user per teammate.
- Put users into a shared group with least-privilege S3 permissions.
- Keep DB password in password manager.

## 3) Provision Production Infra

### 3.1 S3 bucket

1. Create production S3 bucket.
2. Save `AWS_S3_BUCKET`, `AWS_REGION`.

### 3.2 VPC and security group IP whitelist

1. Create/choose production VPC.
2. Create RDS security group.
3. Restrict inbound `TCP 5432` to approved org IP CIDRs only.

### 3.3 RDS PostgreSQL

1. Create production PostgreSQL RDS in the VPC/security group from above.
2. Save `PGHOST`, `PGPORT`, master username/password.

## 4) IAM Policy, Group, and Users

Create a customer-managed IAM policy (replace `<actual bucket name>`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BucketList",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::<actual bucket name>"
    },
    {
      "Sid": "ObjectReadWrite",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:AbortMultipartUpload",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::<actual bucket name>/*"
    },
    {
      "Sid": "ViewVpcAndSecurityGroups",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeSecurityGroupRules",
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeRouteTables",
        "ec2:DescribeInternetGateways"
      ],
      "Resource": "*"
    },
    {
      "Sid": "EditSecurityGroupWhitelist",
      "Effect": "Allow",
      "Action": [
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:ModifySecurityGroupRules"
      ],
      "Resource": "*"
    }
  ]
}
```

Then:

1. Create IAM group for this pipeline and attach policy.
2. Create IAM user per person
3. Add users to the group.
4. Save login and access keys securely.
5. Do not share one IAM key across the team.

## 5) Bootstrap Database

Connect as master user:

```powershell
psql "host=<RDS_ENDPOINT> port=5432 dbname=postgres user=<MASTER_USER> password=<MASTER_PASSWORD> sslmode=require"
```

create schema:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.image_vectors (
  id BIGSERIAL PRIMARY KEY,
  s3_bucket TEXT NOT NULL,
  s3_key TEXT NOT NULL,
  embedding_version TEXT NOT NULL DEFAULT 'dinov2_vitb14',
  embedding vector(768) NOT NULL,
  CONSTRAINT image_vectors_s3_unique UNIQUE (s3_bucket, s3_key)
);

CREATE INDEX IF NOT EXISTS idx_image_vectors_embedding_hnsw
ON public.image_vectors
USING hnsw (embedding vector_cosine_ops);
```

If you use `PGUSER=app_user` in `.env`, create it and grant permissions:

```sql
CREATE ROLE app_user WITH LOGIN PASSWORD '<strong_password>';
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE ON TABLE public.image_vectors TO app_user;
GRANT USAGE, SELECT ON SEQUENCE public.image_vectors_id_seq TO app_user;
```

## 6) Secrets and Sharing

Store `PGPASSWORD` for `app_user` in NordPass (or your team password manager).

## 7) Future Migration (Optional)

### 7.1 Current mode

- Teammates use their own AWS access keys from IAM users/groups.
- Do not share one key across the team.

### 7.2 Target mode (after SSO is ready)

1. In IAM Identity Center, create permission set/role for this pipeline.
2. Minimum S3 permissions:
   - `s3:ListBucket` on `arn:aws:s3:::<bucket>`
   - `s3:GetObject`, `s3:PutObject` on `arn:aws:s3:::<bucket>/*`
3. Assign permission set to a group (not per-user where possible).
4. Share AWS account ID and SSO profile naming convention with team.
