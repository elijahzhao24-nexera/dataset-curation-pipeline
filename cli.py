from __future__ import annotations

import argparse

from config import Config
from context import AppContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset curation pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest-folder", help="Ingest local folder into DB/S3.")
    ingest.add_argument("--input-dir", required=True, help="Local source image directory.")
    ingest.add_argument("--output-folder", required=True, help="Destination folder prefix in bucket.")
    ingest.set_defaults(handler=cmd_ingest_folder)

    diverse = subparsers.add_parser("select-diverse", help="Select k diverse images from bucket folder.")
    diverse.add_argument("--k", required=True, type=_positive_int, help="Number of images to select.")
    diverse.add_argument("--input-folder", required=True, help="Input folder prefix in bucket.")
    diverse.add_argument("--output-folder", required=True, help="Local output directory for downloaded images.")
    diverse.set_defaults(handler=cmd_select_diverse)

    similar = subparsers.add_parser(
        "select-similar",
        help="Select k images similar to candidates from an input bucket folder.",
    )
    similar.add_argument("--k", required=True, type=_positive_int, help="Number of images to select.")
    similar.add_argument("--candidates-folder", required=True, help="Local candidate images directory.")
    similar.add_argument("--input-folder", required=True, help="Input folder prefix in bucket.")
    similar.add_argument("--output-folder", required=True, help="Local output directory for downloaded matches.")
    similar.set_defaults(handler=cmd_select_similar)

    return parser.parse_args()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return parsed


def cmd_ingest_folder(ctx: AppContext, args: argparse.Namespace) -> int:
    from commnds.ingest import ingest as ingest_command

    return ingest_command(ctx, args)


def cmd_select_diverse(ctx: AppContext, args: argparse.Namespace) -> int:
    print(
        f"[select-diverse] k={args.k} input-folder={args.input_folder} output-folder={args.output_folder} "
        f"bucket={ctx.cfg.bucket_name}"
    )

    from commnds.diverse import diverse

    return diverse(ctx, args)


def cmd_select_similar(ctx: AppContext, args: argparse.Namespace) -> int:
    print(
        f"[select-similar] k={args.k} candidates-folder={args.candidates_folder} "
        f"input-folder={args.input_folder} output-folder={args.output_folder} "
        f"bucket={ctx.cfg.bucket_name}"
    )
    from commnds.similar import similar
    return similar(ctx, args)


def main() -> int:
    args = parse_args()
    cfg = Config.from_env()
    context = AppContext(cfg)
    try:
        return args.handler(context, args)
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
