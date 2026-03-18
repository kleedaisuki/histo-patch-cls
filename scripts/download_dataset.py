#!/usr/bin/env python3
"""Download a Kaggle dataset into the project's data/raw directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub


DEFAULT_DATASET = "paultimothymooney/breast-histopathology-images"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset to data/raw using kagglehub."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Kaggle dataset handle, e.g. owner/dataset-name.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Target directory. Defaults to <repo>/data/raw.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    return parser.parse_args()


def resolve_output_dir(raw_output_dir: str | None) -> Path:
    """Resolve output directory path."""
    if raw_output_dir:
        return Path(raw_output_dir).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "data" / "raw").resolve()


def main() -> None:
    """Run dataset download.

    Note:
        Kaggle credentials must be configured first
        (KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json).
    """
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = kagglehub.dataset_download(
        args.dataset,
        force_download=args.force_download,
        output_dir=str(output_dir),
    )

    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Downloaded path: {downloaded_path}")


if __name__ == "__main__":
    main()
