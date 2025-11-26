#!/usr/bin/env python3
"""Download and manage benchmark datasets."""

import io
import zipfile
from pathlib import Path

import requests
import yaml


def load_manifest(manifest_path: str = "benchmark/datasets.yaml") -> dict:
    """Load the datasets manifest."""
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def download_dataset(dataset_id: str, force: bool = False) -> Path:
    """Download and extract a dataset from the manifest.

    Args:
        dataset_id: ID of the dataset in the manifest (e.g., 'hdfs_v1')
        force: If True, re-download even if already exists

    Returns:
        Path to the extracted dataset directory
    """
    manifest = load_manifest()

    if dataset_id not in manifest["datasets"]:
        available = ", ".join(manifest["datasets"].keys())
        raise ValueError(f"Unknown dataset '{dataset_id}'. Available: {available}")

    dataset = manifest["datasets"][dataset_id]
    extract_path = Path(dataset["extract_to"])

    # Check if already exists
    log_file = extract_path / dataset["log_file"]
    labels_file = extract_path / dataset["labels_file"]

    if log_file.exists() and labels_file.exists() and not force:
        print(f"Dataset '{dataset['name']}' already exists at {extract_path}")
        print("Use --force to re-download")
        return extract_path

    # Create parent directory
    extract_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"Downloading {dataset['name']} from {dataset['url']}...")
    print("  (This may take a few minutes)")

    response = requests.get(dataset["url"], stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunks = []

    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            chunks.append(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\r  Downloaded: {downloaded / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="")

    print()  # New line after progress

    # Extract
    print(f"Extracting to {extract_path}...")
    zip_data = b"".join(chunks)

    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        z.extractall(extract_path)

    print(f"Dataset '{dataset['name']}' ready at {extract_path}")

    return extract_path


def main():
    """CLI for dataset management."""
    import argparse

    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("dataset", help="Dataset ID (e.g., 'hdfs_v1')")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    if args.list:
        manifest = load_manifest()
        print("Available datasets:")
        for dataset_id, info in manifest["datasets"].items():
            print(f"  {dataset_id}: {info['name']}")
            print(f"    Lines: {info['total_lines']:,}")
            print(f"    Anomaly rate: {info['anomaly_rate']:.2%}")
        return

    download_dataset(args.dataset, force=args.force)


if __name__ == "__main__":
    main()
