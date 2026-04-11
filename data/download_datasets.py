"""
Dataset download and setup scripts.

- MELD: Publicly available, auto-downloadable from GitHub.
- AffectNet: Requires manual registration. This script provides instructions
  and verifies the directory structure after manual download.

Usage:
    python data/download_datasets.py --dataset meld
    python data/download_datasets.py --dataset affectnet --verify
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


# ============================================================
# MELD Download
# ============================================================

MELD_URLS = {
    "train": "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/data/MELD.Raw.tar.gz",
    "csv_train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "csv_dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "csv_test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}


def download_file(url: str, dest_path: str, desc: str = "Downloading"):
    """Download a file with progress bar."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  Already exists: {dest_path}")
        return

    print(f"  {desc}: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit="iB", unit_scale=True, desc=desc
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)


def setup_meld(root_dir: str = "./datasets/meld"):
    """
    Download and set up the MELD dataset.

    Downloads CSV annotation files and provides instructions for video data.
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MELD Dataset Setup")
    print("=" * 60)

    # Download CSV files
    for split in ["train", "dev", "test"]:
        split_dir = root / split
        split_dir.mkdir(exist_ok=True)

        csv_key = f"csv_{split}"
        if csv_key in MELD_URLS:
            csv_name = f"{split}_sent_emo.csv"
            download_file(
                MELD_URLS[csv_key],
                split_dir / csv_name,
                desc=f"Downloading {split} CSV",
            )

    print()
    print("=" * 60)
    print("CSV annotation files downloaded successfully!")
    print("=" * 60)
    print()
    print("IMPORTANT: Video files need to be downloaded separately.")
    print("Options:")
    print()
    print("Option 1 — From GitHub (recommended):")
    print("  git clone https://github.com/declare-lab/MELD.git /tmp/meld_repo")
    print(f"  cp -r /tmp/meld_repo/data/MELD/train_splits {root}/train/train_splits")
    print(f"  cp -r /tmp/meld_repo/data/MELD/dev_splits_complete {root}/dev/dev_splits_complete")
    print(f"  cp -r /tmp/meld_repo/data/MELD/output_repeated_splits_test {root}/test/output_repeated_splits_test")
    print()
    print("Option 2 — From HuggingFace:")
    print("  Visit: https://huggingface.co/datasets/declare-lab/MELD")
    print(f"  Download and extract video tar.gz files to {root}/")
    print()
    print("Option 3 — Text-only mode:")
    print("  If you only want to work with text (no video/face), the CSV files")
    print("  are sufficient. The model will use a blank face placeholder.")
    print()

    # Verify structure
    verify_meld(root_dir)


def verify_meld(root_dir: str = "./datasets/meld"):
    """Verify MELD directory structure."""
    root = Path(root_dir)

    checks = {
        "Train CSV": root / "train" / "train_sent_emo.csv",
        "Dev CSV": root / "dev" / "dev_sent_emo.csv",
        "Test CSV": root / "test" / "test_sent_emo.csv",
        "Train videos": root / "train" / "train_splits",
        "Dev videos": root / "dev" / "dev_splits_complete",
        "Test videos": root / "test" / "output_repeated_splits_test",
    }

    print("\nMELD Verification:")
    all_ok = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  [{status}] {name}: {path}")
        if not exists and "CSV" in name:
            all_ok = False

    if all_ok:
        # Count samples per split
        for split in ["train", "dev", "test"]:
            import pandas as pd

            csv_path = root / split / f"{split}_sent_emo.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                print(f"  {split}: {len(df)} utterances")

    return all_ok


# ============================================================
# AffectNet Verification
# ============================================================

def setup_affectnet(root_dir: str = "./datasets/affectnet"):
    """
    Provide instructions for AffectNet setup.

    AffectNet cannot be auto-downloaded — it requires manual registration.
    """
    print("=" * 60)
    print("AffectNet Dataset Setup")
    print("=" * 60)
    print()
    print("AffectNet requires MANUAL registration and download.")
    print()
    print("Steps:")
    print("  1. Visit: http://mohammadmahoor.com/affectnet/")
    print("  2. Fill out the registration form")
    print("  3. Wait for download link (usually 1-3 days)")
    print("  4. Download the dataset (~120GB)")
    print(f"  5. Extract to: {root_dir}/")
    print()
    print("Expected structure after extraction:")
    print(f"  {root_dir}/")
    print("  ├── Manually_Annotated/")
    print("  │   ├── Manually_Annotated_Images/")
    print("  │   │   ├── 1/")
    print("  │   │   ├── 2/")
    print("  │   │   └── ...")
    print("  │   └── Manually_Annotated_file_lists/")
    print("  │       ├── training.csv")
    print("  │       └── validation.csv")
    print("  └── Automatically_Annotated/ (optional)")
    print()

    verify_affectnet(root_dir)


def verify_affectnet(root_dir: str = "./datasets/affectnet"):
    """Verify AffectNet directory structure."""
    root = Path(root_dir)

    checks = {
        "Root directory": root,
        "Manual annotations": root / "Manually_Annotated",
        "Images directory": root / "Manually_Annotated" / "Manually_Annotated_Images",
        "Training CSV": root / "Manually_Annotated" / "Manually_Annotated_file_lists" / "training.csv",
        "Validation CSV": root / "Manually_Annotated" / "Manually_Annotated_file_lists" / "validation.csv",
    }

    print("\nAffectNet Verification:")
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  [{status}] {name}: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and verify datasets for MMER-XAI"
    )
    parser.add_argument(
        "--dataset",
        choices=["meld", "affectnet", "all"],
        default="all",
        help="Dataset to download/verify",
    )
    parser.add_argument(
        "--root", type=str, default="./datasets",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only verify existing downloads (don't download)",
    )
    args = parser.parse_args()

    if args.dataset in ("meld", "all"):
        meld_dir = os.path.join(args.root, "meld")
        if args.verify:
            verify_meld(meld_dir)
        else:
            setup_meld(meld_dir)

    if args.dataset in ("affectnet", "all"):
        affectnet_dir = os.path.join(args.root, "affectnet")
        if args.verify:
            verify_affectnet(affectnet_dir)
        else:
            setup_affectnet(affectnet_dir)


if __name__ == "__main__":
    main()
