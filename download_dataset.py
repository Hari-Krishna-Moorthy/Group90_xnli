"""XNLI Dataset Downloader and Local Cache Manager

This script checks whether the XNLI dataset exists locally.
- If it exists, it loads and verifies the local dataset without re-downloading.
- If it does not exist, it downloads the dataset from Hugging Face Hub
  (English train/validation + all multilingual test sets), formats the splits
  with language tags, and saves it locally using `save_to_disk`.

Supports Hugging Face authentication tokens via the HF_TOKEN environment variable
or --token CLI argument to prevent unauthenticated Hub warnings and rate limits.
"""

import argparse
import os
import sys
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from huggingface_hub import hf_hub_download

# All 15 official languages supported in XNLI benchmark
ALL_XNLI_LANGUAGES = [
    "ar", "bg", "de", "el", "en", "es", "fr",
    "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"
]

REPO_ID = "facebook/xnli"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xnli")


def get_hf_token(cli_token=None):
    """Retrieve HF token from CLI argument or HF_TOKEN environment variable."""
    token = cli_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[INFO] HF_TOKEN is not set. Proceeding with public access.")
        print("       Tip: export HF_TOKEN='your_hf_token' to suppress Hub warnings and enable higher rate limits.")
    else:
        print("[INFO] Authenticated with Hugging Face token.")
    return token


def is_local_dataset_valid(output_dir):
    """Check if a valid dataset exists at output_dir."""
    if not os.path.exists(output_dir):
        return False
    try:
        ds = load_from_disk(output_dir)
        required_splits = {"train", "validation", "test"}
        if isinstance(ds, DatasetDict) and required_splits.issubset(set(ds.keys())):
            return True
        return False
    except Exception:
        return False


def download_xnli_if_needed(output_dir=DEFAULT_OUTPUT_DIR, languages=None, token=None, force=False):
    """Check if XNLI exists locally; download and cache if missing.

    Args:
        output_dir: Path to directory where the dataset should be stored.
        languages: List of language codes to include in the test set (default: all 15).
        token: Optional Hugging Face authentication token.
        force: If True, re-downloads even if already present locally.

    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits.
    """
    token = get_hf_token(token)
    os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)

    if not force and is_local_dataset_valid(output_dir):
        print(f"[INFO] Dataset already exists locally at: {output_dir}")
        print("[INFO] Loading dataset from local disk...")
        ds = load_from_disk(output_dir)
        print(f"[SUCCESS] Loaded local dataset: {ds}")
        return ds

    if languages is None:
        languages = ALL_XNLI_LANGUAGES

    print("=" * 70)
    print("XNLI Dataset Downloader")
    print(f"Target directory: {output_dir}")
    print(f"Languages: {languages}")
    print("=" * 70)

    # 1. Download English train and validation splits
    print("\n[Step 1/3] Downloading English training and validation splits...")
    train_file = hf_hub_download(
        repo_id=REPO_ID,
        filename="en/train-00000-of-00001.parquet",
        repo_type="dataset",
        token=token,
    )
    val_file = hf_hub_download(
        repo_id=REPO_ID,
        filename="en/validation-00000-of-00001.parquet",
        repo_type="dataset",
        token=token,
    )

    train_ds = Dataset.from_parquet(train_file)
    val_ds = Dataset.from_parquet(val_file)

    def add_en_tag(batch):
        batch["language"] = ["en"] * len(batch["premise"])
        return batch

    train_ds = train_ds.map(add_en_tag, batched=True, desc="Tagging English train")
    val_ds = val_ds.map(add_en_tag, batched=True, desc="Tagging English validation")

    # 2. Download test splits directly per language
    print(f"\n[Step 2/3] Downloading test splits for {len(languages)} languages: {', '.join(languages)}...")
    test_splits = []
    for lang in languages:
        print(f"  -> Downloading test split for: {lang}")
        test_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{lang}/test-00000-of-00001.parquet",
            repo_type="dataset",
            token=token,
        )
        lang_ds = Dataset.from_parquet(test_file)

        def add_lang_tag(batch, l=lang):
            batch["language"] = [l] * len(batch["premise"])
            return batch

        lang_ds = lang_ds.map(add_lang_tag, batched=True, desc=f"Tagging test ({lang})")
        test_splits.append(lang_ds)

    test_ds = concatenate_datasets(test_splits)

    # 3. Save as DatasetDict to local disk
    print(f"\n[Step 3/3] Saving final DatasetDict to local disk at: {output_dir}")
    final_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    final_dict.save_to_disk(output_dir)
    print(f"[SUCCESS] Dataset successfully saved to {output_dir}!")
    print(f"Summary:")
    print(f"  - Train examples:      {len(final_dict['train']):,} (English)")
    print(f"  - Validation examples: {len(final_dict['validation']):,} (English)")
    print(f"  - Test examples:       {len(final_dict['test']):,} across {len(languages)} languages ({len(final_dict['test']) // len(languages)} per language)")
    return final_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Download and locally cache XNLI dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target directory to save the local dataset (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes (e.g. 'en,fr,es,de,hi,zh') or omit for all 15 languages.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token (or set HF_TOKEN environment variable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset already exists locally.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    langs = [l.strip() for l in args.languages.split(",")] if args.languages else ALL_XNLI_LANGUAGES
    download_xnli_if_needed(output_dir=args.output_dir, languages=langs, token=args.token, force=args.force)
