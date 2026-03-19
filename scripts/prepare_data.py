# scripts/prepare_data.py
"""
Run this once to process all raw datasets into cleaned, split CSVs.
Usage: python scripts/prepare_data.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import load_burstgpt, load_azure, load_alibaba
from src.data.cleaner import clean_pipeline
from src.data.splitter import split_by_days, split_by_ratio
from src.logger import get_logger
from src.config import CONFIG

logger = get_logger("prepare_data")
PROCESSED_DIR = Path(CONFIG["data"]["processed_dir"])
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_split(split, name: str):
    """Save train/val/test splits to CSV."""
    for part in ["train", "val", "test"]:
        df = getattr(split, part)
        out_path = PROCESSED_DIR / f"{name}_{part}.csv"
        df.to_csv(out_path)
        logger.info(f"Saved: {out_path} ({len(df):,} rows)")


def process_burstgpt():
    logger.info("=" * 50)
    logger.info("Processing BurstGPT...")
    raw = load_burstgpt()
    cleaned = clean_pipeline(raw)
    cleaned.to_csv(PROCESSED_DIR / "burstgpt_full.csv")

    # BurstGPT has 121 days → use 60/14/14 day split
    split = split_by_days(cleaned, train_days=60, val_days=14, test_days=14)
    save_split(split, "burstgpt")
    logger.info("BurstGPT done ✓")
    return cleaned


def process_azure():
    logger.info("=" * 50)
    logger.info("Processing Azure LLM 2024...")
    raw = load_azure()
    cleaned = clean_pipeline(raw)
    cleaned.to_csv(PROCESSED_DIR / "azure_full.csv")

    # Azure has only ~10 days → use 70/15/15 ratio split
    split = split_by_ratio(cleaned, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    save_split(split, "azure")
    logger.info("Azure done ✓")
    return cleaned


def process_alibaba():
    logger.info("=" * 50)
    logger.info("Processing Alibaba Microservices 2021...")
    try:
        raw = load_alibaba()
        cleaned = clean_pipeline(raw)
        cleaned.to_csv(PROCESSED_DIR / "alibaba_full.csv")
        split = split_by_ratio(cleaned)
        save_split(split, "alibaba")
        logger.info("Alibaba done ✓")
        return cleaned
    except FileNotFoundError as e:
        logger.warning(f"Alibaba skipped — {e}")
        return None


if __name__ == "__main__":
    logger.info("Starting data preparation pipeline...")
    process_burstgpt()
    process_azure()
    process_alibaba()
    logger.info("All datasets processed. Files saved to data/processed/")
