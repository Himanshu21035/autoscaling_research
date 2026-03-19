# scripts/download_alibaba.py
"""
Downloads only the required Alibaba Microservices 2021 files.
Downloads MSRTQps (RPS proxy) and MSResource (replica count).
Skips MSCallGraph and Node entirely.

Usage: python scripts/download_alibaba.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import urllib.request
import tarfile
import time
from src.logger import get_logger

logger = get_logger("download_alibaba")

BASE_URL = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces"
RAW_DIR = Path("data/raw/alibaba")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a single file with retry logic."""
    if dest.exists():
        logger.info(f"Already exists, skipping: {dest.name}")
        return True

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading [{attempt}/{retries}]: {dest.name}")
            urllib.request.urlretrieve(url, dest)
            logger.info(f"Done: {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            time.sleep(5)

    logger.error(f"FAILED after {retries} attempts: {dest.name}")
    return False


def extract_tarball(tar_path: Path, extract_dir: Path):
    """Extract .tar.gz file."""
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        tar_path.unlink()   # delete archive after extracting to save space
        logger.info(f"Extracted + deleted: {tar_path.name}")
    except Exception as e:
        logger.error(f"Extraction failed for {tar_path.name}: {e}")


def download_msrtqps():
    """Download MSRTQps files (call_rate + response time per microservice per 30s)."""
    logger.info("=" * 50)
    logger.info("Downloading MSRTQps (0-24)...")
    out_dir = RAW_DIR / "MSRTQps"
    out_dir.mkdir(exist_ok=True)

    failed = []
    for i in range(25):   # files 0 to 24
        url = f"{BASE_URL}/MSRTQps/MSRTQps_{i}.tar.gz"
        dest = out_dir / f"MSRTQps_{i}.tar.gz"
        success = download_file(url, dest)
        if success:
            extract_tarball(dest, out_dir)
        else:
            failed.append(i)

    if failed:
        logger.warning(f"MSRTQps failed indices: {failed}")
    else:
        logger.info("MSRTQps: All 25 files downloaded and extracted ✓")


def download_msresource():
    """Download MSResource files (replica count per microservice)."""
    logger.info("=" * 50)
    logger.info("Downloading MSResource (0-11)...")
    out_dir = RAW_DIR / "MSResource"
    out_dir.mkdir(exist_ok=True)

    failed = []
    for i in range(12):   # files 0 to 11
        url = f"{BASE_URL}/MSResource/MSResource_{i}.tar.gz"
        dest = out_dir / f"MSResource_{i}.tar.gz"
        success = download_file(url, dest)
        if success:
            extract_tarball(dest, out_dir)
        else:
            failed.append(i)

    if failed:
        logger.warning(f"MSResource failed indices: {failed}")
    else:
        logger.info("MSResource: All 12 files downloaded and extracted ✓")


if __name__ == "__main__":
    logger.info("Starting Alibaba download — MSRTQps + MSResource only")
    logger.info("Skipping MSCallGraph (145 files) and Node — not needed")

    download_msrtqps()
    download_msresource()

    logger.info("Download complete. Files in data/raw/alibaba/")
    logger.info("Run: python scripts/prepare_data.py to process them")
