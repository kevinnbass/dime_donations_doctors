"""
Download utilities for DIME data files.

Handles scraping download links from the DIME data page and
downloading cycle contribution files with progress tracking.
"""

import html
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

from .config import DIME_RAW_DIR, PROCESSED_DATA_DIR, CYCLES


# DIME data page URL
DIME_DATA_URL = "https://data.stanford.edu/dime"

# Cache file for download links
DOWNLOAD_LINKS_CACHE = DIME_RAW_DIR / "download_links.json"

# Chunk size for downloads (1MB)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def scrape_dime_links(force_refresh: bool = False) -> dict:
    """
    Scrape the DIME data page to extract download links.

    Caches results to avoid repeated scraping.

    Args:
        force_refresh: Force re-scraping even if cache exists

    Returns:
        Dictionary with keys:
            - 'contributors': URL for contributors file
            - 'recipients': URL for recipients file
            - 'cycles': dict mapping year to contribDB URL
    """
    # Check cache first
    if DOWNLOAD_LINKS_CACHE.exists() and not force_refresh:
        print(f"Loading cached download links from {DOWNLOAD_LINKS_CACHE}")
        with open(DOWNLOAD_LINKS_CACHE, "r") as f:
            return json.load(f)

    print(f"Scraping download links from {DIME_DATA_URL}...")

    try:
        response = requests.get(DIME_DATA_URL, timeout=30)
        response.raise_for_status()
        page_html = response.text
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch DIME data page: {e}")

    links = {
        "contributors": None,
        "recipients": None,
        "cycles": {},
    }

    def clean_url(url: str) -> str:
        """Decode HTML entities and set dl=1 for direct download."""
        url = html.unescape(url)  # Decode &amp; -> &
        return url.replace("dl=0", "dl=1")

    # Find Dropbox links for contributors
    contrib_pattern = r'href="(https://[^"]*dropbox[^"]*dime_contributors[^"]*\.csv\.gz[^"]*)"'
    contrib_matches = re.findall(contrib_pattern, page_html, re.IGNORECASE)
    if contrib_matches:
        links["contributors"] = clean_url(contrib_matches[0])

    # Find Dropbox links for recipients
    recip_pattern = r'href="(https://[^"]*dropbox[^"]*dime_recipients[^"]*\.csv\.gz[^"]*)"'
    recip_matches = re.findall(recip_pattern, page_html, re.IGNORECASE)
    if recip_matches:
        links["recipients"] = clean_url(recip_matches[0])

    # Find Dropbox links for cycle contribution files
    cycle_pattern = r'href="(https://[^"]*dropbox[^"]*contribDB_(\d{4})\.csv\.gz[^"]*)"'
    cycle_matches = re.findall(cycle_pattern, page_html, re.IGNORECASE)
    for url, year in cycle_matches:
        year_int = int(year)
        if year_int in CYCLES:
            links["cycles"][year_int] = clean_url(url)

    # Save to cache
    DIME_RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOWNLOAD_LINKS_CACHE, "w") as f:
        json.dump(links, f, indent=2)

    print(f"Found {len(links['cycles'])} cycle files")
    print(f"Cached links to {DOWNLOAD_LINKS_CACHE}")

    return links


def download_file(
    url: str,
    output_path: Path,
    description: str = "Downloading",
    overwrite: bool = False,
) -> Path:
    """
    Download a file with progress bar.

    Args:
        url: URL to download
        output_path: Where to save the file
        description: Description for progress bar
        overwrite: Whether to overwrite existing file

    Returns:
        Path to downloaded file
    """
    if output_path.exists() and not overwrite:
        print(f"File already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use temp file to handle interrupted downloads
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get("content-length", 0))

        with open(temp_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=description,
                disable=total_size == 0,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Move temp file to final location
        shutil.move(temp_path, output_path)
        print(f"Downloaded: {output_path}")
        return output_path

    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Download failed: {e}")


def download_recipients(overwrite: bool = False) -> Path:
    """
    Download the DIME recipients file.

    Args:
        overwrite: Whether to overwrite existing file

    Returns:
        Path to downloaded file
    """
    links = scrape_dime_links()

    if not links.get("recipients"):
        raise RuntimeError("Recipients download link not found. Try --refresh-links.")

    output_path = DIME_RAW_DIR / "dime_recipients.csv.gz"

    return download_file(
        links["recipients"],
        output_path,
        description="Downloading recipients",
        overwrite=overwrite,
    )


def download_cycle_file(
    year: int,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Download a single cycle contribution file.

    Args:
        year: Election cycle year (e.g., 2020)
        output_dir: Directory to save file (default: temp directory)
        overwrite: Whether to overwrite existing file

    Returns:
        Path to downloaded file
    """
    links = scrape_dime_links()

    # Convert year to string for JSON key lookup
    year_str = str(year)
    if year_str not in links.get("cycles", {}):
        raise ValueError(f"No download link found for cycle {year}")

    output_dir = output_dir or DIME_RAW_DIR / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"contribDB_{year}.csv.gz"

    return download_file(
        links["cycles"][year_str],
        output_path,
        description=f"Downloading cycle {year}",
        overwrite=overwrite,
    )


def delete_cycle_file(year: int, output_dir: Optional[Path] = None) -> bool:
    """
    Delete a cycle contribution file after processing.

    Args:
        year: Election cycle year
        output_dir: Directory where file is saved

    Returns:
        True if file was deleted, False if not found
    """
    output_dir = output_dir or DIME_RAW_DIR / "temp"
    file_path = output_dir / f"contribDB_{year}.csv.gz"

    if file_path.exists():
        file_path.unlink()
        print(f"Deleted: {file_path}")
        return True

    return False


def get_available_cycles() -> list[int]:
    """
    Get list of cycles that have download links available.

    Returns:
        List of years with available downloads
    """
    links = scrape_dime_links()
    # Convert string keys to integers
    return sorted(int(k) for k in links.get("cycles", {}).keys())


def check_links_status() -> dict:
    """
    Check status of download links.

    Returns:
        Dictionary with status information
    """
    links = scrape_dime_links()

    return {
        "contributors_available": links.get("contributors") is not None,
        "recipients_available": links.get("recipients") is not None,
        "cycles_available": len(links.get("cycles", {})),
        "cycles_list": sorted(links.get("cycles", {}).keys()),
        "cache_path": str(DOWNLOAD_LINKS_CACHE),
    }


def refresh_download_links() -> dict:
    """
    Force refresh of download links cache.

    Returns:
        Updated links dictionary
    """
    if DOWNLOAD_LINKS_CACHE.exists():
        DOWNLOAD_LINKS_CACHE.unlink()

    return scrape_dime_links(force_refresh=True)
