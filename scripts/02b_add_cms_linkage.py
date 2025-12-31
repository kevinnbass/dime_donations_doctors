#!/usr/bin/env python3
"""
Step 2b: Add CMS Medicare Linkage

Adds CMS Medicare PUF linkage to physician labels:
1. Ingests CMS Medicare PUF data
2. Links to existing NPPES matches via NPI
3. Adds CMS columns to physician_labels.parquet

Usage:
    python scripts/02b_add_cms_linkage.py [OPTIONS]

Options:
    --min-beneficiaries   Minimum beneficiaries for active (default: 11)
    --overwrite           Overwrite existing files
    --skip-ingest         Skip CMS ingestion (use existing parquet)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import click

from src.config import (
    CMS_MEDICARE_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    DEFAULT_CMS_MIN_BENEFICIARIES,
    ensure_directories,
)


@click.command()
@click.option(
    "--min-beneficiaries",
    type=int,
    default=DEFAULT_CMS_MIN_BENEFICIARIES,
    help=f"Minimum Medicare beneficiaries (default: {DEFAULT_CMS_MIN_BENEFICIARIES})"
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing files"
)
@click.option(
    "--skip-ingest",
    is_flag=True,
    help="Skip CMS ingestion, use existing parquet"
)
def main(min_beneficiaries: int, overwrite: bool, skip_ingest: bool):
    """Add CMS Medicare linkage to physician labels."""
    ensure_directories()

    print("=" * 60)
    print("DIME Physician Analysis - CMS Medicare Integration")
    print("=" * 60)

    from src.cms_processing import (
        ingest_cms_medicare_physicians,
        link_nppes_to_cms,
        add_cms_columns_to_labels,
    )

    # Step 1: Ingest CMS data
    if not skip_ingest:
        print("\n--- Step 1: Ingest CMS Medicare PUF ---")
        try:
            ingest_cms_medicare_physicians(
                min_beneficiaries=min_beneficiaries,
                overwrite=overwrite,
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please download CMS Medicare Provider PUF to data/raw/cms/")
            return 1
    else:
        print("\n--- Step 1: Skipping CMS ingestion ---")
        if not CMS_MEDICARE_PARQUET.exists():
            print(f"Error: CMS parquet not found: {CMS_MEDICARE_PARQUET}")
            return 1

    # Step 2: Check NPPES linkage
    print("\n--- Step 2: Check NPPES-to-CMS linkage ---")
    if not LINKAGE_RESULTS_PARQUET.exists():
        print(f"Warning: NPPES linkage not found: {LINKAGE_RESULTS_PARQUET}")
        print("CMS columns will be limited without NPPES linkage.")
    else:
        link_nppes_to_cms()

    # Step 3: Add CMS columns to labels
    print("\n--- Step 3: Add CMS columns to physician labels ---")
    if not PHYSICIAN_LABELS_PARQUET.exists():
        print(f"Error: Physician labels not found: {PHYSICIAN_LABELS_PARQUET}")
        print("Run 02_build_physician_labels.py first.")
        return 1

    add_cms_columns_to_labels(overwrite=overwrite)

    print("\n" + "=" * 60)
    print("CMS Medicare integration complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
