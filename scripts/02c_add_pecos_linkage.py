#!/usr/bin/env python3
"""
Step 2c: Add PECOS Linkage (Graduation Year)

Adds PECOS (CMS Provider Enrollment) data to physician labels:
1. Ingests PECOS "Doctors and Clinicians" file
2. Links to existing NPPES matches via NPI
3. Adds graduation_year and other PECOS columns to physician_labels.parquet

Data source: https://data.cms.gov/provider-data/dataset/mj5m-pzi6
(Download "Doctors and Clinicians National Downloadable File")

Usage:
    python scripts/02c_add_pecos_linkage.py [OPTIONS]

Options:
    --overwrite       Overwrite existing files
    --skip-ingest     Skip PECOS ingestion (use existing parquet)
    --stats-only      Only show statistics, don't modify labels
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import click

from src.config import (
    PECOS_PHYSICIANS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    PECOS_RAW_DIR,
    ensure_directories,
)


@click.command()
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing files"
)
@click.option(
    "--skip-ingest",
    is_flag=True,
    help="Skip PECOS ingestion, use existing parquet"
)
@click.option(
    "--stats-only",
    is_flag=True,
    help="Only show statistics, don't modify labels"
)
def main(overwrite: bool, skip_ingest: bool, stats_only: bool):
    """Add PECOS linkage (graduation year) to physician labels."""
    ensure_directories()

    print("=" * 60)
    print("DIME Physician Analysis - PECOS Integration")
    print("=" * 60)

    from src.pecos_processing import (
        ingest_pecos_physicians,
        link_nppes_to_pecos,
        add_pecos_columns_to_labels,
        get_pecos_stats,
    )

    # Step 1: Ingest PECOS data
    if not skip_ingest:
        print("\n--- Step 1: Ingest PECOS Doctors and Clinicians ---")
        try:
            ingest_pecos_physicians(overwrite=overwrite)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"\nTo download PECOS data:")
            print("  1. Visit: https://data.cms.gov/provider-data/dataset/mj5m-pzi6")
            print("  2. Download 'Doctors and Clinicians National Downloadable File'")
            print(f"  3. Save to: {PECOS_RAW_DIR}/")
            return 1
    else:
        print("\n--- Step 1: Skipping PECOS ingestion ---")
        if not PECOS_PHYSICIANS_PARQUET.exists():
            print(f"Error: PECOS parquet not found: {PECOS_PHYSICIANS_PARQUET}")
            return 1

    # Step 2: Check NPPES-to-PECOS linkage
    print("\n--- Step 2: Check NPPES-to-PECOS linkage ---")
    link_nppes_to_pecos()

    # Step 3: Show statistics
    print("\n--- Step 3: PECOS Statistics ---")
    stats = get_pecos_stats()
    if stats:
        print(f"  Total records: {stats.get('n_total', 'N/A'):,}")
        print(f"  Unique NPIs: {stats.get('n_unique_npis', 'N/A'):,}")
        print(f"  With graduation year: {stats.get('n_with_grad_year', 'N/A'):,}")
        if stats.get('min_grad_year') and stats.get('max_grad_year'):
            print(f"  Graduation year range: {stats['min_grad_year']} - {stats['max_grad_year']}")

        if stats.get('gender_distribution'):
            print("\n  Gender distribution:")
            for g in stats['gender_distribution']:
                print(f"    {g['pecos_gender']}: {g['n']:,}")

        if stats.get('graduation_decades'):
            print("\n  Graduation by decade:")
            for d in stats['graduation_decades']:
                print(f"    {int(d['decade'])}s: {d['n']:,}")

    if stats_only:
        print("\n--- Skipping label modification (--stats-only) ---")
        return 0

    # Step 4: Add PECOS columns to labels
    print("\n--- Step 4: Add PECOS columns to physician labels ---")
    if not PHYSICIAN_LABELS_PARQUET.exists():
        print(f"Error: Physician labels not found: {PHYSICIAN_LABELS_PARQUET}")
        print("Run 02_build_physician_labels.py first.")
        return 1

    if not LINKAGE_RESULTS_PARQUET.exists():
        print(f"Warning: NPPES linkage not found: {LINKAGE_RESULTS_PARQUET}")
        print("PECOS columns will be limited without NPPES linkage.")
    else:
        add_pecos_columns_to_labels(overwrite=overwrite)

    print("\n" + "=" * 60)
    print("PECOS integration complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
