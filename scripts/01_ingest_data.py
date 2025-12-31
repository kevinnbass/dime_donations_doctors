#!/usr/bin/env python3
"""
Step 1: Data Ingestion

Converts raw DIME and NPPES data files to optimized Parquet format.

Usage:
    python scripts/01_ingest_data.py [OPTIONS]

Options:
    --dime-only      Only process DIME data
    --nppes-only     Only process NPPES data
    --overwrite      Overwrite existing files
"""

import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ensure_directories
from src.dime_processing import ingest_dime_contributors, ingest_dime_recipients
from src.nppes_processing import ingest_nppes_physicians


@click.command()
@click.option("--dime-only", is_flag=True, help="Only process DIME data")
@click.option("--nppes-only", is_flag=True, help="Only process NPPES data")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(dime_only: bool, nppes_only: bool, overwrite: bool):
    """Ingest raw data files and convert to Parquet format."""
    print("=" * 60)
    print("DIME Physician Analysis - Data Ingestion")
    print("=" * 60)

    # Ensure directories exist
    ensure_directories()

    if not nppes_only:
        print("\n--- Processing DIME Data ---")

        try:
            print("\n1. DIME Contributors (individual donors)")
            ingest_dime_contributors(overwrite=overwrite)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

        try:
            print("\n2. DIME Recipients")
            ingest_dime_recipients(overwrite=overwrite)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not dime_only:
        print("\n--- Processing NPPES Data ---")

        try:
            print("\n3. NPPES Physicians")
            ingest_nppes_physicians(overwrite=overwrite)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    print("\n" + "=" * 60)
    print("Data ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
