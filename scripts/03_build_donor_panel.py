#!/usr/bin/env python3
"""
Step 3: Build Donor-Cycle Panel

Creates the donor-cycle panel with:
- Static CFScore (from DIME aggregate)
- Cycle-specific revealed ideology (from contribution records)
- Physician classification labels

Usage:
    python scripts/03_build_donor_panel.py [OPTIONS]

Options:
    --min-amount        Minimum donation amount per cycle (default: 200)
    --download-cycles   Auto-download cycle files for cycle-specific ideology
    --cycles            Comma-separated cycles to process (e.g., '2020,2022,2024')
    --overwrite         Overwrite existing files
"""

import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ensure_directories,
    DIME_DONORS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    DEFAULT_MIN_CYCLE_AMOUNT,
)
from src.ideology import (
    build_donor_cycle_panel,
    build_donor_cycle_panel_with_streaming,
    get_panel_stats,
)


@click.command()
@click.option(
    "--min-amount",
    type=float,
    default=DEFAULT_MIN_CYCLE_AMOUNT,
    help=f"Minimum donation amount per cycle (default: {DEFAULT_MIN_CYCLE_AMOUNT})",
)
@click.option(
    "--download-cycles",
    is_flag=True,
    help="Auto-download cycle contribution files for cycle-specific ideology (~3GB temp space)",
)
@click.option(
    "--cycles",
    type=str,
    default=None,
    help="Comma-separated list of cycles to process (e.g., '2020,2022,2024'). Default: all cycles.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(min_amount: float, download_cycles: bool, cycles: str, overwrite: bool):
    """Build the donor-cycle panel with ideology scores."""
    print("=" * 60)
    print("DIME Physician Analysis - Donor-Cycle Panel")
    print("=" * 60)

    ensure_directories()

    # Check prerequisites
    if not DIME_DONORS_PARQUET.exists():
        print("Error: DIME donors parquet not found.")
        print("Run 01_ingest_data.py first.")
        sys.exit(1)

    if not PHYSICIAN_LABELS_PARQUET.exists():
        print("Warning: Physician labels not found.")
        print("Run 02_build_physician_labels.py for full functionality.")

    # Build panel
    print(f"\nBuilding panel with min_amount={min_amount}...")

    if download_cycles:
        # Parse cycles if specified
        cycle_list = None
        if cycles:
            cycle_list = [int(c.strip()) for c in cycles.split(",")]
            print(f"Mode: Streaming download of {len(cycle_list)} cycle(s): {cycle_list}")
        else:
            print("Mode: Streaming download of ALL cycle files")

        print("  - Will download each cycle file (~1-3GB)")
        print("  - Process and extract ideology scores")
        print("  - Delete file before next cycle")
        print("  - Total temp space needed: ~3GB")
        print()

        build_donor_cycle_panel_with_streaming(
            min_amount=min_amount,
            download_cycles=True,
            cycles=cycle_list,
            overwrite=overwrite,
        )
    else:
        print("Mode: Static CFScore only (use --download-cycles for cycle-specific)")
        build_donor_cycle_panel(
            min_amount=min_amount,
            overwrite=overwrite,
        )

    # Print stats
    print("\n--- Panel Statistics ---")
    stats = get_panel_stats()
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Donor-cycle panel complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
