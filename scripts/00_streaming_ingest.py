#!/usr/bin/env python3
"""
Step 0: Streaming Ingestion of Itemized Contribution Files

Downloads and processes DIME itemized contribution records one election
cycle at a time, creating partitioned Parquet output with:
- Dual ideology measures (CFScore-weighted and party-weighted)
- Modal occupation/employer/state per donor-cycle
- No full dataset in memory

Usage:
    python scripts/00_streaming_ingest.py [OPTIONS]

Options:
    --cycles TEXT       Comma-separated cycles or 'all' (default: all)
    --force             Force reprocessing even if output exists
    --keep-downloads    Don't delete CSVs after processing
    --skip-validation   Skip post-processing validation
    --status            Show processing status without processing
"""

import logging
import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CYCLES, ensure_directories
from src.streaming_ingest import (
    get_processing_status,
    process_all_cycles,
    process_cycle,
    validate_cycle_output,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_cycles(cycles_str: str) -> list[int]:
    """Parse cycles argument into list of years."""
    if cycles_str.lower() == "all":
        return CYCLES

    cycles = []
    for part in cycles_str.split(","):
        part = part.strip()
        if "-" in part:
            # Range: e.g., "2010-2020"
            start, end = part.split("-")
            cycles.extend(range(int(start), int(end) + 1, 2))
        else:
            cycles.append(int(part))

    # Filter to valid cycles
    valid = [c for c in cycles if c in CYCLES]
    invalid = [c for c in cycles if c not in CYCLES]

    if invalid:
        logger.warning(f"Ignoring invalid cycles: {invalid}")

    return sorted(valid)


@click.command()
@click.option(
    "--cycles",
    default="all",
    help="Comma-separated cycles or 'all' (e.g., '2020,2022,2024' or '2010-2024')"
)
@click.option(
    "--force",
    is_flag=True,
    help="Force reprocessing even if output exists"
)
@click.option(
    "--keep-downloads",
    is_flag=True,
    help="Don't delete CSVs after processing"
)
@click.option(
    "--skip-validation",
    is_flag=True,
    help="Skip post-processing validation"
)
@click.option(
    "--status",
    is_flag=True,
    help="Show processing status without processing"
)
@click.option(
    "--validate-only",
    is_flag=True,
    help="Only validate existing outputs, don't process"
)
def main(
    cycles: str,
    force: bool,
    keep_downloads: bool,
    skip_validation: bool,
    status: bool,
    validate_only: bool,
):
    """
    Streaming ingestion of DIME itemized contribution files.

    Downloads and processes one cycle at a time to avoid memory issues.
    Creates partitioned Parquet output with dual ideology measures.
    """
    print("=" * 70)
    print("DIME Streaming Ingestion - Itemized Contribution Processing")
    print("=" * 70)

    # Ensure directories exist
    ensure_directories()

    # Status mode
    if status:
        print("\n--- Processing Status ---")
        status_info = get_processing_status()

        print(f"Total cycles in range: {status_info['total_cycles']}")
        print(f"Cycles processed: {status_info['processed_cycles']}")
        print(f"Available in manifest: {status_info['available_in_manifest']}")

        if status_info['processed']:
            print(f"\nProcessed: {status_info['processed']}")

        if status_info['pending']:
            print(f"\nPending: {status_info['pending']}")

        if status_info['not_in_manifest']:
            print(f"\nNot in manifest: {status_info['not_in_manifest']}")

        return

    # Parse cycles
    cycle_list = parse_cycles(cycles)

    if not cycle_list:
        print("No valid cycles to process.")
        return

    print(f"\nCycles to process: {cycle_list}")
    print(f"Force reprocess: {force}")
    print(f"Keep downloads: {keep_downloads}")
    print(f"Skip validation: {skip_validation}")

    # Validate-only mode
    if validate_only:
        print("\n--- Validation Only ---")
        for cycle in cycle_list:
            result = validate_cycle_output(cycle)
            status_str = "VALID" if result.get("valid") else "INVALID"
            row_count = result.get("row_count", 0)
            error = result.get("error", "")

            print(f"  {cycle}: {status_str} ({row_count:,} rows) {error}")
        return

    # Process cycles
    print("\n--- Starting Processing ---")
    results = process_all_cycles(
        cycles=cycle_list,
        force=force,
        keep_downloads=keep_downloads,
        skip_validation=skip_validation,
    )

    # Summary
    print("\n" + "=" * 70)
    print("Processing Summary")
    print("=" * 70)

    success_count = 0
    skipped_count = 0
    error_count = 0
    total_rows = 0

    for result in results:
        cycle = result.get("cycle")
        status_val = result.get("status", "unknown")
        row_count = result.get("row_count", 0)

        if status_val == "success":
            success_count += 1
            total_rows += row_count
            print(f"  {cycle}: SUCCESS ({row_count:,} donors)")
        elif status_val == "skipped":
            skipped_count += 1
            total_rows += row_count
            print(f"  {cycle}: SKIPPED (already processed, {row_count:,} donors)")
        else:
            error_count += 1
            error = result.get("error", "Unknown error")
            print(f"  {cycle}: ERROR - {error}")

    print("\n" + "-" * 70)
    print(f"Processed: {success_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Total donors across cycles: {total_rows:,}")

    if error_count > 0:
        print("\nSome cycles failed. Check logs for details.")
        print("Use --force to retry failed cycles.")

    print("\n" + "=" * 70)
    print("Streaming ingestion complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
