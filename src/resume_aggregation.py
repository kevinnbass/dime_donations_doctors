#!/usr/bin/env python3
"""
Resume aggregation for cycle 2020 after partial failure.

The partitioning completed successfully (223M rows, 128 partitions).
Aggregation failed at partition 109 (0-byte output file).
This script resumes from where we left off.
"""

import logging
import shutil
from pathlib import Path
import duckdb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the main module
from .partitioned_ingest import (
    aggregate_partition,
    MEMORY_LIMIT,
)
from .config import (
    DIME_RECIPIENTS_PARQUET,
    DIME_ITEMIZED_TMP_DIR,
    DONOR_CYCLE_PANEL_PARTITIONED_DIR,
)


def resume_cycle_2020():
    """Resume aggregation for cycle 2020."""
    cycle = 2020
    partition_dir = DIME_ITEMIZED_TMP_DIR / f"parts_{cycle}"
    donors_dir = DIME_ITEMIZED_TMP_DIR / f"donors_{cycle}"
    final_output_dir = DONOR_CYCLE_PANEL_PARTITIONED_DIR / f"cycle={cycle}"
    temp_dir = DIME_ITEMIZED_TMP_DIR / "duckdb_temp"
    recipients_path = DIME_RECIPIENTS_PARQUET

    logger.info("Resuming aggregation for cycle 2020...")

    # Remove 0-byte donor files
    for f in donors_dir.glob("*.parquet"):
        if f.stat().st_size == 0:
            logger.info(f"Removing empty file: {f.name}")
            f.unlink()

    # Get completed donor files
    completed_parts = set()
    for f in donors_dir.glob("part_*.parquet"):
        if f.stat().st_size > 0:
            part_num = f.stem.split("_")[1]
            completed_parts.add(part_num)

    logger.info(f"Already completed: {len(completed_parts)} partitions")

    # Get remaining source partitions
    remaining_dirs = sorted([
        d for d in partition_dir.iterdir()
        if d.is_dir() and d.name.startswith("part=")
    ])

    logger.info(f"Remaining source partitions: {len(remaining_dirs)}")

    # Create DuckDB connection
    con = duckdb.connect()
    temp_dir_str = str(temp_dir).replace("\\", "/")
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{temp_dir_str}'")

    total_donors = 0
    partitions_processed = 0

    for part_dir in remaining_dirs:
        part_num = part_dir.name.split("=")[1]

        # Skip if already completed
        if part_num in completed_parts:
            logger.info(f"Skipping already completed partition {part_num}")
            continue

        output_file = donors_dir / f"part_{part_num}.parquet"

        try:
            donor_count = aggregate_partition(
                con, part_dir, output_file, recipients_path, cycle
            )
            total_donors += donor_count
            partitions_processed += 1

            # Delete partition after successful aggregation
            shutil.rmtree(part_dir)

            if partitions_processed % 20 == 0:
                logger.info(f"  Aggregated {partitions_processed} more partitions, {total_donors:,} new donors")

        except Exception as e:
            logger.error(f"Error aggregating partition {part_num}: {e}")
            # Don't delete partition if aggregation failed
            raise

    logger.info(f"Resume complete: aggregated {total_donors:,} donors from {partitions_processed} partitions")

    # Count total donors including previously completed
    total_donor_files = list(donors_dir.glob("part_*.parquet"))
    logger.info(f"Total donor files: {len(total_donor_files)}")

    # Combine into final output
    logger.info("Combining donor files into final output...")
    donors_dir_str = str(donors_dir).replace("\\", "/")
    final_path = final_output_dir / "part-0.parquet"
    final_path_str = str(final_path).replace("\\", "/")

    # Remove existing final output if present
    if final_path.exists():
        final_path.unlink()

    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{donors_dir_str}/*.parquet')
        ) TO '{final_path_str}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Verify final count
    final_count = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{final_path_str}')
    """).fetchone()[0]

    logger.info(f"Final output: {final_count:,} donors written to {final_path}")

    # Clean up
    logger.info("Cleaning up temp directories...")
    if partition_dir.exists():
        shutil.rmtree(partition_dir)
    if donors_dir.exists():
        shutil.rmtree(donors_dir)

    con.close()
    logger.info("Done!")

    return final_count


if __name__ == "__main__":
    resume_cycle_2020()
