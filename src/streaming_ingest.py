"""
Streaming ingestion module for DIME itemized contribution files.

Downloads and processes itemized contribution records one election cycle
at a time to avoid memory issues with the full 180M+ row dataset.

Key features:
- Streaming download of cycle files
- DuckDB-based processing without full memory load
- Partitioned Parquet output by cycle
- Dual ideology measures (CFScore-weighted and party-weighted)
- Modal occupation/employer/state per donor-cycle
- Automatic cleanup of downloaded files
- Resumable processing (skips existing partitions)
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb

from .config import (
    CYCLES,
    DIME_ITEMIZED_MANIFEST,
    DIME_ITEMIZED_TMP_DIR,
    DIME_RECIPIENTS_PARQUET,
    DONOR_CYCLE_PANEL_PARTITIONED_DIR,
    PROCESSED_DATA_DIR,
    ensure_directories,
)

# Set up logging
logger = logging.getLogger(__name__)


def get_cycle_partition_path(cycle: int, base_dir: Path = None) -> Path:
    """Get the path to a cycle's Parquet partition directory."""
    base_dir = base_dir or DONOR_CYCLE_PANEL_PARTITIONED_DIR
    return base_dir / f"cycle={cycle}"


def get_cycle_manifest_path(cycle: int, base_dir: Path = None) -> Path:
    """Get the path to a cycle's JSON manifest."""
    base_dir = base_dir or DONOR_CYCLE_PANEL_PARTITIONED_DIR
    return base_dir / f"cycle_{cycle}_manifest.json"


def validate_cycle_output(cycle: int, base_dir: Path = None) -> dict:
    """
    Validate that a cycle's Parquet output exists and is valid.

    Args:
        cycle: Election cycle year
        base_dir: Base directory for partitioned output

    Returns:
        Dictionary with validation results:
        - valid: bool
        - exists: bool
        - row_count: int (if valid)
        - error: str (if invalid)
    """
    partition_path = get_cycle_partition_path(cycle, base_dir)

    if not partition_path.exists():
        return {"valid": False, "exists": False, "error": "Partition not found"}

    # Check for Parquet files
    parquet_files = list(partition_path.glob("*.parquet"))
    if not parquet_files:
        return {"valid": False, "exists": True, "error": "No Parquet files in partition"}

    try:
        con = duckdb.connect()
        path_str = str(partition_path / "*.parquet").replace("\\", "/")

        # Check row count
        result = con.execute(f"""
            SELECT COUNT(*) as cnt
            FROM read_parquet('{path_str}')
        """).fetchone()

        row_count = result[0] if result else 0

        if row_count == 0:
            con.close()
            return {"valid": False, "exists": True, "error": "Zero rows in partition"}

        # Check for required columns
        cols_result = con.execute(f"""
            SELECT column_name
            FROM (DESCRIBE SELECT * FROM read_parquet('{path_str}'))
        """).fetchall()

        columns = [row[0] for row in cols_result]
        con.close()

        required_cols = ["bonica_cid", "total_amount", "n_contributions"]
        missing = [c for c in required_cols if c not in columns]

        if missing:
            return {
                "valid": False,
                "exists": True,
                "error": f"Missing columns: {missing}",
                "row_count": row_count,
            }

        return {
            "valid": True,
            "exists": True,
            "row_count": row_count,
            "columns": columns,
        }

    except Exception as e:
        return {"valid": False, "exists": True, "error": str(e)}


def download_cycle_file(
    cycle: int,
    manifest_path: Path = None,
    output_dir: Path = None,
    overwrite: bool = False,
) -> Path:
    """
    Download a cycle's itemized contribution file.

    Args:
        cycle: Election cycle year
        manifest_path: Path to manifest CSV
        output_dir: Directory for downloads
        overwrite: Whether to overwrite existing file

    Returns:
        Path to downloaded file
    """
    from .download import download_cycle_file as _download_cycle_file

    output_dir = output_dir or DIME_ITEMIZED_TMP_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the existing download infrastructure
    return _download_cycle_file(cycle, output_dir=output_dir, overwrite=overwrite)


def cleanup_cycle_download(cycle: int, output_dir: Path = None) -> bool:
    """
    Delete a cycle's downloaded CSV file.

    Args:
        cycle: Election cycle year
        output_dir: Directory where file is saved

    Returns:
        True if file was deleted
    """
    from .download import delete_cycle_file

    output_dir = output_dir or DIME_ITEMIZED_TMP_DIR
    return delete_cycle_file(cycle, output_dir=output_dir)


def compute_party_score(party: str) -> int:
    """
    Convert party affiliation to numeric score.

    Args:
        party: Party string (e.g., '100' for DEM, '200' for REP)

    Returns:
        +1 for Democrat, -1 for Republican, 0 for other/unknown
    """
    if party is None:
        return 0

    party_str = str(party).upper().strip()

    # DIME uses numeric codes: 100 = DEM, 200 = REP
    if party_str in ("100", "DEM", "D", "DEMOCRAT", "DEMOCRATIC"):
        return 1
    elif party_str in ("200", "REP", "R", "REPUBLICAN"):
        return -1
    else:
        return 0


def process_cycle_to_parquet(
    cycle: int,
    csv_path: Path,
    recipients_path: Path = None,
    output_dir: Path = None,
    overwrite: bool = False,
) -> dict:
    """
    Process a cycle's contribution CSV to partitioned Parquet.

    Aggregates contributions to donor-cycle level with:
    - Total amount and contribution count
    - Unique recipient count
    - Revealed ideology (CFScore-weighted)
    - Revealed party ideology (party-weighted, no look-ahead)
    - Modal occupation, employer, state, ZIP, city

    Args:
        cycle: Election cycle year
        csv_path: Path to downloaded CSV file
        recipients_path: Path to recipients Parquet
        output_dir: Base directory for partitioned output
        overwrite: Whether to overwrite existing partition

    Returns:
        Dictionary with processing statistics
    """
    output_dir = output_dir or DONOR_CYCLE_PANEL_PARTITIONED_DIR
    recipients_path = recipients_path or DIME_RECIPIENTS_PARQUET

    partition_path = get_cycle_partition_path(cycle, output_dir)
    manifest_path = get_cycle_manifest_path(cycle, output_dir)

    # Check if already exists
    if partition_path.exists() and not overwrite:
        validation = validate_cycle_output(cycle, output_dir)
        if validation.get("valid"):
            logger.info(f"Cycle {cycle} already processed ({validation['row_count']} rows)")
            return {
                "cycle": cycle,
                "status": "skipped",
                "row_count": validation.get("row_count", 0),
                "message": "Already exists and valid",
            }

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up existing partition if overwriting
    if partition_path.exists():
        shutil.rmtree(partition_path)

    partition_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing cycle {cycle} from {csv_path.name}")

    con = duckdb.connect()

    # Convert paths to forward slashes for DuckDB
    csv_path_str = str(csv_path).replace("\\", "/")
    recipients_path_str = str(recipients_path).replace("\\", "/")
    output_path_str = str(partition_path / "part-0.parquet").replace("\\", "/")

    try:
        # First, check what columns are available in the CSV
        sample_query = f"""
            SELECT * FROM read_csv_auto('{csv_path_str}',
                                         header=true,
                                         sample_size=1000,
                                         all_varchar=true,
                                         ignore_errors=true)
            LIMIT 1
        """
        sample_result = con.execute(sample_query)
        available_cols = [desc[0] for desc in sample_result.description]
        logger.info(f"Available columns: {available_cols[:10]}...")

        # Detect column names (DIME uses different naming in different files)
        # Contribution files typically have columns like:
        # bonica.cid, bonica.rid, amount, date, contributor.*, etc.

        cid_col = None
        rid_col = None
        amount_col = None

        for col in available_cols:
            col_lower = col.lower()
            if "bonica" in col_lower and "cid" in col_lower:
                cid_col = col
            elif "bonica" in col_lower and "rid" in col_lower:
                rid_col = col
            elif col_lower in ("amount", "transaction.amount", "transaction_amount"):
                amount_col = col

        if not cid_col:
            cid_col = next((c for c in available_cols if "cid" in c.lower()), None)
        if not rid_col:
            rid_col = next((c for c in available_cols if "rid" in c.lower()), None)
        if not amount_col:
            amount_col = next((c for c in available_cols if "amount" in c.lower()), None)

        if not cid_col or not amount_col:
            raise ValueError(
                f"Could not find required columns. "
                f"Found: cid={cid_col}, rid={rid_col}, amount={amount_col}"
            )

        # Find contributor metadata columns
        occ_col = next(
            (c for c in available_cols if "occupation" in c.lower()),
            None
        )
        emp_col = next(
            (c for c in available_cols if "employer" in c.lower()),
            None
        )
        state_col = next(
            (c for c in available_cols if "state" in c.lower() and "contributor" in c.lower()),
            None
        )
        zip_col = next(
            (c for c in available_cols if "zip" in c.lower()),
            None
        )
        city_col = next(
            (c for c in available_cols if "city" in c.lower()),
            None
        )

        logger.info(
            f"Using columns: cid={cid_col}, rid={rid_col}, amount={amount_col}, "
            f"occ={occ_col}, emp={emp_col}, state={state_col}"
        )

        # Build the aggregation query
        # This is complex because we need:
        # 1. Sum amounts, count contributions, count unique recipients
        # 2. Weighted mean of recipient CFScore (for revealed_cfscore_cycle)
        # 3. Weighted mean of party score (for revealed_party_cycle)
        # 4. Modal values for occupation, employer, state, ZIP, city

        # Create contributions view with recipient join
        con.execute(f"""
            CREATE TEMP TABLE contribs AS
            SELECT
                c."{cid_col}" AS bonica_cid,
                c."{rid_col}" AS bonica_rid,
                TRY_CAST(c."{amount_col}" AS DOUBLE) AS amount,
                {f'c."{occ_col}"' if occ_col else 'NULL'} AS occupation,
                {f'c."{emp_col}"' if emp_col else 'NULL'} AS employer,
                {f'c."{state_col}"' if state_col else 'NULL'} AS state,
                {f'c."{zip_col}"' if zip_col else 'NULL'} AS zipcode,
                {f'c."{city_col}"' if city_col else 'NULL'} AS city,
                r.recipient_cfscore,
                r.recipient_party
            FROM read_csv_auto('{csv_path_str}',
                              header=true,
                              all_varchar=true,
                              sample_size=100000,
                              ignore_errors=true) c
            LEFT JOIN read_parquet('{recipients_path_str}') r
                ON c."{rid_col}" = r.bonica_rid
            WHERE c."{cid_col}" IS NOT NULL
              AND TRY_CAST(c."{amount_col}" AS DOUBLE) IS NOT NULL
              AND TRY_CAST(c."{amount_col}" AS DOUBLE) > 0
        """)

        # Get raw stats before aggregation
        raw_stats = con.execute("""
            SELECT
                COUNT(*) as n_raw_rows,
                COUNT(DISTINCT bonica_cid) as n_donors,
                SUM(amount) as total_amount,
                SUM(CASE WHEN bonica_cid IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_missing_cid,
                SUM(CASE WHEN bonica_rid IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_missing_rid,
                SUM(CASE WHEN recipient_cfscore IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_missing_cfscore
            FROM contribs
        """).fetchone()

        n_raw_rows = raw_stats[0]
        n_donors = raw_stats[1]
        total_amount = raw_stats[2]
        pct_missing_cid = raw_stats[3]
        pct_missing_rid = raw_stats[4]
        pct_missing_cfscore = raw_stats[5]

        logger.info(f"Raw contributions: {n_raw_rows:,}, Donors: {n_donors:,}")
        logger.info(
            f"Missing: cid={pct_missing_cid:.1f}%, rid={pct_missing_rid:.1f}%, "
            f"cfscore={pct_missing_cfscore:.1f}%"
        )

        # Compute modal values using window functions
        # For each (donor, field), find the value with the highest total amount
        con.execute("""
            CREATE TEMP TABLE modal_occupation AS
            SELECT bonica_cid, occupation AS occupation_cycle
            FROM (
                SELECT
                    bonica_cid,
                    occupation,
                    ROW_NUMBER() OVER (
                        PARTITION BY bonica_cid
                        ORDER BY SUM(amount) DESC, occupation
                    ) as rn
                FROM contribs
                WHERE occupation IS NOT NULL AND occupation != ''
                GROUP BY bonica_cid, occupation
            )
            WHERE rn = 1
        """)

        con.execute("""
            CREATE TEMP TABLE modal_employer AS
            SELECT bonica_cid, employer AS employer_cycle
            FROM (
                SELECT
                    bonica_cid,
                    employer,
                    ROW_NUMBER() OVER (
                        PARTITION BY bonica_cid
                        ORDER BY SUM(amount) DESC, employer
                    ) as rn
                FROM contribs
                WHERE employer IS NOT NULL AND employer != ''
                GROUP BY bonica_cid, employer
            )
            WHERE rn = 1
        """)

        con.execute("""
            CREATE TEMP TABLE modal_state AS
            SELECT bonica_cid, state AS state_cycle
            FROM (
                SELECT
                    bonica_cid,
                    state,
                    ROW_NUMBER() OVER (
                        PARTITION BY bonica_cid
                        ORDER BY SUM(amount) DESC, state
                    ) as rn
                FROM contribs
                WHERE state IS NOT NULL AND state != ''
                GROUP BY bonica_cid, state
            )
            WHERE rn = 1
        """)

        con.execute("""
            CREATE TEMP TABLE modal_zip AS
            SELECT bonica_cid, LEFT(zipcode, 5) AS zip5_cycle
            FROM (
                SELECT
                    bonica_cid,
                    zipcode,
                    ROW_NUMBER() OVER (
                        PARTITION BY bonica_cid
                        ORDER BY SUM(amount) DESC, zipcode
                    ) as rn
                FROM contribs
                WHERE zipcode IS NOT NULL AND zipcode != ''
                GROUP BY bonica_cid, zipcode
            )
            WHERE rn = 1
        """)

        con.execute("""
            CREATE TEMP TABLE modal_city AS
            SELECT bonica_cid, city AS city_cycle
            FROM (
                SELECT
                    bonica_cid,
                    city,
                    ROW_NUMBER() OVER (
                        PARTITION BY bonica_cid
                        ORDER BY SUM(amount) DESC, city
                    ) as rn
                FROM contribs
                WHERE city IS NOT NULL AND city != ''
                GROUP BY bonica_cid, city
            )
            WHERE rn = 1
        """)

        # Final aggregation with all components
        con.execute(f"""
            COPY (
                SELECT
                    agg.bonica_cid,
                    {cycle} AS cycle,
                    agg.total_amount,
                    agg.n_contributions,
                    agg.n_unique_recipients,
                    agg.revealed_cfscore_cycle,
                    agg.revealed_party_cycle,
                    mo.occupation_cycle,
                    me.employer_cycle,
                    ms.state_cycle,
                    mz.zip5_cycle,
                    mc.city_cycle
                FROM (
                    SELECT
                        bonica_cid,
                        SUM(amount) AS total_amount,
                        COUNT(*) AS n_contributions,
                        COUNT(DISTINCT bonica_rid) AS n_unique_recipients,
                        -- Revealed CFScore: weighted mean of recipient CFScores
                        SUM(amount * TRY_CAST(recipient_cfscore AS DOUBLE)) /
                            NULLIF(SUM(CASE WHEN recipient_cfscore IS NOT NULL
                                            THEN amount ELSE 0 END), 0)
                            AS revealed_cfscore_cycle,
                        -- Revealed party: weighted mean of party scores (+1 DEM, -1 REP, 0 other)
                        SUM(amount * CASE
                            WHEN recipient_party IN ('100', 'DEM', 'D') THEN 1.0
                            WHEN recipient_party IN ('200', 'REP', 'R') THEN -1.0
                            ELSE 0.0
                        END) /
                            NULLIF(SUM(CASE WHEN recipient_party IS NOT NULL
                                            THEN amount ELSE 0 END), 0)
                            AS revealed_party_cycle
                    FROM contribs
                    GROUP BY bonica_cid
                ) agg
                LEFT JOIN modal_occupation mo ON agg.bonica_cid = mo.bonica_cid
                LEFT JOIN modal_employer me ON agg.bonica_cid = me.bonica_cid
                LEFT JOIN modal_state ms ON agg.bonica_cid = ms.bonica_cid
                LEFT JOIN modal_zip mz ON agg.bonica_cid = mz.bonica_cid
                LEFT JOIN modal_city mc ON agg.bonica_cid = mc.bonica_cid
            ) TO '{output_path_str}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # Get output stats
        output_count = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{output_path_str}')
        """).fetchone()[0]

        con.close()

        # Write manifest
        manifest_data = {
            "cycle": cycle,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "input_file": str(csv_path),
            "output_dir": str(partition_path),
            "stats": {
                "n_raw_contributions": n_raw_rows,
                "n_donors": output_count,
                "total_amount": float(total_amount) if total_amount else 0,
                "pct_missing_cid": float(pct_missing_cid) if pct_missing_cid else 0,
                "pct_missing_rid": float(pct_missing_rid) if pct_missing_rid else 0,
                "pct_missing_cfscore": float(pct_missing_cfscore) if pct_missing_cfscore else 0,
            },
            "columns_used": {
                "cid": cid_col,
                "rid": rid_col,
                "amount": amount_col,
                "occupation": occ_col,
                "employer": emp_col,
                "state": state_col,
                "zipcode": zip_col,
                "city": city_col,
            },
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Cycle {cycle} complete: {output_count:,} donors written")

        return {
            "cycle": cycle,
            "status": "success",
            "row_count": output_count,
            "stats": manifest_data["stats"],
        }

    except Exception as e:
        logger.error(f"Error processing cycle {cycle}: {e}")
        con.close()

        # Clean up partial output
        if partition_path.exists():
            shutil.rmtree(partition_path)

        return {
            "cycle": cycle,
            "status": "error",
            "error": str(e),
        }


def process_cycle(
    cycle: int,
    force: bool = False,
    keep_download: bool = False,
    skip_validation: bool = False,
) -> dict:
    """
    Download, process, and clean up a single cycle.

    Args:
        cycle: Election cycle year
        force: Force reprocessing even if output exists
        keep_download: Don't delete CSV after processing
        skip_validation: Skip validation checks

    Returns:
        Dictionary with processing results
    """
    ensure_directories()

    # Check if already processed
    if not force:
        validation = validate_cycle_output(cycle)
        if validation.get("valid"):
            logger.info(f"Cycle {cycle} already valid, skipping")
            return {
                "cycle": cycle,
                "status": "skipped",
                "row_count": validation.get("row_count", 0),
            }

    # Download
    logger.info(f"Downloading cycle {cycle}...")
    try:
        csv_path = download_cycle_file(cycle, overwrite=force)
    except Exception as e:
        logger.error(f"Download failed for cycle {cycle}: {e}")
        return {
            "cycle": cycle,
            "status": "download_error",
            "error": str(e),
        }

    # Process - use partitioned processing for large cycles (2020+) to avoid disk overflow
    logger.info(f"Processing cycle {cycle}...")
    if cycle >= 2020:
        # Large cycles need partitioned processing to avoid DuckDB temp file explosion
        # Key insight: COPY ... PARTITION_BY avoids temp table materialization
        from .partitioned_ingest import process_cycle_partitioned
        result = process_cycle_partitioned(
            cycle=cycle,
            csv_path=csv_path,
            overwrite=force,
        )
    else:
        result = process_cycle_to_parquet(
            cycle=cycle,
            csv_path=csv_path,
            overwrite=force,
        )

    # Validate output
    if not skip_validation and result.get("status") == "success":
        validation = validate_cycle_output(cycle)
        if not validation.get("valid"):
            logger.warning(f"Validation failed for cycle {cycle}: {validation.get('error')}")
            result["validation_warning"] = validation.get("error")

    # Clean up download
    if not keep_download and result.get("status") == "success":
        logger.info(f"Cleaning up download for cycle {cycle}...")
        cleanup_cycle_download(cycle)

    return result


def process_all_cycles(
    cycles: list[int] = None,
    force: bool = False,
    keep_downloads: bool = False,
    skip_validation: bool = False,
) -> list[dict]:
    """
    Process multiple cycles with streaming ingestion.

    Args:
        cycles: List of cycles to process (default: all CYCLES)
        force: Force reprocessing even if output exists
        keep_downloads: Don't delete CSVs after processing
        skip_validation: Skip validation checks

    Returns:
        List of results for each cycle
    """
    cycles = cycles or CYCLES
    results = []

    for cycle in cycles:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing cycle {cycle}")
        logger.info(f"{'='*60}")

        result = process_cycle(
            cycle=cycle,
            force=force,
            keep_download=keep_downloads,
            skip_validation=skip_validation,
        )
        results.append(result)

        # Log progress
        success = sum(1 for r in results if r.get("status") == "success")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        errors = sum(1 for r in results if "error" in r.get("status", ""))

        logger.info(
            f"Progress: {len(results)}/{len(cycles)} "
            f"(success={success}, skipped={skipped}, errors={errors})"
        )

    return results


def get_available_cycles() -> list[int]:
    """
    Get list of cycles available in the manifest.

    Returns:
        List of cycle years with available URLs
    """
    from scripts.build_dime_manifest import load_manifest

    try:
        manifest = load_manifest()
        return sorted(manifest.keys())
    except FileNotFoundError:
        logger.warning("Manifest not found. Run build_dime_manifest.py first.")
        return []


def get_processed_cycles() -> list[int]:
    """
    Get list of cycles that have been successfully processed.

    Returns:
        List of cycle years with valid Parquet partitions
    """
    processed = []

    for cycle in CYCLES:
        validation = validate_cycle_output(cycle)
        if validation.get("valid"):
            processed.append(cycle)

    return sorted(processed)


def get_processing_status() -> dict:
    """
    Get overall processing status.

    Returns:
        Dictionary with status summary
    """
    all_cycles = set(CYCLES)
    processed = set(get_processed_cycles())
    available = set(get_available_cycles())

    return {
        "total_cycles": len(all_cycles),
        "processed_cycles": len(processed),
        "available_in_manifest": len(available),
        "pending": sorted(available - processed),
        "processed": sorted(processed),
        "not_in_manifest": sorted(all_cycles - available),
    }
