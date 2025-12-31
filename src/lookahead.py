"""
Look-ahead bias mitigation for physician classification.
Uses DuckDB streaming COPY to avoid memory explosion.
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from .config import (
    CYCLES,
    DIAGNOSTICS_DIR,
    DONOR_CYCLE_METADATA_PARQUET,
    DONOR_CYCLE_PANEL_PARTITIONED_DIR,
    DIME_DONORS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    PROCESSED_DATA_DIR,
    ensure_directories,
)

logger = logging.getLogger(__name__)


def extract_cycle_metadata_from_panel(
    output_path: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
    overwrite: bool = False,
) -> Path:
    """Extract cycle metadata using DuckDB streaming COPY - memory efficient."""
    output_path = output_path or DONOR_CYCLE_METADATA_PARQUET

    if output_path.exists() and not overwrite:
        logger.info(f"Output already exists: {output_path}")
        return output_path

    ensure_directories()

    if not DONOR_CYCLE_PANEL_PARTITIONED_DIR.exists():
        raise FileNotFoundError(
            f"Partitioned panel not found: {DONOR_CYCLE_PANEL_PARTITIONED_DIR}"
        )

    cycles_to_process = cycles or CYCLES
    panel_path = str(DONOR_CYCLE_PANEL_PARTITIONED_DIR).replace("\\", "/")

    logger.info(f"Reading partitioned panel from {panel_path}")

    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")

    cycle_list = ", ".join(str(c) for c in cycles_to_process)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_str = str(output_path).replace("\\", "/")

    query = f"""
        COPY (
            SELECT bonica_cid, cycle, occupation_cycle, employer_cycle,
                state_cycle, zip5_cycle, city_cycle, total_amount,
                n_contributions, n_unique_recipients,
                revealed_cfscore_cycle, revealed_party_cycle
            FROM read_parquet('{panel_path}/*/*.parquet', hive_partitioning=true)
            WHERE cycle IN ({cycle_list})
        ) TO '{output_str}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    try:
        con.execute(query)
    except Exception as e:
        logger.error(f"Error extracting cycle metadata: {e}")
        con.close()
        raise

    count_result = con.execute(f"""
        SELECT COUNT(*) as cnt, COUNT(DISTINCT bonica_cid) as unique_donors
        FROM read_parquet('{output_str}')
    """).fetchone()

    con.close()

    logger.info(f"Extracted {count_result[0]:,} donor-cycle records")
    logger.info(f"Unique donors: {count_result[1]:,}")
    logger.info(f"Created: {output_path}")
    return output_path


def generate_lookahead_audit(
    output_dir: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
) -> dict:
    """Generate lookahead audit using memory-efficient DuckDB."""
    output_dir = output_dir or DIAGNOSTICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cycles_to_analyze = cycles or CYCLES

    if not DIME_DONORS_PARQUET.exists():
        raise FileNotFoundError(f"Aggregate donor file not found: {DIME_DONORS_PARQUET}")

    if not DONOR_CYCLE_METADATA_PARQUET.exists():
        extract_cycle_metadata_from_panel(cycles=cycles)

    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")

    donors_path = str(DIME_DONORS_PARQUET).replace("\\", "/")
    metadata_path = str(DONOR_CYCLE_METADATA_PARQUET).replace("\\", "/")
    cycle_list = ", ".join(str(c) for c in cycles_to_analyze)

    outputs = {}

    state_audit = con.execute(f"""
        SELECT m.cycle, COUNT(*) as total_records,
            SUM(CASE WHEN m.state_cycle IS NOT NULL AND d.contributor_state IS NOT NULL
                     AND m.state_cycle != d.contributor_state THEN 1 ELSE 0 END) as state_disagree_count
        FROM read_parquet('{metadata_path}') m
        LEFT JOIN read_parquet('{donors_path}') d ON m.bonica_cid = d.bonica_cid
        WHERE m.cycle IN ({cycle_list})
        GROUP BY m.cycle ORDER BY m.cycle
    """).fetchdf()
    state_audit["state_disagree_rate"] = state_audit["state_disagree_count"] / state_audit["total_records"]
    state_path = output_dir / "lookahead_audit_state.csv"
    state_audit.to_csv(state_path, index=False)
    outputs["state_audit"] = state_path

    summary_result = con.execute(f"""
        SELECT COUNT(*) as total, COUNT(DISTINCT m.bonica_cid) as unique_donors
        FROM read_parquet('{metadata_path}') m WHERE m.cycle IN ({cycle_list})
    """).fetchone()
    con.close()

    summary = {
        "total_donor_cycle_records": summary_result[0],
        "unique_donors": summary_result[1],
        "cycles_analyzed": len(cycles_to_analyze),
        "first_cycle": min(cycles_to_analyze),
        "last_cycle": max(cycles_to_analyze),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / "lookahead_audit_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    outputs["summary"] = summary_path

    total = summary["total_donor_cycle_records"]
    unique = summary["unique_donors"]
    print(f"Total donor-cycle records: {total:,}")
    print(f"Unique donors: {unique:,}")

    return outputs


def compare_ideology_measures(output_path: Optional[Path] = None, cycles: Optional[list[int]] = None) -> Path:
    """Compare CFScore vs party ideology measures."""
    output_path = output_path or (DIAGNOSTICS_DIR / "ideology_comparison.csv")

    if not DONOR_CYCLE_METADATA_PARQUET.exists():
        raise FileNotFoundError(f"Cycle metadata not found: {DONOR_CYCLE_METADATA_PARQUET}")

    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    metadata_path = str(DONOR_CYCLE_METADATA_PARQUET).replace("\\", "/")
    cycles_to_analyze = cycles or CYCLES
    cycle_list = ", ".join(str(c) for c in cycles_to_analyze)

    comparison = con.execute(f"""
        SELECT cycle, COUNT(*) as n_records,
            AVG(revealed_cfscore_cycle) as cfscore_mean,
            AVG(revealed_party_cycle) as party_mean,
            CORR(revealed_cfscore_cycle, revealed_party_cycle) as correlation
        FROM read_parquet('{metadata_path}')
        WHERE cycle IN ({cycle_list})
          AND revealed_cfscore_cycle IS NOT NULL AND revealed_party_cycle IS NOT NULL
        GROUP BY cycle ORDER BY cycle
    """).fetchdf()
    con.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    logger.info(f"Created: {output_path}")
    return output_path
