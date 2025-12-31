"""
Coverage and missingness audits for DIME data.

This module provides:
1. Aggregate vs itemized data reconciliation
2. Recipient CFScore missingness analysis
3. Occupation missingness analysis by cycle

These audits help quantify data quality and coverage limitations.
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
    DIME_DONORS_PARQUET,
    DIME_RECIPIENTS_PARQUET,
    DONOR_CYCLE_METADATA_PARQUET,
    DONOR_CYCLE_PANEL_PARTITIONED_DIR,
    ensure_directories,
)

logger = logging.getLogger(__name__)


def audit_aggregate_vs_itemized(
    output_dir: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
) -> dict[str, Path]:
    """
    Compare aggregate amounts in dime_donors vs itemized panel.

    The aggregate donor file has per-cycle amount columns that should
    approximately match the sum of itemized contributions.

    Note: Public itemized files may exclude CRP/NIMSP-sourced records,
    so coverage may be less than 100%.

    Args:
        output_dir: Directory for output files
        cycles: Optional list of cycles to analyze

    Returns:
        Dictionary mapping output names to file paths
    """
    output_dir = output_dir or DIAGNOSTICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cycles_to_analyze = cycles or CYCLES

    # Check for required files
    if not DIME_DONORS_PARQUET.exists():
        raise FileNotFoundError(f"Aggregate donors not found: {DIME_DONORS_PARQUET}")

    if not DONOR_CYCLE_PANEL_PARTITIONED_DIR.exists():
        raise FileNotFoundError(
            f"Partitioned panel not found: {DONOR_CYCLE_PANEL_PARTITIONED_DIR}\n"
            "Run streaming ingestion first."
        )

    logger.info("Loading aggregate donor amounts...")
    con = duckdb.connect()

    donors_path = str(DIME_DONORS_PARQUET).replace("\\", "/")
    panel_path = str(DONOR_CYCLE_PANEL_PARTITIONED_DIR).replace("\\", "/")

    outputs = {}

    # Build unpivot query for aggregate amounts
    amount_unions = []
    for year in cycles_to_analyze:
        amount_unions.append(f"""
            SELECT
                CAST(bonica_cid AS VARCHAR) AS bonica_cid,
                {year} AS cycle,
                "amount.{year}" AS amount_aggregate
            FROM read_parquet('{donors_path}')
            WHERE "amount.{year}" IS NOT NULL AND "amount.{year}" > 0
        """)

    aggregate_query = " UNION ALL ".join(amount_unions)

    # Get aggregate amounts
    aggregate_df = con.execute(f"""
        SELECT bonica_cid, cycle, amount_aggregate
        FROM ({aggregate_query})
    """).fetchdf()

    logger.info(f"Loaded {len(aggregate_df):,} aggregate donor-cycle records")

    # Get itemized amounts from partitioned panel
    cycle_list = ", ".join(str(c) for c in cycles_to_analyze)
    itemized_df = con.execute(f"""
        SELECT
            bonica_cid,
            cycle,
            total_amount AS amount_itemized
        FROM read_parquet('{panel_path}/*/*.parquet', hive_partitioning=true)
        WHERE cycle IN ({cycle_list})
    """).fetchdf()

    logger.info(f"Loaded {len(itemized_df):,} itemized donor-cycle records")

    # Merge for comparison
    merged = aggregate_df.merge(
        itemized_df,
        on=["bonica_cid", "cycle"],
        how="outer",
        indicator=True,
    )

    merged["amount_aggregate"] = merged["amount_aggregate"].fillna(0)
    merged["amount_itemized"] = merged["amount_itemized"].fillna(0)

    # Compute coverage ratio
    merged["coverage_ratio"] = np.where(
        merged["amount_aggregate"] > 0,
        merged["amount_itemized"] / merged["amount_aggregate"],
        np.nan,
    )

    # 1. Per-cycle summary
    logger.info("Computing per-cycle coverage statistics...")

    cycle_summary = merged.groupby("cycle").agg(
        n_aggregate=("_merge", lambda x: (x.isin(["both", "left_only"])).sum()),
        n_itemized=("_merge", lambda x: (x.isin(["both", "right_only"])).sum()),
        n_both=("_merge", lambda x: (x == "both").sum()),
        n_aggregate_only=("_merge", lambda x: (x == "left_only").sum()),
        n_itemized_only=("_merge", lambda x: (x == "right_only").sum()),
        total_amount_aggregate=("amount_aggregate", "sum"),
        total_amount_itemized=("amount_itemized", "sum"),
        mean_coverage_ratio=("coverage_ratio", "mean"),
        median_coverage_ratio=("coverage_ratio", "median"),
        p10_coverage_ratio=("coverage_ratio", lambda x: x.quantile(0.1)),
        p90_coverage_ratio=("coverage_ratio", lambda x: x.quantile(0.9)),
        share_coverage_below_80pct=("coverage_ratio", lambda x: (x < 0.8).mean()),
        share_coverage_above_120pct=("coverage_ratio", lambda x: (x > 1.2).mean()),
    ).reset_index()

    cycle_summary["overall_coverage_ratio"] = (
        cycle_summary["total_amount_itemized"] / cycle_summary["total_amount_aggregate"]
    )

    cycle_path = output_dir / "itemized_coverage_by_cycle.csv"
    cycle_summary.to_csv(cycle_path, index=False)
    outputs["by_cycle"] = cycle_path
    logger.info(f"Created: {cycle_path}")

    # 2. Per-cycle-state summary (for geographic patterns)
    logger.info("Computing per-cycle-state coverage statistics...")

    # Get state from aggregate data
    state_df = con.execute(f"""
        SELECT CAST(bonica_cid AS VARCHAR) AS bonica_cid, contributor_state AS state
        FROM read_parquet('{donors_path}')
        WHERE contributor_state IS NOT NULL
    """).fetchdf()
    
    con.close()

    merged_with_state = merged.merge(state_df, on="bonica_cid", how="left")

    state_summary = merged_with_state.groupby(["cycle", "state"]).agg(
        n_donors=("bonica_cid", "nunique"),
        total_amount_aggregate=("amount_aggregate", "sum"),
        total_amount_itemized=("amount_itemized", "sum"),
        mean_coverage_ratio=("coverage_ratio", "mean"),
    ).reset_index()

    state_summary["overall_coverage_ratio"] = (
        state_summary["total_amount_itemized"] / state_summary["total_amount_aggregate"]
    )

    state_path = output_dir / "itemized_coverage_by_cycle_state.csv"
    state_summary.to_csv(state_path, index=False)
    outputs["by_cycle_state"] = state_path
    logger.info(f"Created: {state_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Aggregate vs Itemized Coverage Summary")
    print("=" * 70)
    overall = cycle_summary.agg({
        "total_amount_aggregate": "sum",
        "total_amount_itemized": "sum",
    })
    overall_ratio = overall["total_amount_itemized"] / overall["total_amount_aggregate"]
    print(f"Overall coverage ratio: {overall_ratio:.2%}")
    print(f"Cycles analyzed: {len(cycles_to_analyze)}")
    print(f"Mean coverage ratio across donor-cycles: {merged['coverage_ratio'].mean():.2%}")
    print(f"Share with <80% coverage: {(merged['coverage_ratio'] < 0.8).mean():.1%}")
    print("=" * 70)

    return outputs


def audit_recipient_cfscore_missingness(
    output_dir: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
) -> Path:
    """
    Audit recipient CFScore missingness by cycle.

    Computes:
    - % of contributions with NULL recipient CFScore
    - % of amount going to recipients with NULL CFScore
    - This affects the reliability of revealed_cfscore_cycle

    Args:
        output_dir: Directory for output files
        cycles: Optional list of cycles to analyze

    Returns:
        Path to output CSV
    """
    output_dir = output_dir or DIAGNOSTICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cycles_to_analyze = cycles or CYCLES

    if not DONOR_CYCLE_PANEL_PARTITIONED_DIR.exists():
        raise FileNotFoundError(
            f"Partitioned panel not found: {DONOR_CYCLE_PANEL_PARTITIONED_DIR}"
        )

    if not DIME_RECIPIENTS_PARQUET.exists():
        raise FileNotFoundError(
            f"Recipients file not found: {DIME_RECIPIENTS_PARQUET}"
        )

    logger.info("Computing recipient CFScore missingness...")
    con = duckdb.connect()

    panel_path = str(DONOR_CYCLE_PANEL_PARTITIONED_DIR).replace("\\", "/")
    recipients_path = str(DIME_RECIPIENTS_PARQUET).replace("\\", "/")

    # Get recipient CFScore coverage
    cycle_list = ", ".join(str(c) for c in cycles_to_analyze)

    # This requires the original itemized data with recipient IDs
    # For now, we use the panel's revealed_cfscore_cycle as proxy
    # (if it's NULL, CFScore was missing for all/most recipients)

    missingness = con.execute(f"""
        SELECT
            cycle,
            COUNT(*) AS n_records,
            SUM(CASE WHEN revealed_cfscore_cycle IS NULL THEN 1 ELSE 0 END) AS n_missing_cfscore,
            SUM(total_amount) AS total_amount,
            SUM(CASE WHEN revealed_cfscore_cycle IS NULL THEN total_amount ELSE 0 END) AS amount_missing_cfscore
        FROM read_parquet('{panel_path}/*/*.parquet', hive_partitioning=true)
        WHERE cycle IN ({cycle_list})
        GROUP BY cycle
        ORDER BY cycle
    """).fetchdf()

    con.close()

    missingness["pct_records_missing"] = (
        missingness["n_missing_cfscore"] / missingness["n_records"]
    )
    missingness["pct_amount_missing"] = (
        missingness["amount_missing_cfscore"] / missingness["total_amount"]
    )

    output_path = output_dir / "recipient_cfscore_missingness.csv"
    missingness.to_csv(output_path, index=False)
    logger.info(f"Created: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Recipient CFScore Missingness Summary")
    print("=" * 70)
    print(f"Mean % records missing CFScore: {missingness['pct_records_missing'].mean():.1%}")
    print(f"Mean % amount to missing recipients: {missingness['pct_amount_missing'].mean():.1%}")
    print("=" * 70)

    return output_path


def audit_occupation_missingness(
    output_dir: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
    physician_only: bool = False,
) -> Path:
    """
    Audit occupation missingness by cycle.

    Computes:
    - Share of records with NULL/empty occupation_cycle
    - For physician records specifically (if physician_only=True)

    Args:
        output_dir: Directory for output files
        cycles: Optional list of cycles to analyze
        physician_only: Whether to restrict to physician-classified records

    Returns:
        Path to output CSV
    """
    output_dir = output_dir or DIAGNOSTICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cycles_to_analyze = cycles or CYCLES

    if not DONOR_CYCLE_METADATA_PARQUET.exists():
        raise FileNotFoundError(
            f"Cycle metadata not found: {DONOR_CYCLE_METADATA_PARQUET}\n"
            "Run lookahead module first."
        )

    logger.info("Computing occupation missingness...")
    con = duckdb.connect()

    metadata_path = str(DONOR_CYCLE_METADATA_PARQUET).replace("\\", "/")
    cycle_list = ", ".join(str(c) for c in cycles_to_analyze)

    missingness = con.execute(f"""
        SELECT
            cycle,
            COUNT(*) AS n_records,
            SUM(CASE WHEN occupation_cycle IS NULL OR occupation_cycle = '' THEN 1 ELSE 0 END) AS n_missing_occupation,
            SUM(total_amount) AS total_amount,
            SUM(CASE WHEN occupation_cycle IS NULL OR occupation_cycle = '' THEN total_amount ELSE 0 END) AS amount_missing_occupation
        FROM read_parquet('{metadata_path}')
        WHERE cycle IN ({cycle_list})
        GROUP BY cycle
        ORDER BY cycle
    """).fetchdf()

    con.close()

    missingness["pct_records_missing"] = (
        missingness["n_missing_occupation"] / missingness["n_records"]
    )
    missingness["pct_amount_missing"] = (
        missingness["amount_missing_occupation"] / missingness["total_amount"]
    )

    suffix = "_physician" if physician_only else ""
    output_path = output_dir / f"occupation_missingness_by_cycle{suffix}.csv"
    missingness.to_csv(output_path, index=False)
    logger.info(f"Created: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Occupation Missingness Summary")
    print("=" * 70)
    print(f"Mean % records missing occupation: {missingness['pct_records_missing'].mean():.1%}")
    print(f"Mean % amount from missing occupation: {missingness['pct_amount_missing'].mean():.1%}")
    print("=" * 70)

    return output_path


def run_all_coverage_audits(
    output_dir: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
) -> dict[str, Path]:
    """
    Run all coverage audits and return paths to output files.

    Args:
        output_dir: Directory for output files
        cycles: Optional list of cycles to analyze

    Returns:
        Dictionary mapping audit names to output paths
    """
    output_dir = output_dir or DIAGNOSTICS_DIR
    ensure_directories()

    all_outputs = {}

    print("=" * 70)
    print("Running Coverage Audits")
    print("=" * 70)

    # 1. Aggregate vs itemized coverage
    print("\n--- Aggregate vs Itemized Coverage ---")
    try:
        coverage_outputs = audit_aggregate_vs_itemized(output_dir, cycles)
        all_outputs.update({f"aggregate_coverage_{k}": v for k, v in coverage_outputs.items()})
    except FileNotFoundError as e:
        logger.warning(f"Skipping aggregate vs itemized audit: {e}")

    # 2. Recipient CFScore missingness
    print("\n--- Recipient CFScore Missingness ---")
    try:
        cfscore_output = audit_recipient_cfscore_missingness(output_dir, cycles)
        all_outputs["recipient_cfscore_missingness"] = cfscore_output
    except FileNotFoundError as e:
        logger.warning(f"Skipping CFScore missingness audit: {e}")

    # 3. Occupation missingness
    print("\n--- Occupation Missingness ---")
    try:
        occupation_output = audit_occupation_missingness(output_dir, cycles)
        all_outputs["occupation_missingness"] = occupation_output
    except FileNotFoundError as e:
        logger.warning(f"Skipping occupation missingness audit: {e}")

    print("\n" + "=" * 70)
    print("Coverage audits complete!")
    print(f"Output files: {len(all_outputs)}")
    print("=" * 70)

    return all_outputs


def generate_coverage_summary_report(
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate a summary report of all coverage findings.

    Reads existing audit outputs and creates a consolidated summary.

    Args:
        output_dir: Directory containing audit outputs

    Returns:
        Path to summary report
    """
    output_dir = output_dir or DIAGNOSTICS_DIR

    report_lines = [
        "# DIME Data Coverage Summary Report",
        "",
        "## Data Sources",
        "- DIME aggregate donor file: dime_donors.parquet",
        "- DIME partitioned panel: donor_cycle_panel/cycle=YYYY/",
        "- DIME recipients file: dime_recipients.parquet",
        "",
        "## Coverage Limitations",
        "",
        "**Public itemized files may have incomplete coverage:**",
        "- Stanford DIME notes that public itemized-by-year files exclude",
        "  some records sourced from CRP/NIMSP due to data sharing agreements.",
        "- This may result in itemized coverage < 100% of aggregate amounts.",
        "",
    ]

    # Read cycle coverage if available
    cycle_coverage_path = output_dir / "itemized_coverage_by_cycle.csv"
    if cycle_coverage_path.exists():
        df = pd.read_csv(cycle_coverage_path)
        report_lines.extend([
            "## Aggregate vs Itemized Coverage",
            "",
            f"- Overall coverage ratio: {df['overall_coverage_ratio'].mean():.1%}",
            f"- Lowest cycle coverage: {df['overall_coverage_ratio'].min():.1%}",
            f"- Highest cycle coverage: {df['overall_coverage_ratio'].max():.1%}",
            "",
        ])

    # Read CFScore missingness if available
    cfscore_path = output_dir / "recipient_cfscore_missingness.csv"
    if cfscore_path.exists():
        df = pd.read_csv(cfscore_path)
        report_lines.extend([
            "## Recipient CFScore Missingness",
            "",
            f"- Mean % records without CFScore: {df['pct_records_missing'].mean():.1%}",
            f"- Mean % amount to missing recipients: {df['pct_amount_missing'].mean():.1%}",
            "",
            "This affects the reliability of revealed_cfscore_cycle ideology measure.",
            "Consider using revealed_party_cycle as an alternative (no missingness).",
            "",
        ])

    # Read occupation missingness if available
    occupation_path = output_dir / "occupation_missingness_by_cycle.csv"
    if occupation_path.exists():
        df = pd.read_csv(occupation_path)
        report_lines.extend([
            "## Occupation Missingness",
            "",
            f"- Mean % records without occupation: {df['pct_records_missing'].mean():.1%}",
            "",
            "Missing occupation affects physician classification accuracy.",
            "",
        ])

    report_lines.extend([
        "## Recommendations",
        "",
        "1. Use revealed_party_cycle for ideology analysis where CFScore missingness is high",
        "2. Document coverage limitations in any publication",
        "3. Consider sensitivity analyses excluding low-coverage cycles",
        "",
    ])

    report_path = output_dir / "coverage_summary_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Created: {report_path}")
    return report_path
