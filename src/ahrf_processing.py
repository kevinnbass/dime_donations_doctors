"""
AHRF data processing module.

Ingests HRSA Area Health Resources Files and provides
physician population denominators for donation rate analysis.
"""

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from .config import (
    AHRF_RAW_DIR,
    AHRF_STATE_PARQUET,
    AHRF_COUNT_SUFFIX,
    PHYSICIAN_LABELS_PARQUET,
    DIME_DONORS_PARQUET,
    PROCESSED_DATA_DIR,
)


def find_ahrf_hp_file() -> Path:
    """Locate the AHRF health professions file."""
    # Look in the extracted subdirectory
    patterns = [
        "NCHWA-*/AHRF*hp.csv",
        "*/AHRF*hp.csv",
        "AHRF*hp.csv",
    ]

    for pattern in patterns:
        matches = list(AHRF_RAW_DIR.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No AHRF health professions file found in {AHRF_RAW_DIR}. "
        "Expected file matching pattern: AHRF*hp.csv"
    )


def find_ahrf_geo_file() -> Path:
    """Locate the AHRF geography file."""
    patterns = [
        "NCHWA-*/AHRF*geo.csv",
        "*/AHRF*geo.csv",
        "AHRF*geo.csv",
    ]

    for pattern in patterns:
        matches = list(AHRF_RAW_DIR.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No AHRF geography file found in {AHRF_RAW_DIR}. "
        "Expected file matching pattern: AHRF*geo.csv"
    )


def ingest_ahrf_data(
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Ingest AHRF data and aggregate to state level.

    Joins HP file with geo file, extracts physician counts,
    aggregates to state level, and saves as parquet.

    Args:
        output_path: Output parquet path (default: AHRF_STATE_PARQUET)
        overwrite: Whether to overwrite existing file

    Returns:
        Path to created parquet file
    """
    output_path = output_path or AHRF_STATE_PARQUET

    if output_path.exists() and not overwrite:
        print(f"AHRF state parquet already exists: {output_path}")
        return output_path

    hp_file = find_ahrf_hp_file()
    geo_file = find_ahrf_geo_file()

    print(f"Ingesting AHRF data:")
    print(f"  HP file: {hp_file.name}")
    print(f"  Geo file: {geo_file.name}")

    con = duckdb.connect()

    hp_path = str(hp_file).replace("\\", "/")
    geo_path = str(geo_file).replace("\\", "/")
    output_path_str = str(output_path).replace("\\", "/")

    suffix = AHRF_COUNT_SUFFIX  # "23" for 2023 data

    # Check available columns in HP file
    cols_result = con.execute(f"""
        SELECT column_name FROM (DESCRIBE SELECT * FROM read_csv_auto('{hp_path}', header=true) LIMIT 1)
    """).fetchdf()
    available_cols = set(cols_result["column_name"].str.lower())

    # Define columns to extract (with fallbacks)
    md_col = f"md_nf_activ_{suffix}" if f"md_nf_activ_{suffix}" in available_cols else f"md_nf_{suffix}"
    do_col = f"do_nf_activ_{suffix}" if f"do_nf_activ_{suffix}" in available_cols else f"do_nf_incl_inactv_{suffix}"

    print(f"  Using MD column: {md_col}")
    print(f"  Using DO column: {do_col}")

    # Aggregate to state level
    query = f"""
    COPY (
        WITH hp_data AS (
            SELECT
                fips_st_cnty,
                COALESCE(TRY_CAST("{md_col}" AS INTEGER), 0) as md_count,
                COALESCE(TRY_CAST("{do_col}" AS INTEGER), 0) as do_count,
                COALESCE(TRY_CAST("md_nf_all_pc_{suffix}" AS INTEGER), 0) as md_patient_care,
                COALESCE(TRY_CAST("do_nf_all_pc_{suffix}" AS INTEGER), 0) as do_patient_care
            FROM read_csv_auto('{hp_path}', header=true)
        ),
        geo_data AS (
            SELECT
                fips_st_cnty,
                st_name_abbrev as state,
                fips_st as state_fips
            FROM read_csv_auto('{geo_path}', header=true)
        ),
        joined AS (
            SELECT
                g.state,
                g.state_fips,
                h.md_count,
                h.do_count,
                h.md_patient_care,
                h.do_patient_care
            FROM hp_data h
            JOIN geo_data g ON h.fips_st_cnty = g.fips_st_cnty
            WHERE g.state IS NOT NULL AND g.state != ''
        )
        SELECT
            state,
            state_fips,
            SUM(md_count) as total_mds,
            SUM(do_count) as total_dos,
            SUM(md_count) + SUM(do_count) as total_active_physicians,
            SUM(md_patient_care) as md_patient_care,
            SUM(do_patient_care) as do_patient_care,
            SUM(md_patient_care) + SUM(do_patient_care) as total_patient_care,
            COUNT(*) as n_counties
        FROM joined
        GROUP BY state, state_fips
        ORDER BY state
    ) TO '{output_path_str}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(query)

    # Get summary
    stats = con.execute(f"""
        SELECT
            COUNT(*) as n_states,
            SUM(total_active_physicians) as total_physicians,
            SUM(total_mds) as total_mds,
            SUM(total_dos) as total_dos
        FROM read_parquet('{output_path_str}')
    """).fetchone()

    con.close()

    print(f"Created: {output_path}")
    print(f"  States/territories: {stats[0]}")
    print(f"  Total active physicians: {stats[1]:,}")
    print(f"    MDs: {stats[2]:,}")
    print(f"    DOs: {stats[3]:,}")

    return output_path


def get_state_physician_counts() -> pd.DataFrame:
    """
    Load state-level physician population counts.

    Returns:
        DataFrame with columns:
            - state: 2-letter state abbreviation
            - total_active_physicians: Active MD + DO count
            - total_mds: Total MDs (non-federal)
            - total_dos: Total DOs (non-federal)
            - total_patient_care: MDs + DOs in patient care
    """
    if not AHRF_STATE_PARQUET.exists():
        raise FileNotFoundError(
            f"AHRF state parquet not found: {AHRF_STATE_PARQUET}. "
            "Run ingest_ahrf_data() first."
        )

    con = duckdb.connect()
    path = str(AHRF_STATE_PARQUET).replace("\\", "/")

    df = con.execute(f"""
        SELECT
            state,
            total_active_physicians,
            total_mds,
            total_dos,
            total_patient_care
        FROM read_parquet('{path}')
    """).fetchdf()

    con.close()
    return df


def compute_donation_rates_by_state(
    cycle: Optional[int] = None,
    physician_definition: str = "final",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute physician donation rates by state.

    Args:
        cycle: Optional election cycle (if None, uses all cycles)
        physician_definition: 'naive', 'rule', or 'final'
        output_path: Optional path to save results

    Returns:
        DataFrame with columns:
            - state
            - n_donating_physicians
            - total_physicians (from AHRF)
            - donation_rate (n_donating / total)
            - cycle (if cycle specified)
    """
    if not AHRF_STATE_PARQUET.exists():
        raise FileNotFoundError("AHRF state data not found. Run ingest_ahrf_data() first.")

    if not PHYSICIAN_LABELS_PARQUET.exists():
        raise FileNotFoundError("Physician labels not found.")

    if not DIME_DONORS_PARQUET.exists():
        raise FileNotFoundError("DIME donors not found.")

    physician_col = f"physician_{physician_definition}"

    con = duckdb.connect()

    ahrf_p = str(AHRF_STATE_PARQUET).replace("\\", "/")
    labels_p = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
    donors_p = str(DIME_DONORS_PARQUET).replace("\\", "/")

    # Count donating physicians by state
    query = f"""
        WITH physician_donors AS (
            SELECT
                d.contributor_state as state,
                COUNT(DISTINCT d.bonica_cid) as n_donating_physicians
            FROM read_parquet('{donors_p}') d
            JOIN read_parquet('{labels_p}') l ON d.bonica_cid = l.bonica_cid
            WHERE l.{physician_col} = true
              AND d.contributor_state IS NOT NULL
              AND LENGTH(d.contributor_state) = 2
            GROUP BY d.contributor_state
        ),
        state_totals AS (
            SELECT
                state,
                total_active_physicians
            FROM read_parquet('{ahrf_p}')
        )
        SELECT
            COALESCE(pd.state, st.state) as state,
            COALESCE(pd.n_donating_physicians, 0) as n_donating_physicians,
            st.total_active_physicians as total_physicians,
            CASE
                WHEN st.total_active_physicians > 0
                THEN CAST(pd.n_donating_physicians AS DOUBLE) / st.total_active_physicians
                ELSE NULL
            END as donation_rate
        FROM physician_donors pd
        FULL OUTER JOIN state_totals st ON pd.state = st.state
        WHERE COALESCE(pd.state, st.state) IS NOT NULL
        ORDER BY state
    """

    df = con.execute(query).fetchdf()
    con.close()

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Created: {output_path}")

    return df


def compute_state_weights(
    donation_rates: pd.DataFrame,
    weight_type: str = "population",
) -> pd.DataFrame:
    """
    Compute sampling weights based on donation rates or population.

    Args:
        donation_rates: Output from compute_donation_rates_by_state
        weight_type:
            - 'inverse_rate': 1 / donation_rate (more weight to underrepresented)
            - 'population': total_physicians (weight by population size)
            - 'rake': normalized inverse rate

    Returns:
        DataFrame with state and weight columns
    """
    df = donation_rates.copy()

    if weight_type == "population":
        # Weight by population share
        total = df["total_physicians"].sum()
        df["weight"] = df["total_physicians"] / total

    elif weight_type == "inverse_rate":
        # Inverse of donation rate (underrepresented states get higher weight)
        # Avoid division by zero
        df["weight"] = 1.0 / df["donation_rate"].clip(lower=0.001)
        # Normalize
        df["weight"] = df["weight"] / df["weight"].sum()

    elif weight_type == "rake":
        # Raking: adjust so weighted donation rate equals target
        # Target is the overall donation rate
        overall_rate = (
            df["n_donating_physicians"].sum() /
            df["total_physicians"].sum()
        )
        df["weight"] = df["total_physicians"] * overall_rate / df["n_donating_physicians"].clip(lower=1)
        df["weight"] = df["weight"] / df["weight"].sum()

    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    return df[["state", "weight"]]


def analyze_representativeness(
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Analyze representativeness of DIME physician sample.

    Returns dictionary with:
        - 'donation_rates': State-level donation rates
        - 'summary_stats': Overall representativeness metrics
        - 'over_under_represented': States sorted by representation ratio
    """
    output_dir = output_dir or PROCESSED_DATA_DIR

    donation_rates = compute_donation_rates_by_state()

    # Calculate representation index
    # States with higher donation rates are over-represented in donor sample
    mean_rate = donation_rates["donation_rate"].mean()
    donation_rates["representation_index"] = donation_rates["donation_rate"] / mean_rate

    # Summary statistics
    summary = {
        "n_states": len(donation_rates),
        "overall_donation_rate": (
            donation_rates["n_donating_physicians"].sum() /
            donation_rates["total_physicians"].sum()
        ),
        "mean_state_rate": donation_rates["donation_rate"].mean(),
        "median_state_rate": donation_rates["donation_rate"].median(),
        "std_state_rate": donation_rates["donation_rate"].std(),
        "min_rate_state": donation_rates.loc[donation_rates["donation_rate"].idxmin(), "state"],
        "max_rate_state": donation_rates.loc[donation_rates["donation_rate"].idxmax(), "state"],
    }

    # Over/under represented states
    over_rep = donation_rates[donation_rates["representation_index"] > 1.2].sort_values(
        "representation_index", ascending=False
    )
    under_rep = donation_rates[donation_rates["representation_index"] < 0.8].sort_values(
        "representation_index"
    )

    return {
        "donation_rates": donation_rates,
        "summary_stats": summary,
        "over_represented": over_rep,
        "under_represented": under_rep,
    }


def get_ahrf_stats() -> dict:
    """Get statistics from the processed AHRF data."""
    if not AHRF_STATE_PARQUET.exists():
        return {}

    con = duckdb.connect()
    path = str(AHRF_STATE_PARQUET).replace("\\", "/")

    stats = {}
    stats["n_states"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["total_physicians"] = con.execute(
        f"SELECT SUM(total_active_physicians) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["total_mds"] = con.execute(
        f"SELECT SUM(total_mds) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["total_dos"] = con.execute(
        f"SELECT SUM(total_dos) FROM read_parquet('{path}')"
    ).fetchone()[0]

    # Top states by physician count
    top_states = con.execute(f"""
        SELECT state, total_active_physicians
        FROM read_parquet('{path}')
        ORDER BY total_active_physicians DESC
        LIMIT 10
    """).fetchdf()

    stats["top_states"] = top_states.to_dict("records")

    con.close()
    return stats
