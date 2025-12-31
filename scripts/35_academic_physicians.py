#!/usr/bin/env python3
"""
Academic Physicians Analysis.

Compares political contributions of:
1. All Physicians (baseline)
2. University-Affiliated Physicians
3. Medical/Health Professors

Generates both contribution-weighted and donor-weighted analyses.

Usage:
    python scripts/35_academic_physicians.py
"""

import sys
from pathlib import Path

import pandas as pd
import duckdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ensure_directories,
    PROCESSED_DATA_DIR,
    PLOTS_DIR,
    TABLES_DIR,
)


def get_academic_pool_filters():
    """Return SQL WHERE clauses for academic physician pools."""
    return {
        "All Physicians": {
            "filter": """
                LOWER(p.occupation_cycle) LIKE '%doctor%'
                OR LOWER(p.occupation_cycle) LIKE '%physician%'
            """,
            "color": "blue"
        },
        "Academic Physicians": {
            "filter": """
                (LOWER(p.occupation_cycle) LIKE '%physician%'
                 OR LOWER(p.occupation_cycle) LIKE '%doctor%')
                AND (LOWER(p.employer_cycle) LIKE '%university%'
                     OR LOWER(p.employer_cycle) LIKE '%medical school%'
                     OR LOWER(p.employer_cycle) LIKE '%school of medicine%'
                     OR LOWER(p.employer_cycle) LIKE '%college of medicine%'
                     OR LOWER(p.occupation_cycle) LIKE '%professor%')
            """,
            "color": "red"
        },
        "Medical/Health Professors": {
            "filter": """
                LOWER(p.occupation_cycle) LIKE '%professor%'
                AND (LOWER(p.occupation_cycle) LIKE '%medic%'
                     OR LOWER(p.occupation_cycle) LIKE '%health%'
                     OR LOWER(p.occupation_cycle) LIKE '%physician%'
                     OR LOWER(p.occupation_cycle) LIKE '%doctor%'
                     OR LOWER(p.employer_cycle) LIKE '%medical%'
                     OR LOWER(p.employer_cycle) LIKE '%school of medicine%'
                     OR LOWER(p.employer_cycle) LIKE '%college of medicine%')
            """,
            "color": "darkgreen"
        }
    }


def analyze_contribution_weighted(con, pool_name: str, pool_def: dict):
    """Analyze contributions weighted by dollar amount."""
    panel_path = str(PROCESSED_DATA_DIR / "parquet" / "donor_cycle_panel").replace("\\", "/")

    query = f"""
    SELECT
        p.cycle,
        AVG((1.0 - p.revealed_party_cycle) / 2.0) as rep_share,
        COUNT(*) as n
    FROM read_parquet('{panel_path}/*/*.parquet', hive_partitioning=true) p
    WHERE ({pool_def['filter']})
      AND p.revealed_party_cycle IS NOT NULL
      AND p.total_amount > 0
    GROUP BY p.cycle
    ORDER BY p.cycle
    """

    df = con.execute(query).fetchdf()
    df["pool"] = pool_name

    return df


def analyze_donor_weighted(con, pool_name: str, pool_def: dict):
    """Analyze by unique donors (each donor counts once per cycle)."""
    panel_path = str(PROCESSED_DATA_DIR / "parquet" / "donor_cycle_panel").replace("\\", "/")

    query = f"""
    WITH donor_party AS (
        SELECT
            p.cycle,
            p.bonica_cid,
            -- For each donor-cycle, determine if they're R or D based on revealed_party
            CASE WHEN p.revealed_party_cycle < 0 THEN 1 ELSE 0 END as is_rep,
            CASE WHEN p.revealed_party_cycle > 0 THEN 1 ELSE 0 END as is_dem
        FROM read_parquet('{panel_path}/*/*.parquet', hive_partitioning=true) p
        WHERE ({pool_def['filter']})
          AND p.revealed_party_cycle IS NOT NULL
          AND p.total_amount > 0
    )
    SELECT
        cycle,
        CAST(SUM(is_rep) AS FLOAT) / NULLIF(SUM(is_rep) + SUM(is_dem), 0) as rep_share,
        SUM(is_rep) as rep_donors,
        SUM(is_dem) as dem_donors,
        COUNT(*) as n_donors
    FROM donor_party
    GROUP BY cycle
    ORDER BY cycle
    """

    df = con.execute(query).fetchdf()
    df["pool"] = pool_name

    return df


def main():
    ensure_directories()

    print("=" * 70)
    print("ACADEMIC PHYSICIANS ANALYSIS")
    print("=" * 70)

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    pools = get_academic_pool_filters()

    # Contribution-weighted analysis
    print("\n--- Contribution-Weighted Analysis ---")
    contribution_results = []
    for pool_name, pool_def in pools.items():
        print(f"\nAnalyzing: {pool_name}...")
        try:
            df = analyze_contribution_weighted(con, pool_name, pool_def)
            total_n = df["n"].sum()
            weighted_rep = (df["rep_share"] * df["n"]).sum() / total_n
            print(f"  Total observations: {total_n:,}")
            print(f"  Weighted Rep %: {weighted_rep*100:.1f}%")
            contribution_results.append(df)
        except Exception as e:
            print(f"  ERROR: {e}")

    contribution_df = pd.concat(contribution_results, ignore_index=True)
    output_csv = TABLES_DIR / "academic_physicians_full.csv"
    contribution_df.to_csv(output_csv, index=False)
    print(f"\nSaved contribution-weighted data to {output_csv}")

    # Donor-weighted analysis
    print("\n--- Donor-Weighted Analysis ---")
    donor_results = []
    for pool_name, pool_def in pools.items():
        print(f"\nAnalyzing: {pool_name}...")
        try:
            df = analyze_donor_weighted(con, pool_name, pool_def)
            total_donors = df["n_donors"].sum()
            # Weight by number of donors per cycle
            weighted_rep = (df["rep_share"] * df["n_donors"]).sum() / total_donors
            print(f"  Total unique donors: {total_donors:,}")
            print(f"  Weighted Rep %: {weighted_rep*100:.1f}%")
            donor_results.append(df)
        except Exception as e:
            print(f"  ERROR: {e}")

    donor_df = pd.concat(donor_results, ignore_index=True)
    output_csv = TABLES_DIR / "academic_physicians_donor_weighted.csv"
    donor_df.to_csv(output_csv, index=False)
    print(f"\nSaved donor-weighted data to {output_csv}")

    con.close()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: % REPUBLICAN BY ELECTION CYCLE")
    print("=" * 70)

    print("\nContribution-Weighted:")
    for cycle in [1990, 2000, 2008, 2012, 2016, 2020, 2024]:
        row = f"{cycle}:  "
        for pool_name in pools.keys():
            d = contribution_df[(contribution_df["cycle"] == cycle) & (contribution_df["pool"] == pool_name)]
            if len(d) > 0 and d.iloc[0]["n"] >= 50:
                val = d.iloc[0]["rep_share"] * 100
                row += f"{pool_name[:15]}: {val:5.1f}%  |  "
            else:
                row += f"{pool_name[:15]}:   N/A  |  "
        print(row)

    print("\nDonor-Weighted:")
    for cycle in [1990, 2000, 2008, 2012, 2016, 2020, 2024]:
        row = f"{cycle}:  "
        for pool_name in pools.keys():
            d = donor_df[(donor_df["cycle"] == cycle) & (donor_df["pool"] == pool_name)]
            if len(d) > 0 and d.iloc[0]["n_donors"] >= 50:
                val = d.iloc[0]["rep_share"] * 100
                row += f"{pool_name[:15]}: {val:5.1f}%  |  "
            else:
                row += f"{pool_name[:15]}:   N/A  |  "
        print(row)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
