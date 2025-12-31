#!/usr/bin/env python3
"""
Yearly Contribution Analysis by Physician Pool.

Shows contributions by calendar year (election cycle), not by first donation year.
This shows WHEN contributions were made, not WHEN donors first contributed.

Usage:
    python scripts/34_yearly_contributions_by_pool.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import duckdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ensure_directories,
    PROCESSED_DATA_DIR,
    PHYSICIAN_LABELS_PARQUET,
    PLOTS_DIR,
    TABLES_DIR,
)


def get_pool_sql_filters():
    """Return SQL WHERE clauses for each pool.

    Uses p.occupation_cycle from panel (available 1980-2024) instead of
    d.contributor_occupation from dime_donors (only available through 2018).
    """
    return {
        "doctor_physician": {
            "name": "Doctor/Physician Keyword",
            "type": "occupation",  # Filter on panel occupation_cycle
            "filter": """
                LOWER(p.occupation_cycle) LIKE '%doctor%'
                OR LOWER(p.occupation_cycle) LIKE '%physician%'
            """,
            "color": "blue"
        },
        "md_do_credential": {
            "name": "MD/DO Credential",
            "type": "occupation",
            "filter": """
                (p.occupation_cycle LIKE '% MD%'
                OR p.occupation_cycle LIKE '%,MD%'
                OR UPPER(p.occupation_cycle) = 'MD'
                OR p.occupation_cycle LIKE '% DO%'
                OR UPPER(p.occupation_cycle) = 'DO')
                AND LOWER(p.occupation_cycle) NOT LIKE '%phd%'
            """,
            "color": "green"
        },
        "specialists": {
            "name": "Medical Specialists",
            "type": "occupation",
            "filter": """
                LOWER(p.occupation_cycle) LIKE '%surgeon%'
                OR LOWER(p.occupation_cycle) LIKE '%cardiologist%'
                OR LOWER(p.occupation_cycle) LIKE '%anesthesiologist%'
                OR LOWER(p.occupation_cycle) LIKE '%radiologist%'
                OR LOWER(p.occupation_cycle) LIKE '%oncologist%'
                OR LOWER(p.occupation_cycle) LIKE '%psychiatrist%'
                OR LOWER(p.occupation_cycle) LIKE '%pediatrician%'
            """,
            "color": "red"
        },
        "cms_medicare": {
            "name": "CMS Medicare (thru 2018)",
            "type": "labels",  # Filter on physician_labels - limited to 2018
            "filter": "l.cms_medicare_active = true",
            "color": "purple"
        },
        "tier1_fixed": {
            "name": "Broad Physician Rules",
            "type": "occupation",  # Apply tier1-like rules directly to occupation_cycle
            "filter": """
                (
                    p.occupation_cycle LIKE '% MD%' OR UPPER(p.occupation_cycle) = 'MD'
                    OR p.occupation_cycle LIKE '% DO%' OR UPPER(p.occupation_cycle) = 'DO'
                    OR LOWER(p.occupation_cycle) LIKE '%surgeon%'
                    OR LOWER(p.occupation_cycle) LIKE '%cardiologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%anesthesiologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%radiologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%oncologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%neurologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%psychiatrist%'
                    OR LOWER(p.occupation_cycle) LIKE '%dermatologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%gastroenterologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%urologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%ophthalmologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%pulmonologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%nephrologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%endocrinologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%rheumatologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%pathologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%obstetrician%'
                    OR LOWER(p.occupation_cycle) LIKE '%gynecologist%'
                    OR LOWER(p.occupation_cycle) LIKE '%pediatrician%'
                    OR LOWER(p.occupation_cycle) LIKE '%internist%'
                    OR LOWER(p.occupation_cycle) LIKE '%hospitalist%'
                    OR LOWER(p.occupation_cycle) LIKE '%physician%'
                    OR LOWER(p.occupation_cycle) LIKE '%doctor%'
                )
                AND LOWER(p.occupation_cycle) NOT LIKE '%nurse%'
                AND LOWER(p.occupation_cycle) NOT LIKE '%speech%'
                AND LOWER(p.occupation_cycle) NOT LIKE '%phd%'
                AND LOWER(p.occupation_cycle) NOT LIKE '%chiropract%'
                AND LOWER(p.occupation_cycle) NOT LIKE '%dentist%'
                AND LOWER(p.occupation_cycle) NOT LIKE '%veterinar%'
                AND LOWER(p.occupation_cycle) NOT LIKE '%psycholog%'
            """,
            "color": "orange"
        }
    }


def analyze_yearly_contributions(con, pool_id: str, pool_def: dict):
    """Analyze yearly contributions for a single pool.

    Uses occupation_cycle from panel (1980-2024) for occupation-based pools.
    Uses physician_labels for CMS/tier1 pools.
    """

    labels_path = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
    panel_path = str(PROCESSED_DATA_DIR / "parquet" / "donor_cycle_panel").replace("\\", "/")

    pool_type = pool_def.get("type", "occupation")

    if pool_type == "occupation":
        # Filter directly on panel occupation_cycle
        query = f"""
        SELECT
            p.cycle,
            COUNT(*) as n_donations,
            SUM(p.total_amount) as total_dollars,
            AVG(p.revealed_party_cycle) as avg_party_score,
            AVG((1.0 - p.revealed_party_cycle) / 2.0) as rep_share,
            SUM(CASE WHEN p.revealed_party_cycle < 0 THEN p.total_amount ELSE 0 END) as rep_dollars,
            SUM(CASE WHEN p.revealed_party_cycle > 0 THEN p.total_amount ELSE 0 END) as dem_dollars,
            SUM(CASE WHEN p.revealed_party_cycle < 0 THEN 1 ELSE 0 END) as rep_donations,
            SUM(CASE WHEN p.revealed_party_cycle > 0 THEN 1 ELSE 0 END) as dem_donations
        FROM read_parquet('{panel_path}/*/*.parquet', hive_partitioning=true) p
        WHERE ({pool_def['filter']})
          AND p.revealed_party_cycle IS NOT NULL
          AND p.total_amount > 0
        GROUP BY p.cycle
        ORDER BY p.cycle
        """

    elif pool_type == "labels":
        # Join to labels, filter by labels field
        query = f"""
        SELECT
            p.cycle,
            COUNT(*) as n_donations,
            SUM(p.total_amount) as total_dollars,
            AVG(p.revealed_party_cycle) as avg_party_score,
            AVG((1.0 - p.revealed_party_cycle) / 2.0) as rep_share,
            SUM(CASE WHEN p.revealed_party_cycle < 0 THEN p.total_amount ELSE 0 END) as rep_dollars,
            SUM(CASE WHEN p.revealed_party_cycle > 0 THEN p.total_amount ELSE 0 END) as dem_dollars,
            SUM(CASE WHEN p.revealed_party_cycle < 0 THEN 1 ELSE 0 END) as rep_donations,
            SUM(CASE WHEN p.revealed_party_cycle > 0 THEN 1 ELSE 0 END) as dem_donations
        FROM read_parquet('{panel_path}/*/*.parquet', hive_partitioning=true) p
        INNER JOIN read_parquet('{labels_path}') l
            ON CAST(p.bonica_cid AS VARCHAR) = CAST(l.bonica_cid AS VARCHAR)
        WHERE {pool_def['filter']}
          AND p.revealed_party_cycle IS NOT NULL
          AND p.total_amount > 0
        GROUP BY p.cycle
        ORDER BY p.cycle
        """

    elif pool_type == "labels_with_exclusion":
        # Join to labels + apply occupation exclusions on panel
        exclusion = pool_def.get("exclusion", "")
        query = f"""
        SELECT
            p.cycle,
            COUNT(*) as n_donations,
            SUM(p.total_amount) as total_dollars,
            AVG(p.revealed_party_cycle) as avg_party_score,
            AVG((1.0 - p.revealed_party_cycle) / 2.0) as rep_share,
            SUM(CASE WHEN p.revealed_party_cycle < 0 THEN p.total_amount ELSE 0 END) as rep_dollars,
            SUM(CASE WHEN p.revealed_party_cycle > 0 THEN p.total_amount ELSE 0 END) as dem_dollars,
            SUM(CASE WHEN p.revealed_party_cycle < 0 THEN 1 ELSE 0 END) as rep_donations,
            SUM(CASE WHEN p.revealed_party_cycle > 0 THEN 1 ELSE 0 END) as dem_donations
        FROM read_parquet('{panel_path}/*/*.parquet', hive_partitioning=true) p
        INNER JOIN read_parquet('{labels_path}') l
            ON CAST(p.bonica_cid AS VARCHAR) = CAST(l.bonica_cid AS VARCHAR)
        WHERE {pool_def['filter']}
          {exclusion}
          AND p.revealed_party_cycle IS NOT NULL
          AND p.total_amount > 0
        GROUP BY p.cycle
        ORDER BY p.cycle
        """

    else:
        raise ValueError(f"Unknown pool type: {pool_type}")

    df = con.execute(query).fetchdf()
    df["pool"] = pool_id
    df["pool_name"] = pool_def["name"]

    return df


def main():
    ensure_directories()

    print("=" * 70)
    print("YEARLY CONTRIBUTION ANALYSIS BY PHYSICIAN POOL")
    print("(Contributions by calendar year, not first donation year)")
    print("=" * 70)

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    pools = get_pool_sql_filters()
    all_results = []

    for pool_id, pool_def in pools.items():
        print(f"\nAnalyzing: {pool_def['name']}...")

        try:
            df = analyze_yearly_contributions(con, pool_id, pool_def)
            total_dollars = df["total_dollars"].sum()
            weighted_rep = (df["rep_share"] * df["total_dollars"]).sum() / total_dollars
            print(f"  Total donations: {df['n_donations'].sum():,}")
            print(f"  Total dollars: ${total_dollars/1e9:.2f}B")
            print(f"  Weighted Rep %: {weighted_rep*100:.1f}%")
            all_results.append(df)
        except Exception as e:
            print(f"  ERROR: {e}")

    con.close()

    # Combine results
    combined = pd.concat(all_results, ignore_index=True)

    # Save data
    output_csv = TABLES_DIR / "yearly_contributions_by_pool.csv"
    combined.to_csv(output_csv, index=False)
    print(f"\nSaved data to {output_csv}")

    # Create plots
    plot_yearly_comparison(combined, pools, PLOTS_DIR / "yearly_contributions_all_pools.png")
    plot_yearly_comparison(combined, pools, PLOTS_DIR / "yearly_contributions_no_specialists.png",
                          exclude_pools=["specialists", "cms_medicare"])
    plot_dollar_amounts(combined, pools, PLOTS_DIR / "yearly_dollars_by_party.png")
    plot_donation_counts(combined, pools, PLOTS_DIR)  # Creates one plot per pool

    # Print summary table
    print("\n" + "=" * 70)
    print("% REPUBLICAN BY ELECTION CYCLE")
    print("=" * 70)

    print(f"\n{'Cycle':<6}", end="")
    for pool_id in pools.keys():
        short_name = pool_id[:12]
        print(f" {short_name:<12}", end="")
    print()
    print("-" * 80)

    for cycle in [1980, 1990, 2000, 2008, 2012, 2016, 2020, 2024]:
        row = f"{cycle:<6}"
        for pool_id in pools.keys():
            d = combined[(combined["cycle"] == cycle) & (combined["pool"] == pool_id)]
            if len(d) > 0:
                val = d.iloc[0]["rep_share"] * 100
                row += f" {val:5.1f}%      "
            else:
                row += "  N/A        "
        print(row)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


def plot_yearly_comparison(data: pd.DataFrame, pools: dict, output_path: Path, exclude_pools: list = None):
    """Create comparison plot of % Republican by year for all pools.

    Args:
        data: DataFrame with pool data
        pools: Dict of pool definitions
        output_path: Where to save the plot
        exclude_pools: List of pool_ids to exclude from the plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not available")
        return

    if exclude_pools is None:
        exclude_pools = []

    fig, ax = plt.subplots(figsize=(14, 8))

    for pool_id, pool_def in pools.items():
        if pool_id in exclude_pools:
            continue

        pool_data = data[data["pool"] == pool_id].copy()
        if len(pool_data) == 0:
            continue

        pool_data = pool_data.sort_values("cycle")
        total_n = int(pool_data["n_donations"].sum())

        # Format sample size with K or M suffix for readability
        if total_n >= 1_000_000:
            n_str = f"{total_n/1_000_000:.1f}M"
        elif total_n >= 1_000:
            n_str = f"{total_n/1_000:.0f}K"
        else:
            n_str = str(total_n)

        ax.plot(
            pool_data["cycle"],
            pool_data["rep_share"] * 100,
            color=pool_def["color"],
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{pool_def['name']} (n={n_str})",
            alpha=0.8
        )

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Election Cycle", fontsize=12)
    ax.set_ylabel("% Contributions to Republicans", fontsize=12)
    ax.set_title("Physician Political Contributions by Election Cycle\n(All donations in each year, not cohort-based)", fontsize=13)
    ax.set_xlim(1978, 2026)
    # Use 70 for simplified version, 80 for full version
    if exclude_pools and len(exclude_pools) > 0:
        ax.set_ylim(10, 70)
    else:
        ax.set_ylim(10, 80)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_dollar_amounts(data: pd.DataFrame, pools: dict, output_path: Path):
    """Create plot showing total dollar contributions over time."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not available")
        return

    # Use just doctor_physician pool for dollar amounts
    pool_data = data[data["pool"] == "doctor_physician"].copy()
    pool_data = pool_data.sort_values("cycle")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Stack bar chart of Rep vs Dem dollars
    ax.bar(pool_data["cycle"], pool_data["rep_dollars"] / 1e6, label="Republican", color="red", alpha=0.7)
    ax.bar(pool_data["cycle"], pool_data["dem_dollars"] / 1e6, bottom=pool_data["rep_dollars"] / 1e6,
           label="Democrat", color="blue", alpha=0.7)

    ax.set_xlabel("Election Cycle", fontsize=12)
    ax.set_ylabel("Total Contributions ($ Millions)", fontsize=12)
    ax.set_title("Physician Political Contributions by Party\n(Doctor/Physician Keyword Pool)", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_donation_counts(data: pd.DataFrame, pools: dict, output_dir: Path):
    """Create plots showing number of donations by party over time for each pool."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not available")
        return

    # Create a plot for each pool
    for pool_id, pool_def in pools.items():
        pool_data = data[data["pool"] == pool_id].copy()
        if len(pool_data) == 0:
            continue
        pool_data = pool_data.sort_values("cycle")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Stack bar chart of Rep vs Dem donation counts
        ax.bar(pool_data["cycle"], pool_data["rep_donations"] / 1000, label="Republican", color="red", alpha=0.7)
        ax.bar(pool_data["cycle"], pool_data["dem_donations"] / 1000, bottom=pool_data["rep_donations"] / 1000,
               label="Democrat", color="blue", alpha=0.7)

        ax.set_xlabel("Election Cycle", fontsize=12)
        ax.set_ylabel("Number of Donations (Thousands)", fontsize=12)
        ax.set_title(f"Physician Political Donations by Party\n({pool_def['name']})", fontsize=13)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path = output_dir / f"yearly_donations_{pool_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
