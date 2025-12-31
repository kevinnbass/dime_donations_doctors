"""
Figure 1 Replication from Bonica et al. (2019).

Recreates the figure showing "Percentage of campaign contributions to Republicans
by year of medical school graduation and gender."

Reference: Bonica, A., Rosenthal, H., Rothman, D.J., & Siciliano, K. (2019).
"Physician activism in American politics: The opposition to the Price nomination."
PLOS ONE. DOI: 10.1371/journal.pone.0215802
"""

from pathlib import Path
from typing import Optional

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    DIME_DONORS_PARQUET,
    DIME_RECIPIENTS_PARQUET,
    DONOR_CYCLE_PANEL_PARTITIONED_DIR,
    PHYSICIAN_LABELS_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    NPPES_PHYSICIANS_PARQUET,
    PECOS_PHYSICIANS_PARQUET,
    PLOTS_DIR,
    CYCLES,
)


def compute_physician_republican_share(
    min_grad_year: int = 1950,
    max_grad_year: int = 2020,
) -> pd.DataFrame:
    """
    Compute Republican share of contributions for each physician.

    Republican share = R_dollars / (R_dollars + D_dollars)

    Returns DataFrame with columns:
    - bonica_cid: Donor ID
    - graduation_year: Medical school graduation year
    - gender: Gender (M/F)
    - total_rep_dollars: Total $ to Republicans
    - total_dem_dollars: Total $ to Democrats
    - rep_share: Republican share (0-1)

    Args:
        min_grad_year: Minimum graduation year to include
        max_grad_year: Maximum graduation year to include

    Returns:
        DataFrame with physician-level Republican share
    """
    if not PHYSICIAN_LABELS_PARQUET.exists():
        raise FileNotFoundError(f"Physician labels not found: {PHYSICIAN_LABELS_PARQUET}")
    if not DONOR_CYCLE_PANEL_PARTITIONED_DIR.exists():
        raise FileNotFoundError(f"Donor cycle panel not found: {DONOR_CYCLE_PANEL_PARTITIONED_DIR}")

    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")

    labels_path = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
    panel_path = str(DONOR_CYCLE_PANEL_PARTITIONED_DIR).replace("\\", "/")

    # Check if graduation_year exists in labels
    cols = con.execute(f"""
        SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{labels_path}'))
    """).fetchall()
    col_names = [c[0] for c in cols]

    if 'graduation_year' not in col_names:
        con.close()
        raise ValueError(
            "graduation_year column not found in physician_labels.parquet.\n"
            "Please run: python scripts/02c_add_pecos_linkage.py"
        )

    # Check for gender column
    # Gender can come from NPPES (via linkage) or DIME (contributor_gender)
    # For now, we'll use the panel's revealed_party_cycle to compute partisanship

    print("Computing Republican share for physicians with graduation year...")

    # The revealed_party_cycle is: +1 = Dem, -1 = Rep
    # So for Republican share, we need to aggregate the underlying contributions

    # First approach: Use the existing donor-level aggregates from DIME
    # But we need party-specific amounts, which requires going back to itemized data

    # Alternative: Use revealed_party_cycle as a proxy
    # revealed_party_cycle ranges from -1 (all Rep) to +1 (all Dem)
    # Republican share = (1 - revealed_party_cycle) / 2

    # Get physicians with graduation year and compute their average party score
    query = f"""
    WITH physician_donors AS (
        SELECT
            l.bonica_cid,
            l.graduation_year,
            -- Get gender from NPPES via linkage
            n.gender as nppes_gender
        FROM read_parquet('{labels_path}') l
        LEFT JOIN read_parquet('{str(LINKAGE_RESULTS_PARQUET).replace(chr(92), "/")}') lk
            ON l.bonica_cid = lk.bonica_cid
        LEFT JOIN read_parquet('{str(NPPES_PHYSICIANS_PARQUET).replace(chr(92), "/")}') n
            ON CAST(lk.nppes_npi AS VARCHAR) = CAST(n.npi AS VARCHAR)
        WHERE l.physician_final = true
          AND l.graduation_year IS NOT NULL
          AND l.graduation_year >= {min_grad_year}
          AND l.graduation_year <= {max_grad_year}
    ),
    donor_party AS (
        SELECT
            bonica_cid,
            AVG(revealed_party_cycle) as mean_party_score,
            SUM(total_amount) as total_amount,
            COUNT(*) as n_cycles
        FROM read_parquet('{panel_path}/*/*.parquet', hive_partitioning=true)
        WHERE revealed_party_cycle IS NOT NULL
        GROUP BY bonica_cid
    )
    SELECT
        p.bonica_cid,
        p.graduation_year,
        p.nppes_gender as gender,
        dp.mean_party_score,
        dp.total_amount,
        dp.n_cycles,
        -- Convert party score to Republican share
        -- party_score: -1 = all Rep, +1 = all Dem
        -- rep_share = (1 - party_score) / 2
        (1.0 - dp.mean_party_score) / 2.0 as rep_share
    FROM physician_donors p
    INNER JOIN donor_party dp ON p.bonica_cid = dp.bonica_cid
    WHERE dp.mean_party_score IS NOT NULL
    """

    result = con.execute(query).fetchdf()
    con.close()

    print(f"  Physicians with graduation year and party data: {len(result):,}")
    print(f"  Graduation year range: {result['graduation_year'].min()} - {result['graduation_year'].max()}")
    print(f"  Gender distribution: {result['gender'].value_counts().to_dict()}")

    return result


def aggregate_by_graduation_year_gender(
    df: pd.DataFrame,
    min_n: int = 50,
) -> pd.DataFrame:
    """
    Aggregate Republican share by graduation year and gender.

    Args:
        df: DataFrame from compute_physician_republican_share
        min_n: Minimum observations per cell

    Returns:
        DataFrame with columns: graduation_year, gender, mean_rep_share, n
    """
    # Group by graduation year and gender
    agg = df.groupby(['graduation_year', 'gender']).agg(
        mean_rep_share=('rep_share', 'mean'),
        median_rep_share=('rep_share', 'median'),
        n=('rep_share', 'count'),
        std_rep_share=('rep_share', 'std'),
    ).reset_index()

    # Also compute overall (all genders)
    agg_all = df.groupby('graduation_year').agg(
        mean_rep_share=('rep_share', 'mean'),
        median_rep_share=('rep_share', 'median'),
        n=('rep_share', 'count'),
        std_rep_share=('rep_share', 'std'),
    ).reset_index()
    agg_all['gender'] = 'All'

    # Combine
    result = pd.concat([agg, agg_all], ignore_index=True)

    # Filter by minimum n
    result = result[result['n'] >= min_n].copy()

    # Convert rep_share to percentage
    result['rep_pct'] = result['mean_rep_share'] * 100

    return result


def plot_figure1(
    agg_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = False,
    title: str = "Contributions to Republicans by Year of Medical School Graduation and Gender",
    smoothing: bool = True,
    window: int = 5,
) -> Optional[Path]:
    """
    Create Figure 1: Republican share by graduation year and gender.

    Args:
        agg_df: Aggregated DataFrame from aggregate_by_graduation_year_gender
        output_path: Path to save figure
        show: Whether to display figure
        title: Plot title
        smoothing: Apply rolling average smoothing
        window: Smoothing window size

    Returns:
        Output path if saved
    """
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color scheme matching typical academic style
    colors = {
        'All': '#333333',  # Dark gray
        'M': '#1f77b4',    # Blue
        'F': '#d62728',    # Red
    }

    linestyles = {
        'All': '-',
        'M': '--',
        'F': ':',
    }

    labels = {
        'All': 'All Physicians',
        'M': 'Male Physicians',
        'F': 'Female Physicians',
    }

    # Plot each gender
    for gender in ['All', 'M', 'F']:
        data = agg_df[agg_df['gender'] == gender].sort_values('graduation_year')

        if len(data) == 0:
            continue

        x = data['graduation_year']
        y = data['rep_pct']

        # Apply smoothing if requested
        if smoothing and len(data) >= window:
            y_smooth = data['rep_pct'].rolling(window=window, center=True, min_periods=1).mean()
            ax.plot(
                x, y_smooth,
                color=colors.get(gender, 'gray'),
                linestyle=linestyles.get(gender, '-'),
                linewidth=2.5,
                label=labels.get(gender, gender),
            )
            # Also plot raw data as faint points
            ax.scatter(x, y, color=colors.get(gender, 'gray'), alpha=0.2, s=10)
        else:
            ax.plot(
                x, y,
                color=colors.get(gender, 'gray'),
                linestyle=linestyles.get(gender, '-'),
                linewidth=2,
                marker='o',
                markersize=3,
                label=labels.get(gender, gender),
            )

    # Style the plot
    ax.set_xlabel("Year of Medical School Graduation", fontsize=14)
    ax.set_ylabel("Contributions to Republicans, %", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Add 50% reference line
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Set y-axis limits
    ax.set_ylim(0, 100)

    # Add legend
    ax.legend(loc='best', fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close(fig)
        return output_path

    if show:
        plt.show()

    plt.close(fig)
    return None


def generate_figure1_data(
    output_path: Optional[Path] = None,
    min_grad_year: int = 1950,
    max_grad_year: int = 2020,
) -> pd.DataFrame:
    """
    Generate the underlying data for Figure 1.

    Args:
        output_path: Optional path to save CSV
        min_grad_year: Minimum graduation year
        max_grad_year: Maximum graduation year

    Returns:
        Aggregated DataFrame
    """
    # Compute physician-level Republican share
    physician_df = compute_physician_republican_share(
        min_grad_year=min_grad_year,
        max_grad_year=max_grad_year,
    )

    # Aggregate by graduation year and gender
    agg_df = aggregate_by_graduation_year_gender(physician_df)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(output_path, index=False)
        print(f"Saved data: {output_path}")

    return agg_df


def replicate_figure1(
    output_dir: Optional[Path] = None,
    min_grad_year: int = 1950,
    max_grad_year: int = 2020,
    smoothing: bool = True,
) -> dict:
    """
    Full replication of Figure 1.

    Args:
        output_dir: Directory for outputs
        min_grad_year: Minimum graduation year
        max_grad_year: Maximum graduation year
        smoothing: Apply smoothing to lines

    Returns:
        Dictionary with paths to outputs
    """
    output_dir = output_dir or PLOTS_DIR / "figure1_replication"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Replicating Figure 1: Bonica et al. (2019)")
    print("=" * 60)

    # Generate data
    print("\n--- Step 1: Computing Republican share by physician ---")
    data_path = output_dir / "figure1_data.csv"
    agg_df = generate_figure1_data(
        output_path=data_path,
        min_grad_year=min_grad_year,
        max_grad_year=max_grad_year,
    )

    # Create main figure
    print("\n--- Step 2: Creating Figure 1 ---")
    fig_path = output_dir / "figure1_replication.png"
    plot_figure1(
        agg_df,
        output_path=fig_path,
        smoothing=smoothing,
    )

    # Create variant without smoothing
    fig_raw_path = output_dir / "figure1_replication_raw.png"
    plot_figure1(
        agg_df,
        output_path=fig_raw_path,
        smoothing=False,
        title="Contributions to Republicans by Graduation Year (Raw Data)",
    )

    # Summary statistics
    print("\n--- Summary Statistics ---")
    for gender in ['All', 'M', 'F']:
        subset = agg_df[agg_df['gender'] == gender]
        if len(subset) > 0:
            print(f"\n{gender}:")
            print(f"  Graduation years: {subset['graduation_year'].min()} - {subset['graduation_year'].max()}")
            print(f"  Total physicians: {subset['n'].sum():,}")
            print(f"  Mean Rep %: {subset['rep_pct'].mean():.1f}%")

            # Trend: compare early vs late graduates
            early = subset[subset['graduation_year'] <= 1980]['rep_pct'].mean()
            late = subset[subset['graduation_year'] >= 2000]['rep_pct'].mean()
            if not np.isnan(early) and not np.isnan(late):
                print(f"  Pre-1980 graduates: {early:.1f}%")
                print(f"  Post-2000 graduates: {late:.1f}%")
                print(f"  Change: {late - early:+.1f} percentage points")

    outputs = {
        "figure": fig_path,
        "figure_raw": fig_raw_path,
        "data": data_path,
    }

    print("\n" + "=" * 60)
    print("Figure 1 replication complete!")
    print("=" * 60)

    return outputs
