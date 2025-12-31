"""
Figure 1 Generation from Party Shares.

Creates Figure 1 (Contributions to Republicans by graduation year and gender)
using p.to.rep computed from itemized contribution data.

This allows us to reproduce Bonica et al. (2019) methodology exactly.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    PROCESSED_DATA_DIR,
    PLOTS_DIR,
    PECOS_PHYSICIANS_PARQUET,
    LINKAGE_RESULTS_PARQUET,
)


def load_party_shares(
    path: Optional[Path] = None,
    period: str = "1992_2016",
) -> pd.DataFrame:
    """Load computed party shares."""
    if path is None:
        path = PROCESSED_DATA_DIR / f"donor_party_shares_{period}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Party shares not found: {path}\n"
            "Run: python scripts/download_and_process_itemized.py"
        )

    return pd.read_parquet(path)


def load_bonica_data() -> pd.DataFrame:
    """Load Bonica's pre-linked physician data with graduation years."""
    bonica_path = (
        PROCESSED_DATA_DIR.parent / "raw" / "bonica_replication" /
        "npi_dime_signatories_merged.csv"
    )

    if not bonica_path.exists():
        raise FileNotFoundError(f"Bonica data not found: {bonica_path}")

    df = pd.read_csv(bonica_path)

    # Clean up columns
    df['dime_cid'] = df['dime.cid'].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    df['graduation_year'] = df['gradyear']
    df['gender'] = df['female'].map({1: 'F', 0: 'M'})

    return df[['dime_cid', 'graduation_year', 'gender', 'p.to.rep']].dropna()


def merge_party_shares_with_bonica(
    party_shares: pd.DataFrame,
    bonica: pd.DataFrame,
) -> pd.DataFrame:
    """Merge our computed party shares with Bonica's physician metadata."""
    merged = bonica.merge(
        party_shares,
        left_on='dime_cid',
        right_on='bonica_cid',
        how='inner'
    )

    print(f"Matched {len(merged):,} physicians")
    print(f"Graduation year range: {merged['graduation_year'].min()}-{merged['graduation_year'].max()}")

    return merged


def aggregate_by_graduation_year_gender(
    df: pd.DataFrame,
    min_n: int = 30,
) -> pd.DataFrame:
    """Aggregate p.to.rep by graduation year and gender."""
    # By gender
    agg = df.groupby(['graduation_year', 'gender']).agg(
        mean_p_to_rep=('p_to_rep', 'mean'),
        n=('p_to_rep', 'count'),
    ).reset_index()

    # All genders
    agg_all = df.groupby('graduation_year').agg(
        mean_p_to_rep=('p_to_rep', 'mean'),
        n=('p_to_rep', 'count'),
    ).reset_index()
    agg_all['gender'] = 'All'

    result = pd.concat([agg, agg_all], ignore_index=True)
    result = result[result['n'] >= min_n].copy()

    # Convert to percentage
    result['rep_pct'] = result['mean_p_to_rep'] * 100

    return result


def plot_figure1_from_our_data(
    agg_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Contributions to Republicans by Graduation Year and Gender",
    subtitle: str = "(Computed from DIME itemized data)",
    smoothing: bool = True,
    window: int = 5,
) -> Optional[Path]:
    """Create Figure 1 plot."""
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        'All': '#333333',
        'M': '#1f77b4',
        'F': '#d62728',
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

    for gender in ['All', 'M', 'F']:
        data = agg_df[agg_df['gender'] == gender].sort_values('graduation_year')

        if len(data) == 0:
            continue

        x = data['graduation_year']
        y = data['rep_pct']

        if smoothing and len(data) >= window:
            y_smooth = data['rep_pct'].rolling(window=window, center=True, min_periods=1).mean()
            ax.plot(
                x, y_smooth,
                color=colors.get(gender, 'gray'),
                linestyle=linestyles.get(gender, '-'),
                linewidth=2.5,
                label=labels.get(gender, gender),
            )
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

    ax.set_xlabel("Year of Medical School Graduation", fontsize=14)
    ax.set_ylabel("Contributions to Republicans, %", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                fontsize=10, ha='center', style='italic')

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylim(0, 100)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None


def compare_bonica_vs_ours(
    bonica: pd.DataFrame,
    ours: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> dict:
    """Compare Bonica's p.to.rep with our computed values."""
    # Merge on donor ID
    merged = bonica.merge(
        ours[['bonica_cid', 'p_to_rep']],
        left_on='dime_cid',
        right_on='bonica_cid',
        how='inner'
    )

    if len(merged) < 100:
        return {'n_matched': len(merged), 'valid': False}

    correlation = merged['p.to.rep'].corr(merged['p_to_rep'])
    rmse = ((merged['p.to.rep'] - merged['p_to_rep']) ** 2).mean() ** 0.5

    stats = {
        'n_matched': len(merged),
        'bonica_mean': merged['p.to.rep'].mean(),
        'our_mean': merged['p_to_rep'].mean(),
        'correlation': correlation,
        'rmse': rmse,
    }

    print("\n=== Comparison: Bonica vs Our Calculation ===")
    print(f"Matched donors: {stats['n_matched']:,}")
    print(f"Bonica mean: {stats['bonica_mean']:.3f}")
    print(f"Our mean: {stats['our_mean']:.3f}")
    print(f"Correlation: {stats['correlation']:.3f}")
    print(f"RMSE: {stats['rmse']:.3f}")

    if output_path:
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(merged['p.to.rep'], merged['p_to_rep'], alpha=0.1, s=5)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect match')
        ax.set_xlabel("Bonica p.to.rep", fontsize=14)
        ax.set_ylabel("Our p.to.rep", fontsize=14)
        ax.set_title(f"Validation: r={correlation:.3f}", fontsize=16)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved validation plot: {output_path}")

    return stats


def generate_figure1_from_our_data(
    party_shares_path: Optional[Path] = None,
    period: str = "1992_2016",
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Generate Figure 1 using our computed p.to.rep values.

    Returns dict with paths to outputs.
    """
    output_dir = output_dir or PLOTS_DIR / "figure1_from_party_shares"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Figure 1 from Our Party Shares")
    print("=" * 60)

    # Load our party shares
    print("\n--- Loading party shares ---")
    party_shares = load_party_shares(party_shares_path, period)
    print(f"Loaded {len(party_shares):,} donors")

    # Load Bonica's data for graduation years
    print("\n--- Loading Bonica physician data ---")
    bonica = load_bonica_data()
    print(f"Loaded {len(bonica):,} physicians with graduation year")

    # Merge
    print("\n--- Merging ---")
    merged = merge_party_shares_with_bonica(party_shares, bonica)

    # Validate
    print("\n--- Validation ---")
    val_path = output_dir / "validation_scatter.png"
    validation = compare_bonica_vs_ours(bonica, party_shares, val_path)

    # Aggregate
    print("\n--- Aggregating by graduation year ---")
    agg = aggregate_by_graduation_year_gender(merged)

    # Generate figure
    print("\n--- Generating figure ---")
    fig_path = output_dir / "figure1_our_calculation.png"
    plot_figure1_from_our_data(
        agg,
        output_path=fig_path,
        title="Contributions to Republicans by Graduation Year",
        subtitle=f"(Our calculation from DIME itemized {period.replace('_', '-')})",
    )

    # Save aggregated data
    data_path = output_dir / "figure1_data.csv"
    agg.to_csv(data_path, index=False)
    print(f"Saved data: {data_path}")

    return {
        'figure': fig_path,
        'validation': val_path,
        'data': data_path,
        'stats': validation,
    }
