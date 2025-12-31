#!/usr/bin/env python3
"""
Replicate Physician Political Contributions Figures

This script generates figures showing how physician political donations have
shifted from majority-Republican in the 1980s to majority-Democratic today.

The analysis uses data from the DIME (Database on Ideology, Money in Politics,
and Elections) maintained by Adam Bonica at Stanford University.

Usage:
    python replicate_figure.py              # Generate all figures
    python replicate_figure.py --help       # Show options

Output:
    figures/physician_contributions_by_cycle.png
    figures/physician_contributions_no_specialists.png

For methodology details, see METHODOLOGY.md
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server/CI environments


# =============================================================================
# Configuration
# =============================================================================

# Pool definitions: how we identify physicians in the data
POOL_DEFINITIONS = {
    "doctor_physician": {
        "name": "Doctor/Physician Keyword",
        "color": "#1f77b4",  # Blue
        "description": "Occupation contains 'doctor' or 'physician'"
    },
    "md_do_credential": {
        "name": "MD/DO Credential",
        "color": "#2ca02c",  # Green
        "description": "Occupation includes MD or DO credentials"
    },
    "specialists": {
        "name": "Medical Specialists",
        "color": "#d62728",  # Red
        "description": "Occupation indicates a medical specialty (surgeon, cardiologist, etc.)"
    },
    "cms_medicare": {
        "name": "CMS Medicare (thru 2018)",
        "color": "#9467bd",  # Purple
        "description": "Matched to CMS Medicare billing physician records"
    },
    "tier1_fixed": {
        "name": "Broad Physician Rules",
        "color": "#ff7f0e",  # Orange
        "description": "Comprehensive patterns including credentials and specialties"
    },
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_path: Path = None) -> pd.DataFrame:
    """Load the pre-computed contribution data.

    Args:
        data_path: Path to CSV file. Defaults to data/yearly_contributions_by_pool.csv

    Returns:
        DataFrame with columns: cycle, n_donations, total_dollars, rep_share,
        rep_dollars, dem_dollars, rep_donations, dem_donations, pool, pool_name
    """
    if data_path is None:
        data_path = Path(__file__).parent / "data" / "yearly_contributions_by_pool.csv"

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Please ensure the data file exists in the data/ directory.")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # Validate expected columns
    required_cols = ["cycle", "n_donations", "rep_share", "pool"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)

    return df


# =============================================================================
# Plotting Functions
# =============================================================================

def format_sample_size(n: int) -> str:
    """Format a number with K/M suffix for readability.

    Examples:
        1234 -> "1K"
        1234567 -> "1.2M"
    """
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    else:
        return str(n)


def plot_contributions(
    data: pd.DataFrame,
    pools: dict,
    output_path: Path,
    exclude_pools: list = None,
    title: str = None,
    subtitle: str = None,
    figsize: tuple = (14, 8),
    dpi: int = 150,
) -> None:
    """Create a plot of physician contributions by election cycle.

    Args:
        data: DataFrame with contribution data
        pools: Dict of pool definitions with 'name' and 'color' keys
        output_path: Where to save the figure
        exclude_pools: List of pool IDs to exclude from the plot
        title: Main title (default: "Physician Political Contributions by Election Cycle")
        subtitle: Subtitle shown below main title
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saved figure
    """
    if exclude_pools is None:
        exclude_pools = []

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each pool
    for pool_id, pool_def in pools.items():
        if pool_id in exclude_pools:
            continue

        pool_data = data[data["pool"] == pool_id].copy()
        if len(pool_data) == 0:
            continue

        pool_data = pool_data.sort_values("cycle")
        total_n = int(pool_data["n_donations"].sum())
        n_str = format_sample_size(total_n)

        ax.plot(
            pool_data["cycle"],
            pool_data["rep_share"] * 100,
            color=pool_def["color"],
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{pool_def['name']} (n={n_str})",
            alpha=0.85
        )

    # Add 50% reference line (partisan parity)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1,
               label="50% (partisan parity)")

    # Labels and formatting
    ax.set_xlabel("Election Cycle", fontsize=12)
    ax.set_ylabel("% Contributions to Republicans", fontsize=12)

    # Title
    if title is None:
        title = "Physician Political Contributions by Election Cycle"
    if subtitle:
        title = f"{title}\n{subtitle}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Axis limits
    ax.set_xlim(1978, 2026)
    ax.set_ylim(10, 80)

    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Tick formatting
    ax.set_xticks(range(1980, 2025, 4))
    ax.set_yticks(range(10, 81, 10))

    # Add annotation for key insight
    ax.annotate(
        "Majority Republican",
        xy=(1985, 52), fontsize=9, color="gray", alpha=0.7
    )
    ax.annotate(
        "Majority Democratic",
        xy=(1985, 48), fontsize=9, color="gray", alpha=0.7,
        verticalalignment="top"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")


def plot_summary_stats(data: pd.DataFrame, output_path: Path) -> None:
    """Create a summary table image showing key statistics."""
    # This could be extended to create additional visualizations
    pass


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Generate all replication figures."""
    parser = argparse.ArgumentParser(
        description="Replicate physician political contributions figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python replicate_figure.py                    # Generate all figures
    python replicate_figure.py --output-dir ./my_figures
    python replicate_figure.py --no-specialists   # Only generate the version without specialists
        """
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Path to CSV data file (default: data/yearly_contributions_by_pool.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output figures (default: figures/)"
    )
    parser.add_argument(
        "--no-specialists",
        action="store_true",
        help="Only generate the figure without medical specialists"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure resolution (default: 150)"
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("PHYSICIAN POLITICAL CONTRIBUTIONS - FIGURE REPLICATION")
    print("=" * 60)

    # Load data
    data = load_data(args.data_file)

    print(f"\nData loaded:")
    print(f"  - {len(data):,} data points")
    print(f"  - Years: {data['cycle'].min()} to {data['cycle'].max()}")
    print(f"  - Pools: {data['pool'].nunique()}")

    # Summary statistics
    print("\nSummary by pool:")
    for pool_id, pool_def in POOL_DEFINITIONS.items():
        pool_data = data[data["pool"] == pool_id]
        if len(pool_data) > 0:
            total_n = pool_data["n_donations"].sum()
            mean_rep = (pool_data["rep_share"] * pool_data["n_donations"]).sum() / total_n
            print(f"  - {pool_def['name']}: {total_n:,} donations, {mean_rep*100:.1f}% Republican")

    # Create output directory
    output_dir = args.output_dir or (Path(__file__).parent / "figures")
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating figures in: {output_dir}")

    # Generate figures
    if not args.no_specialists:
        # All pools
        plot_contributions(
            data,
            POOL_DEFINITIONS,
            output_dir / "physician_contributions_by_cycle.png",
            subtitle="Data from DIME database (1980-2024)",
            dpi=args.dpi,
        )

    # Without specialists
    plot_contributions(
        data,
        POOL_DEFINITIONS,
        output_dir / "physician_contributions_no_specialists.png",
        exclude_pools=["specialists"],
        subtitle="Excluding Medical Specialists pool",
        dpi=args.dpi,
    )

    # Print completion message
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {output_dir.absolute()}")
    print("\nFor methodology details, see METHODOLOGY.md")


if __name__ == "__main__":
    main()
