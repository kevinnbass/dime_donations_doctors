"""
Visualization module for physician ideology analysis.

Creates KDE plots, time series, and summary figures.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    CYCLES,
    DONOR_CYCLE_PANEL_PARQUET,
    PLOTS_DIR,
    TABLES_DIR,
)
from .ideology import get_physician_cycle_data


# Plot style configuration
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 10,
}

# Color for physician plots (matching original)
PHYSICIAN_COLOR = "blue"


def setup_plot_style():
    """Apply consistent plot styling."""
    plt.rcParams.update(PLOT_STYLE)
    sns.set_style("whitegrid")


def plot_ideology_distribution(
    data: pd.Series,
    year: int,
    occupation: str = "Physicians",
    color: str = PHYSICIAN_COLOR,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Plot a KDE of ideology distribution for a single year.

    Matches the style of the original script.

    Args:
        data: Series of CFScore values
        year: Election cycle year
        occupation: Occupation name for title
        color: Fill color for KDE
        output_path: Optional output file path
        show: Whether to display the plot

    Returns:
        Output path if saved, None otherwise
    """
    setup_plot_style()

    n = len(data)
    if n < 2 or data.std() == 0:
        print(f"Insufficient data for {occupation} in {year}: n={n}")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create KDE plot
    try:
        sns.kdeplot(
            data,
            bw_adjust=0.3,
            color=color,
            fill=True,
            alpha=1,
            ax=ax,
        )
    except Exception as e:
        print(f"Error creating KDE for {year}: {e}")
        plt.close(fig)
        return None

    # Calculate percentages
    positive_pct = round((np.sum(data > 0) + np.sum(data == 0) / 2) / n * 100)
    negative_pct = round((np.sum(data < 0) + np.sum(data == 0) / 2) / n * 100)

    # Style the plot
    ax.set_title(f"Political ideology of American {occupation}, {year} (n={n:,})", fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xticks([-1.2, -0.6, 0, 0.6, 1.2])
    ax.set_xticklabels([
        "Very left",
        f"Left\n{negative_pct}%",
        "Moderate",
        f"Right\n{positive_pct}%",
        "Very right",
    ], fontsize=12)
    ax.set_xlim(-2, 2)

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    if show:
        plt.show()

    plt.close(fig)
    return None


def plot_time_series(
    summary_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = False,
    title: str = "Physician Political Ideology Over Time",
) -> Optional[Path]:
    """
    Plot time series of median ideology with IQR bands.

    Args:
        summary_df: DataFrame with columns: cycle, median, p25, p75
        output_path: Optional output file path
        show: Whether to display the plot
        title: Plot title

    Returns:
        Output path if saved, None otherwise
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    cycles = summary_df["cycle"]
    median = summary_df["median"]
    p25 = summary_df["p25"]
    p75 = summary_df["p75"]

    # Plot IQR band
    ax.fill_between(cycles, p25, p75, alpha=0.3, color=PHYSICIAN_COLOR, label="IQR")

    # Plot median line
    ax.plot(cycles, median, color=PHYSICIAN_COLOR, linewidth=2, marker="o", label="Median")

    # Add reference line at 0
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Election Cycle", fontsize=12)
    ax.set_ylabel("CFScore (Ideology)", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="best")

    # Set x-axis ticks
    ax.set_xticks(cycles[::2])  # Every other cycle
    ax.set_xticklabels([str(c) for c in cycles[::2]], rotation=45)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    if show:
        plt.show()

    plt.close(fig)
    return None


def plot_comparison(
    data_dict: dict[str, pd.DataFrame],
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Plot comparison of different physician definitions.

    Args:
        data_dict: Dict mapping definition name to summary DataFrame
        output_path: Optional output file path
        show: Whether to display

    Returns:
        Output path if saved
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["blue", "green", "orange", "red"]

    for (name, df), color in zip(data_dict.items(), colors):
        ax.plot(
            df["cycle"],
            df["median"],
            label=name,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Election Cycle", fontsize=12)
    ax.set_ylabel("Median CFScore", fontsize=12)
    ax.set_title("Physician Ideology by Definition", fontsize=16)
    ax.legend(loc="best")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    if show:
        plt.show()

    plt.close(fig)
    return None


def generate_cycle_plots(
    physician_definition: str = "final",
    ideology_measure: str = "static",
    output_dir: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
) -> list[Path]:
    """
    Generate KDE plots for all cycles.

    Args:
        physician_definition: 'naive', 'rule', or 'final'
        ideology_measure: 'static' or 'cycle'
        output_dir: Output directory for plots
        cycles: Optional list of cycles to plot

    Returns:
        List of created file paths
    """
    output_dir = output_dir or PLOTS_DIR / physician_definition

    cycles_to_plot = cycles or CYCLES

    # Load data
    df = get_physician_cycle_data(physician_definition, cycles_to_plot)

    if ideology_measure == "static":
        score_col = "cfscore_static"
    else:
        score_col = "cfscore_cycle"

    created_files = []

    for year in cycles_to_plot:
        year_data = df[df["cycle"] == year][score_col].dropna()

        if len(year_data) < 10:
            print(f"Skipping {year}: insufficient data ({len(year_data)})")
            continue

        output_path = output_dir / f"physicians_smoothed_histogram_{year}.png"

        result = plot_ideology_distribution(
            year_data,
            year,
            occupation="Physicians",
            output_path=output_path,
        )

        if result:
            created_files.append(result)
            print(f"Created: {result}")

    return created_files


def generate_summary_table(
    physician_definition: str = "final",
    ideology_measure: str = "static",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate summary statistics table by cycle.

    Args:
        physician_definition: 'naive', 'rule', or 'final'
        ideology_measure: 'static' or 'cycle'
        output_path: Optional CSV output path

    Returns:
        Summary DataFrame
    """
    df = get_physician_cycle_data(physician_definition)

    if ideology_measure == "static":
        score_col = "cfscore_static"
    else:
        score_col = "cfscore_cycle"

    # Group by cycle
    summary = df.groupby("cycle")[score_col].agg([
        ("n", "count"),
        ("mean", "mean"),
        ("median", "median"),
        ("std", "std"),
        ("p25", lambda x: x.quantile(0.25)),
        ("p75", lambda x: x.quantile(0.75)),
    ]).reset_index()

    # Add share left/right
    def calc_shares(group):
        total = len(group)
        return pd.Series({
            "share_left": (group < 0).sum() / total,
            "share_right": (group > 0).sum() / total,
        })

    shares = df.groupby("cycle")[score_col].apply(calc_shares).unstack().reset_index()
    summary = summary.merge(shares, on="cycle")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"Created: {output_path}")

    return summary


def generate_all_outputs(
    physician_definitions: list[str] = None,
    overwrite: bool = False,
) -> dict:
    """
    Generate all plots and tables for the analysis.

    Args:
        physician_definitions: List of definitions to generate for
        overwrite: Whether to overwrite existing files

    Returns:
        Dictionary with paths to all created files
    """
    physician_definitions = physician_definitions or ["naive", "rule", "final"]

    results = {
        "plots": [],
        "tables": [],
        "time_series": [],
    }

    for definition in physician_definitions:
        print(f"\nGenerating outputs for '{definition}' definition...")

        # Cycle plots
        plots = generate_cycle_plots(
            physician_definition=definition,
            output_dir=PLOTS_DIR / definition,
        )
        results["plots"].extend(plots)

        # Summary table
        table_path = TABLES_DIR / f"summary_{definition}.csv"
        summary = generate_summary_table(
            physician_definition=definition,
            output_path=table_path,
        )
        results["tables"].append(table_path)

        # Time series plot
        ts_path = PLOTS_DIR / f"time_series_{definition}.png"
        plot_time_series(
            summary,
            output_path=ts_path,
            title=f"Physician Political Ideology Over Time ({definition})",
        )
        results["time_series"].append(ts_path)

    # Comparison plot
    print("\nGenerating comparison plot...")
    comparison_data = {}
    for definition in physician_definitions:
        comparison_data[definition] = generate_summary_table(definition)

    comparison_path = PLOTS_DIR / "comparison_by_definition.png"
    plot_comparison(comparison_data, output_path=comparison_path)
    results["comparison"] = comparison_path

    return results
