"""
Representativeness analysis for DIME physician donors.

This module provides:
1. Intensity stratification - ideology by donation amount
2. NPPES raking - reweight to physician population margins
3. Non-donor bounds - sensitivity analysis for selection bias

Key design decisions:
- Linked-only raking as PRIMARY (headline) result
- Hybrid raking as SECONDARY (sensitivity analysis)
- 5-scenario grid for non-donor bounds
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import yaml

from .config import (
    CONFIG_DIR,
    CYCLES,
    DIAGNOSTICS_DIR,
    DONOR_CYCLE_PANEL_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    NPPES_PHYSICIANS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    ensure_directories,
)

logger = logging.getLogger(__name__)

# Non-donor bounds config path
NON_DONOR_BOUNDS_CONFIG = CONFIG_DIR / "non_donor_bounds.yaml"


def compute_intensity_stratification(
    output_dir: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
    physician_only: bool = True,
) -> Path:
    """
    Compute ideology statistics stratified by donation intensity.

    Creates bins by total_amount_cycle and computes ideology stats per bin.

    Args:
        output_dir: Directory for output files
        cycles: Optional list of cycles to analyze
        physician_only: Whether to restrict to physician-classified donors

    Returns:
        Path to output CSV
    """
    output_dir = output_dir or DIAGNOSTICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not DONOR_CYCLE_PANEL_PARQUET.exists():
        raise FileNotFoundError(f"Panel not found: {DONOR_CYCLE_PANEL_PARQUET}")

    logger.info("Computing intensity stratification...")
    con = duckdb.connect()

    panel_path = str(DONOR_CYCLE_PANEL_PARQUET).replace("\\", "/")

    # Define amount bins (thresholds)
    bins = [0, 50, 200, 1000, 10000, float("inf")]
    bin_labels = ["$0-50", "$50-200", "$200-1000", "$1000-10000", "$10000+"]

    # Build query
    cycles_to_analyze = cycles or CYCLES
    cycle_list = ", ".join(str(c) for c in cycles_to_analyze)

    physician_filter = "AND physician_final = true" if physician_only else ""

    df = con.execute(f"""
        SELECT
            cycle,
            amount_total,
            cfscore_static,
            cfscore_cycle,
            revealed_party_cycle
        FROM read_parquet('{panel_path}')
        WHERE cycle IN ({cycle_list})
          AND cfscore_static IS NOT NULL
          {physician_filter}
    """).fetchdf()

    con.close()

    logger.info(f"Loaded {len(df):,} records for stratification")

    # Create amount bins
    df["amount_bin"] = pd.cut(
        df["amount_total"],
        bins=bins,
        labels=bin_labels,
        include_lowest=True,
    )

    # Compute stats per cycle and bin
    stratified = df.groupby(["cycle", "amount_bin"]).agg(
        n_donors=("cfscore_static", "count"),
        total_amount=("amount_total", "sum"),
        # CFScore stats
        cfscore_mean=("cfscore_static", "mean"),
        cfscore_median=("cfscore_static", "median"),
        cfscore_std=("cfscore_static", "std"),
        cfscore_pct_left=("cfscore_static", lambda x: (x < 0).mean()),
        cfscore_pct_right=("cfscore_static", lambda x: (x > 0).mean()),
        # Cycle-specific CFScore
        cfscore_cycle_mean=("cfscore_cycle", "mean"),
        cfscore_cycle_median=("cfscore_cycle", "median"),
        # Party-based stats
        party_mean=("revealed_party_cycle", "mean"),
        party_pct_dem=("revealed_party_cycle", lambda x: (x > 0).mean()),
        party_pct_rep=("revealed_party_cycle", lambda x: (x < 0).mean()),
    ).reset_index()

    # Compute polarization metric (distance from center)
    stratified["polarization"] = stratified["cfscore_std"]

    output_path = output_dir / "intensity_stratified_stats.csv"
    stratified.to_csv(output_path, index=False)
    logger.info(f"Created: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Intensity Stratification Summary")
    print("=" * 70)

    overall = df.groupby("amount_bin").agg(
        n_donors=("cfscore_static", "count"),
        mean_cfscore=("cfscore_static", "mean"),
    )
    print(overall.to_string())
    print("=" * 70)

    return output_path


def build_nppes_margins(
    output_path: Optional[Path] = None,
) -> Path:
    """
    Build NPPES physician population margins for raking.

    Creates counts by state x gender (and specialty_group if possible).

    Args:
        output_path: Output CSV path

    Returns:
        Path to margins file
    """
    output_path = output_path or (DIAGNOSTICS_DIR / "nppes_margins.csv")

    if not NPPES_PHYSICIANS_PARQUET.exists():
        raise FileNotFoundError(f"NPPES not found: {NPPES_PHYSICIANS_PARQUET}")

    logger.info("Building NPPES population margins...")
    con = duckdb.connect()

    nppes_path = str(NPPES_PHYSICIANS_PARQUET).replace("\\", "/")

    # Get margins by state x gender
    margins = con.execute(f"""
        SELECT
            state_norm AS state,
            gender,
            COUNT(*) AS n_physicians
        FROM read_parquet('{nppes_path}')
        WHERE state_norm IS NOT NULL AND state_norm != ''
        GROUP BY state_norm, gender
        ORDER BY state_norm, gender
    """).fetchdf()

    con.close()

    # Also compute state-only and gender-only margins
    state_margins = margins.groupby("state")["n_physicians"].sum().reset_index()
    gender_margins = margins.groupby("gender")["n_physicians"].sum().reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    margins.to_csv(output_path, index=False)
    logger.info(f"Created: {output_path}")

    # Also save marginal totals
    state_path = output_path.parent / "nppes_margins_state.csv"
    state_margins.to_csv(state_path, index=False)

    gender_path = output_path.parent / "nppes_margins_gender.csv"
    gender_margins.to_csv(gender_path, index=False)

    print(f"\nNPPES total physicians: {margins['n_physicians'].sum():,}")
    print(f"States: {margins['state'].nunique()}")

    return output_path


def rake_linked_only(
    output_path: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
    max_iterations: int = 50,
    tolerance: float = 0.001,
) -> Path:
    """
    Rake linked physician donors to NPPES margins.

    PRIMARY method - uses only high-confidence NPPES-linked donors.

    Args:
        output_path: Output CSV path
        cycles: Optional list of cycles
        max_iterations: Maximum raking iterations
        tolerance: Convergence tolerance

    Returns:
        Path to raked stats
    """
    output_path = output_path or (DIAGNOSTICS_DIR / "raked_stats_linked_only.csv")

    # Check required files
    if not LINKAGE_RESULTS_PARQUET.exists():
        raise FileNotFoundError(f"Linkage results not found: {LINKAGE_RESULTS_PARQUET}")

    if not DONOR_CYCLE_PANEL_PARQUET.exists():
        raise FileNotFoundError(f"Panel not found: {DONOR_CYCLE_PANEL_PARQUET}")

    nppes_margins_path = DIAGNOSTICS_DIR / "nppes_margins.csv"
    if not nppes_margins_path.exists():
        build_nppes_margins()

    logger.info("Computing raked statistics (linked-only)...")
    con = duckdb.connect()

    panel_path = str(DONOR_CYCLE_PANEL_PARQUET).replace("\\", "/")
    linkage_path = str(LINKAGE_RESULTS_PARQUET).replace("\\", "/")
    nppes_path = str(NPPES_PHYSICIANS_PARQUET).replace("\\", "/")

    cycles_to_analyze = cycles or CYCLES
    cycle_list = ", ".join(str(c) for c in cycles_to_analyze)

    # Load linked donors with NPPES info
    linked_df = con.execute(f"""
        SELECT
            p.bonica_cid,
            p.cycle,
            p.cfscore_static,
            p.cfscore_cycle,
            p.revealed_party_cycle,
            p.contributor_state AS state,
            l.match_score,
            n.gender,
            n.state_norm AS nppes_state
        FROM read_parquet('{panel_path}') p
        INNER JOIN read_parquet('{linkage_path}') l ON p.bonica_cid = l.bonica_cid
        INNER JOIN read_parquet('{nppes_path}') n ON l.npi = n.npi
        WHERE p.cycle IN ({cycle_list})
          AND p.physician_final = true
          AND p.cfscore_static IS NOT NULL
          AND l.match_score >= 0.85
    """).fetchdf()

    con.close()

    logger.info(f"Loaded {len(linked_df):,} linked donor-cycle records")

    if len(linked_df) == 0:
        logger.warning("No linked donors found")
        return output_path

    # Load NPPES margins
    nppes_margins = pd.read_csv(nppes_margins_path)

    # Simple raking by state
    # Use NPPES state when available, fall back to DIME state
    linked_df["rake_state"] = linked_df["nppes_state"].fillna(linked_df["state"])

    # Get observed state distribution
    observed_state = linked_df.groupby("rake_state").size()

    # Get target state distribution
    target_state = nppes_margins.groupby("state")["n_physicians"].sum()

    # Compute state adjustment weights
    states_in_common = observed_state.index.intersection(target_state.index)
    state_weights = {}

    for state in states_in_common:
        obs = observed_state.get(state, 0)
        tgt = target_state.get(state, 0)
        if obs > 0 and tgt > 0:
            state_weights[state] = tgt / obs

    # Normalize weights
    total_weight = sum(state_weights.values())
    if total_weight > 0:
        state_weights = {k: v / total_weight * len(state_weights) for k, v in state_weights.items()}

    # Apply weights
    linked_df["raking_weight"] = linked_df["rake_state"].map(state_weights).fillna(1.0)

    # Compute weighted statistics
    def weighted_mean(x, w):
        return np.average(x.dropna(), weights=w[x.dropna().index])

    def weighted_std(x, w):
        m = weighted_mean(x, w)
        return np.sqrt(np.average((x.dropna() - m) ** 2, weights=w[x.dropna().index]))

    results = []
    for cycle in cycles_to_analyze:
        cycle_data = linked_df[linked_df["cycle"] == cycle]
        if len(cycle_data) == 0:
            continue

        weights = cycle_data["raking_weight"]

        result = {
            "cycle": cycle,
            "n_donors": len(cycle_data),
            "n_donors_effective": weights.sum(),
            # Unweighted stats
            "cfscore_mean_unweighted": cycle_data["cfscore_static"].mean(),
            "cfscore_median_unweighted": cycle_data["cfscore_static"].median(),
            # Weighted stats
            "cfscore_mean_raked": np.average(
                cycle_data["cfscore_static"].dropna(),
                weights=weights[cycle_data["cfscore_static"].notna()]
            ),
            "cfscore_std_raked": weighted_std(cycle_data["cfscore_static"], weights),
            "pct_left_raked": np.average(
                cycle_data["cfscore_static"] < 0,
                weights=weights
            ),
            "pct_right_raked": np.average(
                cycle_data["cfscore_static"] > 0,
                weights=weights
            ),
            # Party-based (weighted)
            "party_mean_raked": np.average(
                cycle_data["revealed_party_cycle"].dropna(),
                weights=weights[cycle_data["revealed_party_cycle"].notna()]
            ) if cycle_data["revealed_party_cycle"].notna().any() else np.nan,
        }
        results.append(result)

    results_df = pd.DataFrame(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Created: {output_path}")

    print("\n" + "=" * 70)
    print("Raked Statistics (Linked-Only) - PRIMARY RESULT")
    print("=" * 70)
    print(f"Total linked donors: {len(linked_df):,}")
    print(f"Mean CFScore (unweighted): {linked_df['cfscore_static'].mean():.3f}")
    print(f"Mean CFScore (raked): {results_df['cfscore_mean_raked'].mean():.3f}")
    print("=" * 70)

    return output_path


def rake_hybrid_weighted(
    output_path: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
) -> Path:
    """
    Rake all physician-likely donors using p_physician weights.

    SECONDARY method - sensitivity analysis using all donors with p_physician weights.

    Args:
        output_path: Output CSV path
        cycles: Optional list of cycles

    Returns:
        Path to raked stats
    """
    output_path = output_path or (DIAGNOSTICS_DIR / "raked_stats_hybrid.csv")

    if not DONOR_CYCLE_PANEL_PARQUET.exists():
        raise FileNotFoundError(f"Panel not found: {DONOR_CYCLE_PANEL_PARQUET}")

    nppes_margins_path = DIAGNOSTICS_DIR / "nppes_margins.csv"
    if not nppes_margins_path.exists():
        build_nppes_margins()

    logger.info("Computing raked statistics (hybrid weighted)...")
    con = duckdb.connect()

    panel_path = str(DONOR_CYCLE_PANEL_PARQUET).replace("\\", "/")

    cycles_to_analyze = cycles or CYCLES
    cycle_list = ", ".join(str(c) for c in cycles_to_analyze)

    # Load all physician-likely donors
    df = con.execute(f"""
        SELECT
            bonica_cid,
            cycle,
            cfscore_static,
            cfscore_cycle,
            revealed_party_cycle,
            contributor_state AS state,
            p_physician
        FROM read_parquet('{panel_path}')
        WHERE cycle IN ({cycle_list})
          AND physician_final = true
          AND cfscore_static IS NOT NULL
    """).fetchdf()

    con.close()

    logger.info(f"Loaded {len(df):,} physician donor-cycle records")

    # Load NPPES margins
    nppes_margins = pd.read_csv(nppes_margins_path)

    # Compute state weights
    observed_state = df.groupby("state")["p_physician"].sum()
    target_state = nppes_margins.groupby("state")["n_physicians"].sum()

    states_in_common = observed_state.index.intersection(target_state.index)
    state_weights = {}

    for state in states_in_common:
        obs = observed_state.get(state, 0)
        tgt = target_state.get(state, 0)
        if obs > 0 and tgt > 0:
            state_weights[state] = tgt / obs

    # Normalize
    total_weight = sum(state_weights.values())
    if total_weight > 0:
        state_weights = {k: v / total_weight * len(state_weights) for k, v in state_weights.items()}

    # Apply weights (state weight * p_physician)
    df["state_weight"] = df["state"].map(state_weights).fillna(1.0)
    df["combined_weight"] = df["state_weight"] * df["p_physician"].fillna(1.0)

    # Compute weighted statistics per cycle
    results = []
    for cycle in cycles_to_analyze:
        cycle_data = df[df["cycle"] == cycle]
        if len(cycle_data) == 0:
            continue

        weights = cycle_data["combined_weight"]

        result = {
            "cycle": cycle,
            "n_donors": len(cycle_data),
            "sum_p_physician": cycle_data["p_physician"].sum(),
            "sum_combined_weight": weights.sum(),
            # Weighted stats
            "cfscore_mean_hybrid": np.average(
                cycle_data["cfscore_static"].dropna(),
                weights=weights[cycle_data["cfscore_static"].notna()]
            ),
            "pct_left_hybrid": np.average(
                cycle_data["cfscore_static"] < 0,
                weights=weights
            ),
            "pct_right_hybrid": np.average(
                cycle_data["cfscore_static"] > 0,
                weights=weights
            ),
        }
        results.append(result)

    results_df = pd.DataFrame(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Created: {output_path}")

    print("\n" + "=" * 70)
    print("Raked Statistics (Hybrid) - SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Total donors: {len(df):,}")
    print(f"Mean CFScore (hybrid raked): {results_df['cfscore_mean_hybrid'].mean():.3f}")
    print("=" * 70)

    return output_path


def load_non_donor_bounds_config() -> dict:
    """Load non-donor bounds configuration."""
    if not NON_DONOR_BOUNDS_CONFIG.exists():
        # Return default config
        return {
            "scenarios": {
                "S1_same_as_donors": {"name": "Same as donors", "mu_non_formula": "mu_donor"},
                "S2_neutral": {"name": "Neutral", "mu_non_formula": "0"},
                "S3_moderated": {"name": "Moderated", "mu_non_formula": "0.5 * mu_donor"},
                "S4_opposite": {"name": "Opposite", "mu_non_formula": "-mu_donor"},
                "S5_slightly_opposite": {"name": "Slightly opposite", "mu_non_formula": "-0.5 * mu_donor"},
            }
        }

    with open(NON_DONOR_BOUNDS_CONFIG, "r") as f:
        return yaml.safe_load(f)


def compute_non_donor_bounds(
    output_dir: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
) -> dict[str, Path]:
    """
    Compute non-donor selection bounds using 5-scenario sensitivity grid.

    For each scenario, computes implied population mean under different
    assumptions about non-donor ideology.

    Args:
        output_dir: Directory for output files
        cycles: Optional list of cycles

    Returns:
        Dictionary of output paths
    """
    output_dir = output_dir or DIAGNOSTICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_non_donor_bounds_config()
    scenarios = config.get("scenarios", {})

    if not DONOR_CYCLE_PANEL_PARQUET.exists():
        raise FileNotFoundError(f"Panel not found: {DONOR_CYCLE_PANEL_PARQUET}")

    logger.info("Computing non-donor selection bounds...")
    con = duckdb.connect()

    panel_path = str(DONOR_CYCLE_PANEL_PARQUET).replace("\\", "/")
    nppes_path = str(NPPES_PHYSICIANS_PARQUET).replace("\\", "/")

    cycles_to_analyze = cycles or CYCLES
    cycle_list = ", ".join(str(c) for c in cycles_to_analyze)

    # Get donor counts and ideology by cycle
    donor_stats = con.execute(f"""
        SELECT
            cycle,
            COUNT(DISTINCT bonica_cid) AS n_donors,
            AVG(cfscore_static) AS mu_donor,
            STDDEV(cfscore_static) AS sigma_donor
        FROM read_parquet('{panel_path}')
        WHERE cycle IN ({cycle_list})
          AND physician_final = true
          AND cfscore_static IS NOT NULL
        GROUP BY cycle
    """).fetchdf()

    # Get NPPES total physician count
    nppes_total = con.execute(f"""
        SELECT COUNT(*) AS n FROM read_parquet('{nppes_path}')
    """).fetchone()[0]

    con.close()

    logger.info(f"NPPES total physicians: {nppes_total:,}")

    # Compute bounds for each scenario
    results = []

    for _, row in donor_stats.iterrows():
        cycle = row["cycle"]
        n_donors = row["n_donors"]
        mu_donor = row["mu_donor"]
        sigma_donor = row["sigma_donor"]

        # Participation rate (donors / total physicians)
        p = n_donors / nppes_total

        for scenario_id, scenario_config in scenarios.items():
            scenario_name = scenario_config.get("name", scenario_id)
            formula = scenario_config.get("mu_non_formula", "0")

            # Evaluate formula for mu_non
            try:
                mu_non = eval(formula, {"mu_donor": mu_donor})
            except Exception:
                mu_non = 0.0

            # Compute implied population mean
            # mu_all = p * mu_donor + (1-p) * mu_non
            mu_all = p * mu_donor + (1 - p) * mu_non

            results.append({
                "cycle": cycle,
                "scenario": scenario_id,
                "scenario_name": scenario_name,
                "n_donors": n_donors,
                "n_population": nppes_total,
                "participation_rate": p,
                "mu_donor": mu_donor,
                "sigma_donor": sigma_donor,
                "mu_non": mu_non,
                "mu_all_implied": mu_all,
            })

    results_df = pd.DataFrame(results)

    # Save detailed results
    detail_path = output_dir / "non_donor_bounds_by_cycle.csv"
    results_df.to_csv(detail_path, index=False)
    logger.info(f"Created: {detail_path}")

    # Create summary across all cycles
    summary = results_df.groupby("scenario_name").agg(
        mean_mu_donor=("mu_donor", "mean"),
        mean_mu_non=("mu_non", "mean"),
        mean_mu_all=("mu_all_implied", "mean"),
        min_mu_all=("mu_all_implied", "min"),
        max_mu_all=("mu_all_implied", "max"),
    ).reset_index()

    summary_path = output_dir / "non_donor_bounds_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Created: {summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Non-Donor Selection Bounds Summary")
    print("=" * 70)
    print(f"Average donor mean ideology: {donor_stats['mu_donor'].mean():.3f}")
    print(f"Average participation rate: {(donor_stats['n_donors'] / nppes_total).mean():.1%}")
    print("\nImplied population means by scenario:")
    for _, s in summary.iterrows():
        print(f"  {s['scenario_name']}: {s['mean_mu_all']:.3f} ({s['min_mu_all']:.3f} to {s['max_mu_all']:.3f})")
    print("=" * 70)

    return {"detail": detail_path, "summary": summary_path}


def generate_raking_comparison(
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate comparison between linked-only and hybrid raking results.

    Args:
        output_path: Output CSV path

    Returns:
        Path to comparison file
    """
    output_path = output_path or (DIAGNOSTICS_DIR / "raking_comparison.csv")

    linked_path = DIAGNOSTICS_DIR / "raked_stats_linked_only.csv"
    hybrid_path = DIAGNOSTICS_DIR / "raked_stats_hybrid.csv"

    if not linked_path.exists() or not hybrid_path.exists():
        raise FileNotFoundError("Run rake_linked_only and rake_hybrid_weighted first")

    linked = pd.read_csv(linked_path)
    hybrid = pd.read_csv(hybrid_path)

    comparison = linked[["cycle", "n_donors", "cfscore_mean_raked"]].merge(
        hybrid[["cycle", "cfscore_mean_hybrid"]],
        on="cycle",
        how="outer",
    )

    comparison["difference"] = comparison["cfscore_mean_hybrid"] - comparison["cfscore_mean_raked"]
    comparison["pct_change"] = (
        comparison["difference"] / comparison["cfscore_mean_raked"].abs() * 100
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    logger.info(f"Created: {output_path}")

    print("\n" + "=" * 70)
    print("Raking Comparison: Linked-Only vs Hybrid")
    print("=" * 70)
    print(f"Mean difference: {comparison['difference'].mean():.4f}")
    print(f"Mean % change: {comparison['pct_change'].mean():.1f}%")
    print("=" * 70)

    return output_path


def run_all_representativeness(
    output_dir: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
) -> dict[str, Path]:
    """
    Run all representativeness analyses.

    Args:
        output_dir: Directory for outputs
        cycles: Optional list of cycles

    Returns:
        Dictionary of output paths
    """
    output_dir = output_dir or DIAGNOSTICS_DIR
    ensure_directories()

    outputs = {}

    print("=" * 70)
    print("Running Representativeness Analyses")
    print("=" * 70)

    # 1. Intensity stratification
    print("\n--- Intensity Stratification ---")
    try:
        outputs["intensity"] = compute_intensity_stratification(output_dir, cycles)
    except Exception as e:
        logger.warning(f"Skipping intensity stratification: {e}")

    # 2. Build NPPES margins
    print("\n--- NPPES Margins ---")
    try:
        outputs["nppes_margins"] = build_nppes_margins()
    except Exception as e:
        logger.warning(f"Skipping NPPES margins: {e}")

    # 3. Linked-only raking (PRIMARY)
    print("\n--- Linked-Only Raking (PRIMARY) ---")
    try:
        outputs["raking_linked"] = rake_linked_only(cycles=cycles)
    except Exception as e:
        logger.warning(f"Skipping linked-only raking: {e}")

    # 4. Hybrid raking (SENSITIVITY)
    print("\n--- Hybrid Raking (SENSITIVITY) ---")
    try:
        outputs["raking_hybrid"] = rake_hybrid_weighted(cycles=cycles)
    except Exception as e:
        logger.warning(f"Skipping hybrid raking: {e}")

    # 5. Raking comparison
    print("\n--- Raking Comparison ---")
    try:
        outputs["raking_comparison"] = generate_raking_comparison()
    except Exception as e:
        logger.warning(f"Skipping raking comparison: {e}")

    # 6. Non-donor bounds
    print("\n--- Non-Donor Bounds ---")
    try:
        bounds_outputs = compute_non_donor_bounds(output_dir, cycles)
        outputs.update({f"bounds_{k}": v for k, v in bounds_outputs.items()})
    except Exception as e:
        logger.warning(f"Skipping non-donor bounds: {e}")

    print("\n" + "=" * 70)
    print("Representativeness analyses complete!")
    print(f"Output files: {len(outputs)}")
    print("=" * 70)

    return outputs
