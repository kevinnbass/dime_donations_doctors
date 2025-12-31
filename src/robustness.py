"""
Robustness checks and sensitivity analyses.

Provides bootstrap confidence intervals, reweighting, and
comparison across different physician definitions.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    CYCLES,
    DIAGNOSTICS_DIR,
    DONOR_CYCLE_PANEL_PARQUET,
    NPPES_PHYSICIANS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    DIME_DONORS_PARQUET,
    AHRF_STATE_PARQUET,
    CMS_MEDICARE_PARQUET,
    TABLES_DIR,
)
from .ideology import get_physician_cycle_data


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.median,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Array of values
        statistic: Function to compute statistic (default: median)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed

    Returns:
        Tuple of (estimate, lower_ci, upper_ci)
    """
    rng = np.random.RandomState(random_state)

    n = len(data)
    if n == 0:
        return (np.nan, np.nan, np.nan)

    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Point estimate
    estimate = statistic(data)

    # Percentile CI
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return (estimate, lower, upper)


def compute_cycle_statistics_with_ci(
    df: pd.DataFrame,
    score_col: str = "cfscore_static",
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    Compute summary statistics with bootstrap CIs for each cycle.

    Args:
        df: DataFrame with cycle and score columns
        score_col: Column name for ideology score
        n_bootstrap: Number of bootstrap samples

    Returns:
        DataFrame with cycle statistics and CIs
    """
    results = []

    for cycle in sorted(df["cycle"].unique()):
        cycle_data = df[df["cycle"] == cycle][score_col].dropna().values

        if len(cycle_data) < 10:
            continue

        # Mean with CI
        mean_est, mean_lower, mean_upper = bootstrap_ci(
            cycle_data, np.mean, n_bootstrap
        )

        # Median with CI
        median_est, median_lower, median_upper = bootstrap_ci(
            cycle_data, np.median, n_bootstrap
        )

        # Share left with CI
        share_left_est, sl_lower, sl_upper = bootstrap_ci(
            cycle_data, lambda x: np.mean(x < 0), n_bootstrap
        )

        # Share right with CI
        share_right_est, sr_lower, sr_upper = bootstrap_ci(
            cycle_data, lambda x: np.mean(x > 0), n_bootstrap
        )

        results.append({
            "cycle": cycle,
            "n": len(cycle_data),
            "mean": mean_est,
            "mean_ci_lower": mean_lower,
            "mean_ci_upper": mean_upper,
            "median": median_est,
            "median_ci_lower": median_lower,
            "median_ci_upper": median_upper,
            "share_left": share_left_est,
            "share_left_ci_lower": sl_lower,
            "share_left_ci_upper": sl_upper,
            "share_right": share_right_est,
            "share_right_ci_lower": sr_lower,
            "share_right_ci_upper": sr_upper,
        })

    return pd.DataFrame(results)


def compare_definitions(
    definitions: list[str] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare statistics across different physician definitions.

    Args:
        definitions: List of definition names to compare
        output_path: Optional CSV output path

    Returns:
        DataFrame comparing definitions
    """
    definitions = definitions or ["naive", "rule", "final"]

    all_results = []

    for definition in definitions:
        try:
            df = get_physician_cycle_data(definition)
            stats_df = compute_cycle_statistics_with_ci(df)
            stats_df["definition"] = definition
            all_results.append(stats_df)
        except Exception as e:
            print(f"Error computing stats for {definition}: {e}")

    if not all_results:
        return pd.DataFrame()

    result = pd.concat(all_results, ignore_index=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Created: {output_path}")

    return result


def compute_definition_agreement(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute agreement between different physician definitions.

    Shows overlap and differences between naive, rule-based, and NPPES-linked.

    Returns:
        DataFrame with agreement statistics
    """
    import duckdb

    if not DONOR_CYCLE_PANEL_PARQUET.exists():
        return pd.DataFrame()

    con = duckdb.connect()
    path = str(DONOR_CYCLE_PANEL_PARQUET).replace("\\", "/")

    # Get unique donors with their labels
    donors = con.execute(f"""
        SELECT DISTINCT
            bonica_cid,
            COALESCE(physician_naive, false) as naive,
            COALESCE(physician_rule_label, false) as rule_based,
            COALESCE(physician_final, false) as final
        FROM read_parquet('{path}')
    """).fetchdf()

    con.close()

    # Compute agreement
    n_total = len(donors)
    n_naive = donors["naive"].sum()
    n_rule = donors["rule_based"].sum()
    n_final = donors["final"].sum()

    # Overlap between definitions
    n_naive_and_rule = ((donors["naive"]) & (donors["rule_based"])).sum()
    n_naive_and_final = ((donors["naive"]) & (donors["final"])).sum()
    n_rule_and_final = ((donors["rule_based"]) & (donors["final"])).sum()
    n_all_three = ((donors["naive"]) & (donors["rule_based"]) & (donors["final"])).sum()

    # Exclusive to each definition
    n_naive_only = ((donors["naive"]) & ~(donors["rule_based"]) & ~(donors["final"])).sum()
    n_rule_only = (~(donors["naive"]) & (donors["rule_based"]) & ~(donors["final"])).sum()
    n_final_only = (~(donors["naive"]) & ~(donors["rule_based"]) & (donors["final"])).sum()

    results = pd.DataFrame([
        {"metric": "Total donors", "value": n_total},
        {"metric": "Naive (doctor|physician)", "value": n_naive},
        {"metric": "Rule-based", "value": n_rule},
        {"metric": "Final (combined)", "value": n_final},
        {"metric": "Naive AND Rule-based", "value": n_naive_and_rule},
        {"metric": "Naive AND Final", "value": n_naive_and_final},
        {"metric": "Rule-based AND Final", "value": n_rule_and_final},
        {"metric": "All three definitions", "value": n_all_three},
        {"metric": "Naive only (false positives?)", "value": n_naive_only},
        {"metric": "Rule-based only", "value": n_rule_only},
        {"metric": "Final only (NPPES-added)", "value": n_final_only},
    ])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Created: {output_path}")

    return results


def compute_probability_weighted_statistics(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute statistics using p_physician as weights.

    This provides a continuous measure rather than hard thresholds.

    Returns:
        DataFrame with weighted statistics by cycle
    """
    import duckdb

    if not DONOR_CYCLE_PANEL_PARQUET.exists():
        return pd.DataFrame()

    con = duckdb.connect()
    path = str(DONOR_CYCLE_PANEL_PARQUET).replace("\\", "/")

    # Load data
    df = con.execute(f"""
        SELECT
            cycle,
            cfscore_static,
            p_physician
        FROM read_parquet('{path}')
        WHERE cfscore_static IS NOT NULL
          AND p_physician IS NOT NULL
          AND p_physician > 0
    """).fetchdf()

    con.close()

    results = []
    for cycle in sorted(df["cycle"].unique()):
        cycle_df = df[df["cycle"] == cycle]

        weights = cycle_df["p_physician"].values
        scores = cycle_df["cfscore_static"].values

        # Weighted mean
        weighted_mean = np.average(scores, weights=weights)

        # Effective sample size
        eff_n = weights.sum() ** 2 / (weights ** 2).sum()

        # Weighted percentiles (approximate)
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cumsum = cumsum / cumsum[-1]

        weighted_median = sorted_scores[np.searchsorted(cumsum, 0.5)]
        weighted_p25 = sorted_scores[np.searchsorted(cumsum, 0.25)]
        weighted_p75 = sorted_scores[np.searchsorted(cumsum, 0.75)]

        # Weighted share left/right
        left_mask = scores < 0
        right_mask = scores > 0
        weighted_share_left = np.sum(weights[left_mask]) / np.sum(weights)
        weighted_share_right = np.sum(weights[right_mask]) / np.sum(weights)

        results.append({
            "cycle": cycle,
            "effective_n": eff_n,
            "raw_n": len(cycle_df),
            "weighted_mean": weighted_mean,
            "weighted_median": weighted_median,
            "weighted_p25": weighted_p25,
            "weighted_p75": weighted_p75,
            "weighted_share_left": weighted_share_left,
            "weighted_share_right": weighted_share_right,
        })

    result = pd.DataFrame(results)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Created: {output_path}")

    return result


def run_all_robustness_checks(
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run all robustness checks and save outputs.

    Returns:
        Dictionary with paths to all output files
    """
    output_dir = output_dir or DIAGNOSTICS_DIR

    results = {}

    print("Running robustness checks...")

    # 1. Compare definitions
    print("  Comparing physician definitions...")
    comparison_path = output_dir / "definition_comparison.csv"
    compare_definitions(output_path=comparison_path)
    results["definition_comparison"] = comparison_path

    # 2. Definition agreement
    print("  Computing definition agreement...")
    agreement_path = output_dir / "definition_agreement.csv"
    compute_definition_agreement(output_path=agreement_path)
    results["definition_agreement"] = agreement_path

    # 3. Probability-weighted statistics
    print("  Computing probability-weighted statistics...")
    weighted_path = output_dir / "probability_weighted_stats.csv"
    compute_probability_weighted_statistics(output_path=weighted_path)
    results["probability_weighted"] = weighted_path

    # 4. Bootstrap CIs for each definition
    for definition in ["naive", "rule", "final"]:
        print(f"  Computing bootstrap CIs for '{definition}'...")
        try:
            df = get_physician_cycle_data(definition)
            stats_df = compute_cycle_statistics_with_ci(df, n_bootstrap=1000)
            ci_path = output_dir / f"bootstrap_ci_{definition}.csv"
            stats_df.to_csv(ci_path, index=False)
            results[f"bootstrap_ci_{definition}"] = ci_path
        except Exception as e:
            print(f"    Error: {e}")

    print("\nRobustness checks complete.")
    return results


def generate_diagnostics_report(
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a text report of classification diagnostics.

    Returns:
        Report text
    """
    from .physician_classifier import get_physician_label_stats
    from .ideology import get_panel_stats

    label_stats = get_physician_label_stats()
    panel_stats = get_panel_stats()

    report_lines = [
        "=" * 60,
        "PHYSICIAN CLASSIFICATION DIAGNOSTICS REPORT",
        "=" * 60,
        "",
        "PHYSICIAN LABELS SUMMARY",
        "-" * 40,
        f"Total donors: {label_stats.get('n_total', 'N/A'):,}",
        f"Naive keyword matches: {label_stats.get('n_naive', 'N/A'):,}",
        f"Rule-based physicians: {label_stats.get('n_rule', 'N/A'):,}",
        f"NPPES-linked: {label_stats.get('n_nppes', 'N/A'):,}",
        f"Final physicians: {label_stats.get('n_final', 'N/A'):,}",
        "",
        "DONOR-CYCLE PANEL SUMMARY",
        "-" * 40,
        f"Total records: {panel_stats.get('n_records', 'N/A'):,}",
        f"Unique donors: {panel_stats.get('n_donors', 'N/A'):,}",
        f"Physician records: {panel_stats.get('n_physician_records', 'N/A'):,}",
        f"Records with cycle ideology: {panel_stats.get('n_with_cycle_ideology', 'N/A'):,}",
        "",
        "=" * 60,
    ]

    report = "\n".join(report_lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Created: {output_path}")

    return report


def compare_medicare_active_vs_all(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare ideology distributions: all physicians vs Medicare-active.

    This is a key selection bias robustness check:
    - Are Medicare-active physicians ideologically different?
    - If not, suggests donation behavior is not driven by Medicare policy exposure

    Returns:
        DataFrame with comparison statistics
    """
    import duckdb

    if not PHYSICIAN_LABELS_PARQUET.exists():
        raise FileNotFoundError("Physician labels not found")
    if not DIME_DONORS_PARQUET.exists():
        raise FileNotFoundError("DIME donors not found")

    con = duckdb.connect()

    labels_p = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
    donors_p = str(DIME_DONORS_PARQUET).replace("\\", "/")

    # Check if CMS columns exist
    cols = con.execute(f"""
        SELECT column_name
        FROM (DESCRIBE SELECT * FROM read_parquet('{labels_p}') LIMIT 1)
    """).fetchdf()["column_name"].tolist()

    if "cms_medicare_active" not in cols:
        con.close()
        print("CMS columns not found in physician labels. Run CMS integration first.")
        return pd.DataFrame()

    # All physicians
    all_stats = con.execute(f"""
        SELECT
            'all_physicians' as group_name,
            COUNT(*) as n,
            AVG(d.contributor_cfscore) as mean_cfscore,
            APPROX_QUANTILE(d.contributor_cfscore, 0.5) as median_cfscore,
            STDDEV(d.contributor_cfscore) as std_cfscore,
            AVG(CASE WHEN d.contributor_cfscore < 0 THEN 1.0 ELSE 0.0 END) as share_left,
            AVG(CASE WHEN d.contributor_cfscore > 0 THEN 1.0 ELSE 0.0 END) as share_right
        FROM read_parquet('{donors_p}') d
        JOIN read_parquet('{labels_p}') l ON d.bonica_cid = l.bonica_cid
        WHERE l.physician_final = true
          AND d.contributor_cfscore IS NOT NULL
    """).fetchdf()

    # Medicare-active physicians
    medicare_stats = con.execute(f"""
        SELECT
            'medicare_active' as group_name,
            COUNT(*) as n,
            AVG(d.contributor_cfscore) as mean_cfscore,
            APPROX_QUANTILE(d.contributor_cfscore, 0.5) as median_cfscore,
            STDDEV(d.contributor_cfscore) as std_cfscore,
            AVG(CASE WHEN d.contributor_cfscore < 0 THEN 1.0 ELSE 0.0 END) as share_left,
            AVG(CASE WHEN d.contributor_cfscore > 0 THEN 1.0 ELSE 0.0 END) as share_right
        FROM read_parquet('{donors_p}') d
        JOIN read_parquet('{labels_p}') l ON d.bonica_cid = l.bonica_cid
        WHERE l.physician_final = true
          AND l.cms_medicare_active = true
          AND d.contributor_cfscore IS NOT NULL
    """).fetchdf()

    # Non-Medicare physicians (for contrast)
    non_medicare_stats = con.execute(f"""
        SELECT
            'non_medicare' as group_name,
            COUNT(*) as n,
            AVG(d.contributor_cfscore) as mean_cfscore,
            APPROX_QUANTILE(d.contributor_cfscore, 0.5) as median_cfscore,
            STDDEV(d.contributor_cfscore) as std_cfscore,
            AVG(CASE WHEN d.contributor_cfscore < 0 THEN 1.0 ELSE 0.0 END) as share_left,
            AVG(CASE WHEN d.contributor_cfscore > 0 THEN 1.0 ELSE 0.0 END) as share_right
        FROM read_parquet('{donors_p}') d
        JOIN read_parquet('{labels_p}') l ON d.bonica_cid = l.bonica_cid
        WHERE l.physician_final = true
          AND (l.cms_medicare_active = false OR l.cms_medicare_active IS NULL)
          AND d.contributor_cfscore IS NOT NULL
    """).fetchdf()

    con.close()

    result = pd.concat([all_stats, medicare_stats, non_medicare_stats], ignore_index=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Created: {output_path}")

    return result


def compute_cms_specialty_statistics(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute ideology statistics by CMS Medicare specialty.

    Allows comparison of specialty effects using CMS classification
    vs NPPES taxonomy classification.

    Returns:
        DataFrame with ideology stats by CMS specialty
    """
    import duckdb

    if not PHYSICIAN_LABELS_PARQUET.exists():
        raise FileNotFoundError("Physician labels not found")
    if not DIME_DONORS_PARQUET.exists():
        raise FileNotFoundError("DIME donors not found")

    con = duckdb.connect()

    labels_p = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
    donors_p = str(DIME_DONORS_PARQUET).replace("\\", "/")

    # Check if CMS columns exist
    cols = con.execute(f"""
        SELECT column_name
        FROM (DESCRIBE SELECT * FROM read_parquet('{labels_p}') LIMIT 1)
    """).fetchdf()["column_name"].tolist()

    if "cms_specialty" not in cols:
        con.close()
        print("CMS columns not found in physician labels. Run CMS integration first.")
        return pd.DataFrame()

    result = con.execute(f"""
        SELECT
            l.cms_specialty,
            COUNT(*) as n,
            AVG(d.contributor_cfscore) as mean_cfscore,
            APPROX_QUANTILE(d.contributor_cfscore, 0.5) as median_cfscore,
            STDDEV(d.contributor_cfscore) as std_cfscore,
            AVG(CASE WHEN d.contributor_cfscore < 0 THEN 1.0 ELSE 0.0 END) as share_left,
            AVG(CASE WHEN d.contributor_cfscore > 0 THEN 1.0 ELSE 0.0 END) as share_right,
            AVG(l.cms_total_beneficiaries) as avg_beneficiaries,
            AVG(l.cms_total_payment) as avg_payment
        FROM read_parquet('{donors_p}') d
        JOIN read_parquet('{labels_p}') l ON d.bonica_cid = l.bonica_cid
        WHERE l.physician_final = true
          AND l.cms_specialty IS NOT NULL
          AND d.contributor_cfscore IS NOT NULL
        GROUP BY l.cms_specialty
        HAVING COUNT(*) >= 50
        ORDER BY n DESC
    """).fetchdf()

    con.close()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Created: {output_path}")

    return result


def compute_population_weighted_statistics(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute ideology statistics with geographic population weighting.

    Uses AHRF state-level physician populations to reweight donors
    so that each state contributes proportionally to its physician population.

    Returns:
        DataFrame with weighted and unweighted statistics
    """
    import duckdb

    if not AHRF_STATE_PARQUET.exists():
        print("AHRF state data not found. Run AHRF integration first.")
        return pd.DataFrame()

    if not PHYSICIAN_LABELS_PARQUET.exists():
        raise FileNotFoundError("Physician labels not found")
    if not DIME_DONORS_PARQUET.exists():
        raise FileNotFoundError("DIME donors not found")

    con = duckdb.connect()

    ahrf_p = str(AHRF_STATE_PARQUET).replace("\\", "/")
    labels_p = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
    donors_p = str(DIME_DONORS_PARQUET).replace("\\", "/")

    # Get state-level donor counts and population
    state_data = con.execute(f"""
        WITH donor_counts AS (
            SELECT
                d.contributor_state as state,
                COUNT(*) as n_donors,
                AVG(d.contributor_cfscore) as mean_cfscore
            FROM read_parquet('{donors_p}') d
            JOIN read_parquet('{labels_p}') l ON d.bonica_cid = l.bonica_cid
            WHERE l.physician_final = true
              AND d.contributor_cfscore IS NOT NULL
              AND d.contributor_state IS NOT NULL
              AND LENGTH(d.contributor_state) = 2
            GROUP BY d.contributor_state
        ),
        pop_data AS (
            SELECT state, total_active_physicians
            FROM read_parquet('{ahrf_p}')
        )
        SELECT
            dc.state,
            dc.n_donors,
            dc.mean_cfscore,
            pd.total_active_physicians as population,
            CAST(pd.total_active_physicians AS DOUBLE) /
                (SELECT SUM(total_active_physicians) FROM pop_data) as pop_weight
        FROM donor_counts dc
        JOIN pop_data pd ON dc.state = pd.state
    """).fetchdf()

    # Compute weighted vs unweighted
    # Unweighted: simple average across donors
    unweighted_mean = state_data["mean_cfscore"].mean()

    # Population weighted: weight each state by its physician population share
    state_data["weighted_contrib"] = state_data["mean_cfscore"] * state_data["pop_weight"]
    weighted_mean = state_data["weighted_contrib"].sum() / state_data["pop_weight"].sum()

    # Donor-weighted (status quo)
    state_data["donor_weight"] = state_data["n_donors"] / state_data["n_donors"].sum()
    donor_weighted_mean = (state_data["mean_cfscore"] * state_data["donor_weight"]).sum()

    con.close()

    summary = pd.DataFrame([
        {"weighting": "unweighted_state_avg", "mean_cfscore": unweighted_mean},
        {"weighting": "population_weighted", "mean_cfscore": weighted_mean},
        {"weighting": "donor_weighted", "mean_cfscore": donor_weighted_mean},
    ])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"Created: {output_path}")

    return summary


def run_extended_robustness_checks(
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run extended robustness checks including new datasets.

    Includes CMS Medicare, AHRF population weighting, and Bonica validation.

    Returns:
        Dictionary with paths to all output files
    """
    output_dir = output_dir or DIAGNOSTICS_DIR

    # First run standard checks
    results = run_all_robustness_checks(output_dir)

    # CMS Medicare comparison
    print("  Comparing Medicare-active vs all physicians...")
    try:
        medicare_path = output_dir / "medicare_active_comparison.csv"
        compare_medicare_active_vs_all(output_path=medicare_path)
        results["medicare_comparison"] = medicare_path
    except Exception as e:
        print(f"    Skipped: {e}")

    # CMS specialty analysis
    print("  Computing CMS specialty statistics...")
    try:
        specialty_path = output_dir / "cms_specialty_stats.csv"
        compute_cms_specialty_statistics(output_path=specialty_path)
        results["cms_specialty"] = specialty_path
    except Exception as e:
        print(f"    Skipped: {e}")

    # Population-weighted statistics
    print("  Computing population-weighted statistics...")
    try:
        weighted_path = output_dir / "population_weighted_stats.csv"
        compute_population_weighted_statistics(output_path=weighted_path)
        results["population_weighted"] = weighted_path
    except Exception as e:
        print(f"    Skipped: {e}")

    # Bonica validation
    print("  Running Bonica validation...")
    try:
        from .bonica_validation import validate_all_definitions, generate_validation_report
        validation_path = output_dir / "bonica_validation_metrics.csv"
        validate_all_definitions(output_path=validation_path)
        results["bonica_validation"] = validation_path

        generate_validation_report(output_dir=output_dir)
        results["bonica_report"] = output_dir / "bonica_validation_report.txt"
    except Exception as e:
        print(f"    Skipped: {e}")

    print("\nExtended robustness checks complete.")
    return results
