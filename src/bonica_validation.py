"""
Validation against Bonica's curated JAMA-IM physician dataset.

Provides "silver label" validation for physician classification by comparing
our rule-based/ML classifier against Bonica's manually curated labels.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from .config import (
    BONICA_RAW_DIR,
    BONICA_REFERENCE_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    DIME_DONORS_PARQUET,
    DIAGNOSTICS_DIR,
)


# Path to source CSV
BONICA_MD_DONORS_CSV = BONICA_RAW_DIR / "md_donors.csv"


@dataclass
class ValidationMetrics:
    """Metrics from comparing classifier against reference."""
    n_reference: int           # Total in Bonica reference
    n_matched_cids: int        # Reference records with matching bonica_cid in our data
    n_true_positive: int       # We classified as physician AND in Bonica
    n_false_negative: int      # Bonica says physician, we missed
    n_our_physicians: int      # Total we classified as physician
    recall: float              # TP / (TP + FN) - coverage of Bonica's list

    def to_dict(self) -> dict:
        return {
            "n_reference": self.n_reference,
            "n_matched_cids": self.n_matched_cids,
            "n_true_positive": self.n_true_positive,
            "n_false_negative": self.n_false_negative,
            "n_our_physicians": self.n_our_physicians,
            "recall": self.recall,
        }


def ingest_bonica_reference(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Ingest Bonica JAMA-IM CSV to Parquet format for efficient querying.

    Args:
        input_path: Path to md_donors.csv
        output_path: Output parquet path
        overwrite: Whether to overwrite existing file

    Returns:
        Path to output parquet file
    """
    input_path = input_path or BONICA_MD_DONORS_CSV
    output_path = output_path or BONICA_REFERENCE_PARQUET

    if output_path.exists() and not overwrite:
        print(f"Bonica reference parquet already exists: {output_path}")
        return output_path

    if not input_path.exists():
        raise FileNotFoundError(f"Bonica md_donors.csv not found: {input_path}")

    print(f"Ingesting Bonica reference data: {input_path.name}")

    con = duckdb.connect()

    csv_path = str(input_path).replace("\\", "/")
    output_path_str = str(output_path).replace("\\", "/")

    # Count total records (use all_varchar to handle NA values)
    total = con.execute(f"""
        SELECT COUNT(*) FROM read_csv_auto('{csv_path}', header=true, all_varchar=true)
    """).fetchone()[0]
    print(f"  Total Bonica records: {total:,}")

    # Extract key columns - use all_varchar=true to handle NA values
    query = f"""
    COPY (
        SELECT
            CAST(bonica_cid AS VARCHAR) as bonica_cid,
            CASE WHEN npi = 'NA' OR npi IS NULL THEN NULL ELSE npi END as bonica_npi,
            CASE WHEN upin = 'NA' OR upin IS NULL THEN NULL ELSE upin END as bonica_upin,
            lname as bonica_last_name,
            fname as bonica_first_name,
            credential as bonica_credential,
            gender as bonica_gender,
            business_state as bonica_state,
            business_zipcode as bonica_zip,
            specialty_norm as bonica_specialty,
            specialty_name1 as bonica_specialty_detail,
            taxonomy1 as bonica_taxonomy,
            TRY_CAST(CASE WHEN total_to_rep = 'NA' THEN NULL ELSE total_to_rep END AS DOUBLE) as bonica_total_to_rep,
            TRY_CAST(CASE WHEN total_to_dem = 'NA' THEN NULL ELSE total_to_dem END AS DOUBLE) as bonica_total_to_dem,
            TRY_CAST(CASE WHEN p_to_reps = 'NA' THEN NULL ELSE p_to_reps END AS DOUBLE) as bonica_p_to_reps,
            TRY_CAST(CASE WHEN total_donate = 'NA' THEN NULL ELSE total_donate END AS DOUBLE) as bonica_total_donated
        FROM read_csv_auto('{csv_path}', header=true, all_varchar=true, null_padding=true)
        WHERE bonica_cid IS NOT NULL
          AND TRIM(CAST(bonica_cid AS VARCHAR)) != ''
          AND bonica_cid != 'NA'
    ) TO '{output_path_str}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(query)

    # Get summary
    stats = con.execute(f"""
        SELECT
            COUNT(*) as n_total,
            COUNT(DISTINCT bonica_cid) as n_unique_cids,
            COUNT(bonica_npi) as n_with_npi,
            COUNT(DISTINCT bonica_specialty) as n_specialties
        FROM read_parquet('{output_path_str}')
    """).fetchone()

    con.close()

    print(f"Created: {output_path}")
    print(f"  Total records: {stats[0]:,}")
    print(f"  Unique bonica_cid: {stats[1]:,}")
    print(f"  With NPI: {stats[2]:,}")
    print(f"  Specialties: {stats[3]}")

    return output_path


def validate_classifier(
    labels_path: Optional[Path] = None,
    bonica_path: Optional[Path] = None,
    physician_col: str = "physician_final",
) -> ValidationMetrics:
    """
    Validate our classifier against Bonica's reference.

    Computes recall: What fraction of Bonica's physicians did we catch?

    Args:
        labels_path: Our physician labels parquet
        bonica_path: Bonica reference parquet
        physician_col: Which column to validate (physician_naive, physician_rule_label, physician_final)

    Returns:
        ValidationMetrics dataclass
    """
    labels_path = labels_path or PHYSICIAN_LABELS_PARQUET
    bonica_path = bonica_path or BONICA_REFERENCE_PARQUET

    if not labels_path.exists():
        raise FileNotFoundError(f"Physician labels not found: {labels_path}")
    if not bonica_path.exists():
        raise FileNotFoundError(f"Bonica reference not found: {bonica_path}")

    con = duckdb.connect()

    labels_p = str(labels_path).replace("\\", "/")
    bonica_p = str(bonica_path).replace("\\", "/")

    # Get Bonica reference count
    n_reference = con.execute(f"""
        SELECT COUNT(DISTINCT bonica_cid)
        FROM read_parquet('{bonica_p}')
    """).fetchone()[0]

    # Get our total physicians
    n_our_physicians = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{labels_p}')
        WHERE {physician_col} = true
    """).fetchone()[0]

    # Compute overlap metrics
    metrics = con.execute(f"""
        WITH bonica_cids AS (
            SELECT DISTINCT bonica_cid
            FROM read_parquet('{bonica_p}')
        ),
        our_labels AS (
            SELECT bonica_cid, {physician_col} as our_label
            FROM read_parquet('{labels_p}')
        ),
        matched AS (
            SELECT
                b.bonica_cid as bonica_cid,
                COALESCE(l.our_label, false) as our_label
            FROM bonica_cids b
            LEFT JOIN our_labels l ON b.bonica_cid = l.bonica_cid
        )
        SELECT
            COUNT(*) as n_matched,
            SUM(CASE WHEN our_label THEN 1 ELSE 0 END) as n_true_positive,
            SUM(CASE WHEN NOT our_label THEN 1 ELSE 0 END) as n_false_negative
        FROM matched
    """).fetchone()

    con.close()

    n_matched = metrics[0]
    n_true_positive = metrics[1]
    n_false_negative = metrics[2]

    recall = n_true_positive / n_matched if n_matched > 0 else 0

    return ValidationMetrics(
        n_reference=n_reference,
        n_matched_cids=n_matched,
        n_true_positive=n_true_positive,
        n_false_negative=n_false_negative,
        n_our_physicians=n_our_physicians,
        recall=recall,
    )


def validate_all_definitions(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Validate all physician definitions against Bonica reference.

    Returns DataFrame with recall metrics for each definition.
    """
    definitions = ["physician_naive", "physician_rule_label", "physician_final"]

    results = []
    for defn in definitions:
        try:
            metrics = validate_classifier(physician_col=defn)
            result = metrics.to_dict()
            result["definition"] = defn
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not validate {defn}: {e}")

    df = pd.DataFrame(results)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Created: {output_path}")

    return df


def analyze_false_negatives(
    output_path: Optional[Path] = None,
    physician_col: str = "physician_final",
) -> pd.DataFrame:
    """
    Analyze physicians in Bonica that our classifier missed.

    Identifies patterns in:
    - Specialties that are underrepresented
    - Occupation strings we failed to match

    Returns:
        DataFrame with false negative analysis
    """
    if not BONICA_REFERENCE_PARQUET.exists():
        raise FileNotFoundError("Bonica reference not found")
    if not PHYSICIAN_LABELS_PARQUET.exists():
        raise FileNotFoundError("Physician labels not found")
    if not DIME_DONORS_PARQUET.exists():
        raise FileNotFoundError("DIME donors not found")

    con = duckdb.connect()

    bonica_p = str(BONICA_REFERENCE_PARQUET).replace("\\", "/")
    labels_p = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
    donors_p = str(DIME_DONORS_PARQUET).replace("\\", "/")

    # Find false negatives with context
    query = f"""
        SELECT
            b.bonica_cid,
            b.bonica_specialty,
            b.bonica_credential,
            d.contributor_occupation,
            d.contributor_employer,
            l.physician_rule_tier,
            l.p_physician
        FROM read_parquet('{bonica_p}') b
        LEFT JOIN read_parquet('{donors_p}') d ON b.bonica_cid = d.bonica_cid
        LEFT JOIN read_parquet('{labels_p}') l ON b.bonica_cid = l.bonica_cid
        WHERE l.{physician_col} = false OR l.{physician_col} IS NULL
        LIMIT 100000
    """

    missed_df = con.execute(query).fetchdf()

    # Analyze by specialty
    specialty_counts = missed_df["bonica_specialty"].value_counts().head(20)

    # Analyze by rule tier
    tier_counts = missed_df["physician_rule_tier"].value_counts()

    con.close()

    analysis = {
        "total_missed": len(missed_df),
        "by_specialty": specialty_counts.to_dict(),
        "by_rule_tier": tier_counts.to_dict(),
    }

    # Save sample of missed records
    if output_path:
        missed_df.head(1000).to_csv(output_path, index=False)
        print(f"Created: {output_path}")
        print(f"  Total missed: {len(missed_df):,}")
        print(f"  Top specialties missed: {list(specialty_counts.head(5).index)}")

    return missed_df, analysis


def compare_ideology_distributions(
    output_path: Optional[Path] = None,
) -> dict:
    """
    Compare ideology distributions between:
    1. Bonica's physician sample (using p_to_reps)
    2. Our classifier's physician sample (using CFScore)

    Note: CFScore and p_to_reps are on different scales:
    - CFScore: continuous, typically -2 to +2, negative=liberal
    - p_to_reps: 0-1, fraction donated to Republicans

    Returns:
        Dictionary with distribution comparison statistics
    """
    if not BONICA_REFERENCE_PARQUET.exists():
        raise FileNotFoundError("Bonica reference not found")
    if not PHYSICIAN_LABELS_PARQUET.exists():
        raise FileNotFoundError("Physician labels not found")
    if not DIME_DONORS_PARQUET.exists():
        raise FileNotFoundError("DIME donors not found")

    con = duckdb.connect()

    bonica_p = str(BONICA_REFERENCE_PARQUET).replace("\\", "/")
    labels_p = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
    donors_p = str(DIME_DONORS_PARQUET).replace("\\", "/")

    # Bonica's ideology distribution (p_to_reps)
    bonica_stats = con.execute(f"""
        SELECT
            COUNT(*) as n,
            AVG(bonica_p_to_reps) as mean_p_to_reps,
            APPROX_QUANTILE(bonica_p_to_reps, 0.5) as median_p_to_reps,
            STDDEV(bonica_p_to_reps) as std_p_to_reps,
            AVG(CASE WHEN bonica_p_to_reps > 0.5 THEN 1.0 ELSE 0.0 END) as share_majority_rep
        FROM read_parquet('{bonica_p}')
        WHERE bonica_p_to_reps IS NOT NULL
    """).fetchone()

    # Our classifier's ideology distribution (CFScore)
    our_stats = con.execute(f"""
        SELECT
            COUNT(*) as n,
            AVG(d.contributor_cfscore) as mean_cfscore,
            APPROX_QUANTILE(d.contributor_cfscore, 0.5) as median_cfscore,
            STDDEV(d.contributor_cfscore) as std_cfscore,
            AVG(CASE WHEN d.contributor_cfscore > 0 THEN 1.0 ELSE 0.0 END) as share_right
        FROM read_parquet('{donors_p}') d
        JOIN read_parquet('{labels_p}') l ON d.bonica_cid = l.bonica_cid
        WHERE l.physician_final = true
          AND d.contributor_cfscore IS NOT NULL
    """).fetchone()

    con.close()

    results = {
        "bonica_sample": {
            "n": bonica_stats[0],
            "mean_p_to_reps": bonica_stats[1],
            "median_p_to_reps": bonica_stats[2],
            "std_p_to_reps": bonica_stats[3],
            "share_majority_rep": bonica_stats[4],
        },
        "our_classifier": {
            "n": our_stats[0],
            "mean_cfscore": our_stats[1],
            "median_cfscore": our_stats[2],
            "std_cfscore": our_stats[3],
            "share_right": our_stats[4],
        },
        "note": "CFScore and p_to_reps are on different scales. "
                "p_to_reps is fraction to Republicans (0-1). "
                "CFScore is continuous (-2 to +2, negative=liberal)."
    }

    if output_path:
        import json
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Created: {output_path}")

    return results


def generate_validation_report(
    output_dir: Optional[Path] = None,
) -> str:
    """
    Generate comprehensive validation report comparing
    our classifier to Bonica's reference.

    Returns:
        Report text
    """
    output_dir = output_dir or DIAGNOSTICS_DIR

    lines = [
        "=" * 60,
        "BONICA JAMA-IM VALIDATION REPORT",
        "=" * 60,
        "",
    ]

    # Reference summary
    if BONICA_REFERENCE_PARQUET.exists():
        con = duckdb.connect()
        path = str(BONICA_REFERENCE_PARQUET).replace("\\", "/")
        ref_stats = con.execute(f"""
            SELECT
                COUNT(*) as n_total,
                COUNT(DISTINCT bonica_cid) as n_unique,
                COUNT(bonica_npi) as n_with_npi
            FROM read_parquet('{path}')
        """).fetchone()
        con.close()

        lines.extend([
            "REFERENCE DATASET SUMMARY",
            "-" * 40,
            f"  Total Bonica records: {ref_stats[0]:,}",
            f"  Unique bonica_cid: {ref_stats[1]:,}",
            f"  With NPI: {ref_stats[2]:,}",
            "",
        ])

    # Validation metrics by definition
    lines.extend([
        "VALIDATION METRICS BY DEFINITION",
        "-" * 40,
        f"{'Definition':<25} {'Recall':>10} {'Physicians Found':>18}",
    ])

    for defn in ["physician_naive", "physician_rule_label", "physician_final"]:
        try:
            metrics = validate_classifier(physician_col=defn)
            lines.append(
                f"{defn:<25} {metrics.recall:>10.3f} {metrics.n_true_positive:>18,}"
            )
        except Exception as e:
            lines.append(f"{defn:<25} {'ERROR':>10} {str(e)[:18]:>18}")

    lines.append("")

    # False negative summary
    try:
        _, fn_analysis = analyze_false_negatives()
        lines.extend([
            "FALSE NEGATIVE ANALYSIS",
            "-" * 40,
            f"  Total missed: {fn_analysis['total_missed']:,}",
            "",
            "  Top specialties missed:",
        ])
        for spec, count in list(fn_analysis["by_specialty"].items())[:5]:
            lines.append(f"    {spec}: {count:,}")
    except Exception as e:
        lines.append(f"False negative analysis failed: {e}")

    lines.append("")

    # Ideology comparison
    try:
        ideology = compare_ideology_distributions()
        lines.extend([
            "IDEOLOGY DISTRIBUTION COMPARISON",
            "-" * 40,
            "  Bonica sample (p_to_reps, 0=Dem, 1=Rep):",
            f"    N: {ideology['bonica_sample']['n']:,}",
            f"    Mean: {ideology['bonica_sample']['mean_p_to_reps']:.3f}",
            f"    Median: {ideology['bonica_sample']['median_p_to_reps']:.3f}",
            f"    Share majority Rep: {ideology['bonica_sample']['share_majority_rep']:.1%}",
            "",
            "  Our classifier (CFScore, negative=liberal):",
            f"    N: {ideology['our_classifier']['n']:,}",
            f"    Mean: {ideology['our_classifier']['mean_cfscore']:.3f}",
            f"    Median: {ideology['our_classifier']['median_cfscore']:.3f}",
            f"    Share right (>0): {ideology['our_classifier']['share_right']:.1%}",
        ])
    except Exception as e:
        lines.append(f"Ideology comparison failed: {e}")

    lines.extend(["", "=" * 60])

    report = "\n".join(lines)

    # Save report
    report_path = output_dir / "bonica_validation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Created: {report_path}")

    return report


def get_bonica_stats() -> dict:
    """Get statistics from the processed Bonica reference data."""
    if not BONICA_REFERENCE_PARQUET.exists():
        return {}

    con = duckdb.connect()
    path = str(BONICA_REFERENCE_PARQUET).replace("\\", "/")

    stats = {}
    stats["n_total"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_unique_cids"] = con.execute(
        f"SELECT COUNT(DISTINCT bonica_cid) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_with_npi"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}') WHERE bonica_npi IS NOT NULL"
    ).fetchone()[0]

    stats["n_specialties"] = con.execute(
        f"SELECT COUNT(DISTINCT bonica_specialty) FROM read_parquet('{path}')"
    ).fetchone()[0]

    # Top specialties
    top_specs = con.execute(f"""
        SELECT bonica_specialty, COUNT(*) as n
        FROM read_parquet('{path}')
        WHERE bonica_specialty IS NOT NULL
        GROUP BY bonica_specialty
        ORDER BY n DESC
        LIMIT 10
    """).fetchdf()

    stats["top_specialties"] = top_specs.to_dict("records")

    con.close()
    return stats
