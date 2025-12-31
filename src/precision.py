"""
Precision estimation for physician classification.

This module provides:
1. Likely-negative set construction from NPPES non-physician taxonomies
2. Manual audit sample generation (stratified by p_physician)
3. Calibration curve computation using silver labels
4. Precision metrics computation

These analyses help quantify classification quality beyond just recall.
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from .config import (
    DIAGNOSTICS_DIR,
    DIME_DONORS_PARQUET,
    LIKELY_NEGATIVES_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    NPPES_RAW_DIR,
    PHYSICIAN_LABELS_PARQUET,
    PROCESSED_DATA_DIR,
    ensure_directories,
)

logger = logging.getLogger(__name__)


# Non-physician NPPES taxonomy prefixes for likely negatives
NON_PHYSICIAN_TAXONOMY_PREFIXES = [
    "36",   # Nursing & Nurse Practitioners
    "37",   # Pharmacists & Pharmacy Technicians
    "364",  # Physician Assistants
    "22",   # Dentists
    "11",   # Chiropractors
    "103",  # Psychologists
    "152",  # Optometrists
    "156",  # Veterinarians
    "17",   # Podiatrists (not always considered physicians)
    "18",   # Speech-Language Pathologists
    "19",   # Audiologists
]


def build_likely_negative_set(
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Build a likely-negative set from NPPES non-physician taxonomies.

    Identifies DIME donors who are linked to NPPES records with
    non-physician taxonomies (NP, PA, RN, PharmD, etc.).

    These serve as proxy negatives for precision estimation.

    Args:
        output_path: Output parquet path
        overwrite: Whether to overwrite existing file

    Returns:
        Path to likely negatives file
    """
    output_path = output_path or LIKELY_NEGATIVES_PARQUET

    if output_path.exists() and not overwrite:
        logger.info(f"Output already exists: {output_path}")
        return output_path

    # Find NPPES data file
    nppes_files = list(NPPES_RAW_DIR.glob("npidata_pfile_*.csv"))
    if not nppes_files:
        nppes_files = list(NPPES_RAW_DIR.glob("*.csv"))

    if not nppes_files:
        raise FileNotFoundError(f"No NPPES file found in {NPPES_RAW_DIR}")

    nppes_file = max(nppes_files, key=lambda p: p.stat().st_size)
    logger.info(f"Using NPPES file: {nppes_file}")

    if not DIME_DONORS_PARQUET.exists():
        raise FileNotFoundError(f"DIME donors not found: {DIME_DONORS_PARQUET}")

    logger.info("Building likely-negative set from non-physician taxonomies...")
    con = duckdb.connect()

    donors_path = str(DIME_DONORS_PARQUET).replace("\\", "/")
    nppes_path = str(nppes_file).replace("\\", "/")

    # Build taxonomy prefix conditions for non-physicians
    taxonomy_conditions = []
    for prefix in NON_PHYSICIAN_TAXONOMY_PREFIXES:
        for col_num in range(1, 16):
            col = f'"Healthcare Provider Taxonomy Code_{col_num}"'
            taxonomy_conditions.append(f"{col} LIKE '{prefix}%'")

    taxonomy_filter = " OR ".join(taxonomy_conditions)

    # Extract non-physician providers from NPPES
    logger.info("Extracting non-physician providers from NPPES...")

    try:
        # Get non-physician NPIs
        non_physician_npis = con.execute(f"""
            SELECT DISTINCT
                NPI AS npi,
                "Provider First Name" AS first_name,
                "Provider Last Name (Legal Name)" AS last_name,
                "Provider Business Practice Location Address State Name" AS state,
                "Healthcare Provider Taxonomy Code_1" AS taxonomy_code_1
            FROM read_csv_auto(
                '{nppes_path}',
                header=true,
                all_varchar=true,
                sample_size=100000,
                ignore_errors=true
            )
            WHERE "Entity Type Code" = '1'
              AND "NPI Deactivation Date" IS NULL
              AND ({taxonomy_filter})
              -- Exclude anyone who also has physician codes
              AND NOT "Healthcare Provider Taxonomy Code_1" LIKE '20%'
              AND NOT "Healthcare Provider Taxonomy Code_2" LIKE '20%'
              AND NOT "Healthcare Provider Taxonomy Code_3" LIKE '20%'
              AND NOT "Healthcare Provider Taxonomy Code_4" LIKE '20%'
              AND NOT "Healthcare Provider Taxonomy Code_5" LIKE '20%'
        """).fetchdf()

        logger.info(f"Found {len(non_physician_npis):,} non-physician NPIs")

    except Exception as e:
        logger.error(f"Error reading NPPES: {e}")
        con.close()
        raise

    if len(non_physician_npis) == 0:
        logger.warning("No non-physician NPIs found")
        con.close()
        return output_path

    # Check if we have linkage results to find DIME matches
    if LINKAGE_RESULTS_PARQUET.exists():
        linkage_path = str(LINKAGE_RESULTS_PARQUET).replace("\\", "/")

        # Find DIME donors linked to non-physician NPIs
        non_phys_npi_list = ", ".join([f"'{npi}'" for npi in non_physician_npis["npi"].head(50000)])

        likely_negatives = con.execute(f"""
            SELECT
                l.bonica_cid,
                l.npi,
                l.match_score,
                'nppes_non_physician' AS negative_source
            FROM read_parquet('{linkage_path}') l
            WHERE l.npi IN ({non_phys_npi_list})
        """).fetchdf()

        logger.info(f"Found {len(likely_negatives):,} DIME donors linked to non-physician NPIs")
    else:
        # Without linkage, we can't identify DIME matches
        logger.warning("No linkage results found - cannot identify DIME non-physician matches")
        likely_negatives = pd.DataFrame()

    con.close()

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(likely_negatives) > 0:
        likely_negatives.to_parquet(output_path, compression="zstd", index=False)
        logger.info(f"Created: {output_path}")
    else:
        # Create empty file with schema
        pd.DataFrame(columns=["bonica_cid", "npi", "match_score", "negative_source"]).to_parquet(
            output_path, compression="zstd", index=False
        )
        logger.info(f"Created empty likely negatives file: {output_path}")

    return output_path


def create_manual_audit_sample(
    output_path: Optional[Path] = None,
    sample_size: int = 500,
    random_seed: int = 42,
) -> Path:
    """
    Create a stratified sample for manual audit.

    Samples across p_physician deciles and rule tiers to ensure
    comprehensive coverage of classification uncertainty.

    Args:
        output_path: Output CSV path
        sample_size: Total number of records to sample
        random_seed: Random seed for reproducibility

    Returns:
        Path to audit sample file
    """
    output_path = output_path or (DIAGNOSTICS_DIR / "manual_audit_sample.csv")

    if not PHYSICIAN_LABELS_PARQUET.exists():
        raise FileNotFoundError(f"Labels not found: {PHYSICIAN_LABELS_PARQUET}")

    if not DIME_DONORS_PARQUET.exists():
        raise FileNotFoundError(f"Donors not found: {DIME_DONORS_PARQUET}")

    logger.info(f"Creating manual audit sample (n={sample_size})...")
    con = duckdb.connect()

    labels_path = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
    donors_path = str(DIME_DONORS_PARQUET).replace("\\", "/")

    # Load labels with donor info
    df = con.execute(f"""
        SELECT
            l.bonica_cid,
            l.p_physician,
            l.physician_final,
            l.physician_naive,
            l.physician_rule_label,
            d.contributor_occupation,
            d.contributor_employer,
            d.contributor_state
        FROM read_parquet('{labels_path}') l
        LEFT JOIN read_parquet('{donors_path}') d ON l.bonica_cid = d.bonica_cid
        WHERE l.p_physician IS NOT NULL
    """).fetchdf()

    con.close()

    logger.info(f"Loaded {len(df):,} records with p_physician scores")

    # Create p_physician deciles
    df["p_decile"] = pd.qcut(
        df["p_physician"],
        q=10,
        labels=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"],
        duplicates="drop",
    )

    # Stratified sampling by decile
    np.random.seed(random_seed)
    samples_per_decile = sample_size // 10

    sampled = []
    for decile in df["p_decile"].unique():
        decile_df = df[df["p_decile"] == decile]
        n_sample = min(samples_per_decile, len(decile_df))
        sampled.append(decile_df.sample(n=n_sample, random_state=random_seed))

    sample_df = pd.concat(sampled, ignore_index=True)

    # Add columns for manual labeling
    sample_df["manual_label"] = ""  # Blank for annotator
    sample_df["annotator"] = ""
    sample_df["notes"] = ""

    # Reorder columns
    output_cols = [
        "bonica_cid",
        "p_physician",
        "p_decile",
        "physician_final",
        "physician_naive",
        "physician_rule_label",
        "contributor_occupation",
        "contributor_employer",
        "contributor_state",
        "manual_label",
        "annotator",
        "notes",
    ]
    sample_df = sample_df[[c for c in output_cols if c in sample_df.columns]]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False)

    logger.info(f"Created: {output_path}")
    logger.info(f"Total samples: {len(sample_df)}")
    logger.info(f"Samples per decile: {sample_df['p_decile'].value_counts().to_dict()}")

    return output_path


def compute_calibration_curve(
    output_path: Optional[Path] = None,
    n_bins: int = 10,
) -> Path:
    """
    Compute calibration curve using available silver labels.

    Uses:
    - Bonica reference data (confirmed physicians)
    - NPPES linkage (high-confidence matches)
    - Likely negatives (non-physician NPPES matches)

    Args:
        output_path: Output path for calibration report
        n_bins: Number of probability bins

    Returns:
        Path to calibration report
    """
    output_path = output_path or (DIAGNOSTICS_DIR / "calibration_report.txt")

    if not PHYSICIAN_LABELS_PARQUET.exists():
        raise FileNotFoundError(f"Labels not found: {PHYSICIAN_LABELS_PARQUET}")

    logger.info("Computing calibration curve...")
    con = duckdb.connect()

    labels_path = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")

    # Load predictions
    labels_df = con.execute(f"""
        SELECT bonica_cid, p_physician, physician_final
        FROM read_parquet('{labels_path}')
        WHERE p_physician IS NOT NULL
    """).fetchdf()

    # Load silver labels from various sources
    silver_labels = {}

    # 1. NPPES linkage (confirmed physicians)
    if LINKAGE_RESULTS_PARQUET.exists():
        linkage_path = str(LINKAGE_RESULTS_PARQUET).replace("\\", "/")
        linked = con.execute(f"""
            SELECT DISTINCT bonica_cid
            FROM read_parquet('{linkage_path}')
            WHERE match_score >= 0.85
        """).fetchdf()
        for cid in linked["bonica_cid"]:
            silver_labels[cid] = 1
        logger.info(f"NPPES positives: {len(linked):,}")

    # 2. Likely negatives
    if LIKELY_NEGATIVES_PARQUET.exists():
        negatives_path = str(LIKELY_NEGATIVES_PARQUET).replace("\\", "/")
        negatives = con.execute(f"""
            SELECT DISTINCT bonica_cid
            FROM read_parquet('{negatives_path}')
        """).fetchdf()
        for cid in negatives["bonica_cid"]:
            silver_labels[cid] = 0
        logger.info(f"Likely negatives: {len(negatives):,}")

    con.close()

    if not silver_labels:
        logger.warning("No silver labels available for calibration")
        with open(output_path, "w") as f:
            f.write("No silver labels available for calibration.\n")
            f.write("Run build_likely_negative_set first.\n")
        return output_path

    # Add silver labels to predictions
    labels_df["silver_label"] = labels_df["bonica_cid"].map(silver_labels)
    calibration_df = labels_df[labels_df["silver_label"].notna()].copy()

    logger.info(f"Records with silver labels: {len(calibration_df):,}")

    # Compute calibration by p_physician bins
    calibration_df["p_bin"] = pd.cut(
        calibration_df["p_physician"],
        bins=n_bins,
        labels=[f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}" for i in range(n_bins)],
    )

    calibration = calibration_df.groupby("p_bin").agg(
        n_samples=("bonica_cid", "count"),
        mean_p=("p_physician", "mean"),
        actual_positive_rate=("silver_label", "mean"),
    ).reset_index()

    # Generate report
    lines = [
        "=" * 60,
        "Physician Classification Calibration Report",
        "=" * 60,
        "",
        f"Total records with silver labels: {len(calibration_df):,}",
        f"Silver positives (NPPES-linked): {(calibration_df['silver_label'] == 1).sum():,}",
        f"Silver negatives (non-physician): {(calibration_df['silver_label'] == 0).sum():,}",
        "",
        "Calibration by p_physician bin:",
        "-" * 60,
        f"{'Bin':<15} {'N':<10} {'Mean P':<10} {'Actual+':<10}",
        "-" * 60,
    ]

    for _, row in calibration.iterrows():
        lines.append(
            f"{row['p_bin']:<15} {row['n_samples']:<10} "
            f"{row['mean_p']:.3f}      {row['actual_positive_rate']:.3f}"
        )

    # Compute overall metrics
    from sklearn.metrics import brier_score_loss, log_loss

    try:
        brier = brier_score_loss(calibration_df["silver_label"], calibration_df["p_physician"])
        logloss = log_loss(calibration_df["silver_label"], calibration_df["p_physician"])

        lines.extend([
            "",
            "-" * 60,
            f"Brier Score: {brier:.4f}",
            f"Log Loss: {logloss:.4f}",
        ])
    except Exception as e:
        lines.append(f"\nCould not compute Brier/Log Loss: {e}")

    lines.extend([
        "",
        "=" * 60,
        "Notes:",
        "- Lower Brier score = better calibration",
        "- Actual+ should match Mean P if well-calibrated",
        "=" * 60,
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Created: {output_path}")

    # Also save calibration data as CSV
    csv_path = output_path.with_suffix(".csv")
    calibration.to_csv(csv_path, index=False)

    return output_path


def compute_precision_metrics(
    output_path: Optional[Path] = None,
    threshold: float = 0.5,
) -> Path:
    """
    Compute precision metrics using silver labels.

    Args:
        output_path: Output CSV path
        threshold: Classification threshold for p_physician

    Returns:
        Path to precision metrics file
    """
    output_path = output_path or (DIAGNOSTICS_DIR / "precision_proxy_eval.csv")

    if not PHYSICIAN_LABELS_PARQUET.exists():
        raise FileNotFoundError(f"Labels not found: {PHYSICIAN_LABELS_PARQUET}")

    logger.info("Computing precision metrics...")
    con = duckdb.connect()

    labels_path = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")

    labels_df = con.execute(f"""
        SELECT bonica_cid, p_physician, physician_final
        FROM read_parquet('{labels_path}')
    """).fetchdf()

    # Load silver labels
    silver_labels = {}

    if LINKAGE_RESULTS_PARQUET.exists():
        linkage_path = str(LINKAGE_RESULTS_PARQUET).replace("\\", "/")
        linked = con.execute(f"""
            SELECT DISTINCT bonica_cid
            FROM read_parquet('{linkage_path}')
            WHERE match_score >= 0.85
        """).fetchdf()
        for cid in linked["bonica_cid"]:
            silver_labels[cid] = 1

    if LIKELY_NEGATIVES_PARQUET.exists():
        negatives_path = str(LIKELY_NEGATIVES_PARQUET).replace("\\", "/")
        negatives = con.execute(f"""
            SELECT DISTINCT bonica_cid
            FROM read_parquet('{negatives_path}')
        """).fetchdf()
        for cid in negatives["bonica_cid"]:
            silver_labels[cid] = 0

    con.close()

    labels_df["silver_label"] = labels_df["bonica_cid"].map(silver_labels)
    eval_df = labels_df[labels_df["silver_label"].notna()].copy()

    if len(eval_df) == 0:
        logger.warning("No silver labels for evaluation")
        return output_path

    # Binary predictions at threshold
    eval_df["predicted"] = (eval_df["p_physician"] >= threshold).astype(int)

    # Confusion matrix
    tp = ((eval_df["predicted"] == 1) & (eval_df["silver_label"] == 1)).sum()
    fp = ((eval_df["predicted"] == 1) & (eval_df["silver_label"] == 0)).sum()
    tn = ((eval_df["predicted"] == 0) & (eval_df["silver_label"] == 0)).sum()
    fn = ((eval_df["predicted"] == 0) & (eval_df["silver_label"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(eval_df)

    metrics = {
        "threshold": threshold,
        "n_eval": len(eval_df),
        "n_positive": int((eval_df["silver_label"] == 1).sum()),
        "n_negative": int((eval_df["silver_label"] == 0).sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
    }

    metrics_df = pd.DataFrame([metrics])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    logger.info(f"Created: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Precision Evaluation (Silver Labels)")
    print("=" * 60)
    print(f"Threshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print("=" * 60)

    return output_path


def run_all_precision_analyses(
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Run all precision analyses.

    Args:
        output_dir: Directory for outputs

    Returns:
        Dictionary of output paths
    """
    output_dir = output_dir or DIAGNOSTICS_DIR
    ensure_directories()

    outputs = {}

    print("=" * 70)
    print("Running Precision Analyses")
    print("=" * 70)

    # 1. Build likely negatives
    print("\n--- Building Likely Negatives ---")
    try:
        outputs["likely_negatives"] = build_likely_negative_set()
    except Exception as e:
        logger.warning(f"Skipping likely negatives: {e}")

    # 2. Create manual audit sample
    print("\n--- Creating Manual Audit Sample ---")
    try:
        outputs["audit_sample"] = create_manual_audit_sample()
    except Exception as e:
        logger.warning(f"Skipping audit sample: {e}")

    # 3. Compute calibration
    print("\n--- Computing Calibration ---")
    try:
        outputs["calibration"] = compute_calibration_curve()
    except Exception as e:
        logger.warning(f"Skipping calibration: {e}")

    # 4. Compute precision metrics
    print("\n--- Computing Precision Metrics ---")
    try:
        outputs["precision"] = compute_precision_metrics()
    except Exception as e:
        logger.warning(f"Skipping precision: {e}")

    print("\n" + "=" * 70)
    print("Precision analyses complete!")
    print(f"Output files: {len(outputs)}")
    print("=" * 70)

    return outputs
