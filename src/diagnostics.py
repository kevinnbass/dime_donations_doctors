"""
Diagnostics for physician identification coverage.

Compares rule-based classifications to NPPES ground truth
and analyzes false negatives to identify missing patterns.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    DIME_DONORS_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    DIAGNOSTICS_DIR,
)
from .physician_rules import PhysicianRuleClassifier, get_rule_classifier


def compare_rule_vs_nppes(
    donors_path: Optional[Path] = None,
    linkage_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare rule-based classifications to NPPES linkage results.

    Creates a confusion matrix comparing:
    - rule_positive: Rule-based classifier says physician (is_physician=True)
    - nppes_match: Has high-confidence NPPES linkage match

    Args:
        donors_path: Path to donors parquet
        linkage_path: Path to linkage results parquet
        labels_path: Path to physician labels parquet

    Returns:
        DataFrame with columns:
        - bonica_cid: Contributor ID
        - occupation, employer: Original text fields
        - rule_positive: Rule classifier result
        - rule_score: Rule classifier probability
        - rule_tier: Which tier matched
        - nppes_match: Has NPPES linkage
        - classification: TP, FP, TN, FN
    """
    donors_path = donors_path or DIME_DONORS_PARQUET
    linkage_path = linkage_path or LINKAGE_RESULTS_PARQUET
    labels_path = labels_path or PHYSICIAN_LABELS_PARQUET

    if not donors_path.exists():
        raise FileNotFoundError(f"Donors file not found: {donors_path}")

    # Load donors
    print("Loading donors data...")
    donors = pd.read_parquet(
        donors_path,
        columns=["bonica_cid", "contributor_name", "contributor_occupation", "contributor_employer"],
    )

    # Initialize classifier
    classifier = get_rule_classifier()

    # Classify all donors
    print(f"Classifying {len(donors):,} donors with rule-based classifier...")
    results = []
    for _, row in donors.iterrows():
        result = classifier.classify(
            occupation=row.get("contributor_occupation"),
            employer=row.get("contributor_employer"),
            name=row.get("contributor_name"),
        )
        results.append({
            "bonica_cid": row["bonica_cid"],
            "occupation": row.get("contributor_occupation", ""),
            "employer": row.get("contributor_employer", ""),
            "rule_positive": result["is_physician"],
            "rule_score": result["score"],
            "rule_tier": result["tier"],
        })

    df = pd.DataFrame(results)

    # Add NPPES linkage if available
    if linkage_path.exists():
        print("Loading NPPES linkage results...")
        linkage = pd.read_parquet(
            linkage_path,
            columns=["bonica_cid", "npi", "linkage_score"],
        )
        # A "match" is having any NPPES linkage (already filtered by threshold)
        linked_cids = set(linkage["bonica_cid"].unique())
        df["nppes_match"] = df["bonica_cid"].isin(linked_cids)
    else:
        print("Warning: NPPES linkage file not found, using labels file...")
        if labels_path.exists():
            labels = pd.read_parquet(
                labels_path,
                columns=["bonica_cid", "physician_nppes_linked"],
            )
            df = df.merge(labels[["bonica_cid", "physician_nppes_linked"]], on="bonica_cid", how="left")
            df["nppes_match"] = df["physician_nppes_linked"].fillna(False)
            df = df.drop(columns=["physician_nppes_linked"])
        else:
            print("Warning: No NPPES data available, cannot compute confusion matrix")
            df["nppes_match"] = False

    # Compute classification (only meaningful if we have NPPES data)
    def classify_result(row):
        if row["rule_positive"] and row["nppes_match"]:
            return "TP"  # True positive: both agree
        elif row["rule_positive"] and not row["nppes_match"]:
            return "FP"  # False positive: rule says yes, NPPES says no
        elif not row["rule_positive"] and row["nppes_match"]:
            return "FN"  # False negative: rule says no, NPPES says yes
        else:
            return "TN"  # True negative: both agree not physician

    df["classification"] = df.apply(classify_result, axis=1)

    return df


def get_confusion_matrix(comparison_df: pd.DataFrame) -> dict:
    """
    Compute confusion matrix from comparison results.

    Args:
        comparison_df: Output from compare_rule_vs_nppes()

    Returns:
        Dictionary with counts and metrics
    """
    counts = comparison_df["classification"].value_counts().to_dict()

    tp = counts.get("TP", 0)
    fp = counts.get("FP", 0)
    fn = counts.get("FN", 0)
    tn = counts.get("TN", 0)

    total = tp + fp + fn + tn
    total_nppes = tp + fn  # All NPPES-verified physicians

    # Metrics (using NPPES as ground truth)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "total": total,
        "total_nppes_physicians": total_nppes,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
    }


def analyze_false_negatives(
    comparison_df: pd.DataFrame,
    top_n: int = 100,
) -> dict:
    """
    Analyze occupation/employer strings from false negatives.

    These are NPPES-verified physicians missed by the rule classifier.

    Args:
        comparison_df: Output from compare_rule_vs_nppes()
        top_n: Number of top terms to return

    Returns:
        Dictionary with:
        - occupation_terms: Counter of words in occupation field
        - employer_terms: Counter of words in employer field
        - sample_occupations: List of full occupation strings
        - sample_employers: List of full employer strings
    """
    fn_df = comparison_df[comparison_df["classification"] == "FN"].copy()

    print(f"Analyzing {len(fn_df):,} false negatives...")

    # Tokenize and count occupation terms
    occupation_terms = Counter()
    sample_occupations = []

    for occ in fn_df["occupation"].dropna().head(500):
        if occ and len(occ) > 2:
            sample_occupations.append(occ)
            # Tokenize: split on non-alphanumeric, lowercase
            tokens = re.findall(r"[a-zA-Z]+", occ.lower())
            occupation_terms.update(tokens)

    # Tokenize and count employer terms
    employer_terms = Counter()
    sample_employers = []

    for emp in fn_df["employer"].dropna().head(500):
        if emp and len(emp) > 2:
            sample_employers.append(emp)
            tokens = re.findall(r"[a-zA-Z]+", emp.lower())
            employer_terms.update(tokens)

    # Filter out common stop words
    stop_words = {
        "the", "and", "of", "in", "to", "a", "an", "for", "is", "on", "at",
        "by", "with", "from", "or", "as", "be", "this", "that", "it", "not",
        "are", "was", "were", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might",
        "self", "employed", "retired", "none", "na", "unknown", "other",
        "inc", "llc", "llp", "pc", "pa", "pllc", "corp", "corporation",
    }

    occupation_filtered = Counter({
        k: v for k, v in occupation_terms.items()
        if k not in stop_words and len(k) > 2
    })

    employer_filtered = Counter({
        k: v for k, v in employer_terms.items()
        if k not in stop_words and len(k) > 2
    })

    return {
        "occupation_terms": occupation_filtered.most_common(top_n),
        "employer_terms": employer_filtered.most_common(top_n),
        "sample_occupations": sample_occupations[:top_n],
        "sample_employers": sample_employers[:top_n],
        "total_false_negatives": len(fn_df),
    }


def suggest_patterns(
    analysis: dict,
    existing_patterns: Optional[list] = None,
) -> list[str]:
    """
    Suggest regex patterns based on false negative analysis.

    Args:
        analysis: Output from analyze_false_negatives()
        existing_patterns: List of patterns already in config

    Returns:
        List of suggested patterns to add
    """
    existing_patterns = existing_patterns or []

    # Terms that strongly suggest physician
    physician_indicators = {
        "medicine", "medical", "physician", "doctor", "surgeon", "clinic",
        "hospital", "health", "healthcare", "practice", "specialist",
        "cardio", "neuro", "ortho", "pediatric", "internal", "family",
        "emergency", "radiology", "pathology", "anesthesia", "psychiatry",
    }

    suggestions = []
    occupation_terms = dict(analysis.get("occupation_terms", []))

    # Find high-frequency terms that look like physician indicators
    for term, count in occupation_terms.items():
        if count >= 5:  # At least 5 occurrences
            if any(ind in term for ind in physician_indicators):
                pattern = f"\\\\b{term}\\\\b"
                if pattern not in existing_patterns:
                    suggestions.append(f"{pattern}  # Found {count} times in false negatives")

    # Also look for multi-word patterns in sample occupations
    sample_occs = analysis.get("sample_occupations", [])
    pattern_counts = Counter()

    for occ in sample_occs:
        occ_lower = occ.lower()
        # Look for common multi-word physician patterns
        multi_patterns = [
            r"medical\s+\w+",
            r"\w+\s+physician",
            r"\w+\s+surgeon",
            r"\w+\s+medicine",
            r"\w+\s+specialist",
        ]
        for pat in multi_patterns:
            matches = re.findall(pat, occ_lower)
            pattern_counts.update(matches)

    for pattern, count in pattern_counts.most_common(20):
        if count >= 3:
            suggestions.append(f"# Multi-word pattern: '{pattern}' ({count} occurrences)")

    return suggestions


def generate_coverage_report(
    output_dir: Optional[Path] = None,
    donors_path: Optional[Path] = None,
    linkage_path: Optional[Path] = None,
) -> dict:
    """
    Generate comprehensive coverage diagnostic report.

    Args:
        output_dir: Directory for output files
        donors_path: Path to donors parquet
        linkage_path: Path to linkage results parquet

    Returns:
        Dictionary with summary statistics
    """
    output_dir = output_dir or DIAGNOSTICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHYSICIAN IDENTIFICATION COVERAGE DIAGNOSTICS")
    print("=" * 60)

    # Run comparison
    comparison_df = compare_rule_vs_nppes(
        donors_path=donors_path,
        linkage_path=linkage_path,
    )

    # Save full comparison
    comparison_path = output_dir / "rule_vs_nppes_comparison.parquet"
    comparison_df.to_parquet(comparison_path, index=False)
    print(f"\nSaved comparison data to: {comparison_path}")

    # Compute confusion matrix
    confusion = get_confusion_matrix(comparison_df)

    # Save confusion matrix
    confusion_path = output_dir / "rule_vs_nppes_confusion.csv"
    confusion_df = pd.DataFrame([confusion])
    confusion_df.to_csv(confusion_path, index=False)
    print(f"Saved confusion matrix to: {confusion_path}")

    # Print summary
    print("\n--- Confusion Matrix ---")
    print(f"  True Positives (Rule+, NPPES+):  {confusion['true_positive']:,}")
    print(f"  False Positives (Rule+, NPPES-): {confusion['false_positive']:,}")
    print(f"  False Negatives (Rule-, NPPES+): {confusion['false_negative']:,}")
    print(f"  True Negatives (Rule-, NPPES-):  {confusion['true_negative']:,}")
    print(f"\n--- Metrics (NPPES as ground truth) ---")
    print(f"  Precision: {confusion['precision']:.3f}")
    print(f"  Recall:    {confusion['recall']:.3f}")
    print(f"  F1 Score:  {confusion['f1_score']:.3f}")

    # Analyze false negatives
    print("\n--- False Negative Analysis ---")
    analysis = analyze_false_negatives(comparison_df)

    # Save false negative occupations
    fn_occ_path = output_dir / "false_negative_occupations.csv"
    fn_occ_df = pd.DataFrame(
        analysis["occupation_terms"],
        columns=["term", "count"],
    )
    fn_occ_df.to_csv(fn_occ_path, index=False)
    print(f"Saved occupation term analysis to: {fn_occ_path}")

    # Save false negative employers
    fn_emp_path = output_dir / "false_negative_employers.csv"
    fn_emp_df = pd.DataFrame(
        analysis["employer_terms"],
        columns=["term", "count"],
    )
    fn_emp_df.to_csv(fn_emp_path, index=False)
    print(f"Saved employer term analysis to: {fn_emp_path}")

    # Print top false negative terms
    print(f"\nTotal false negatives: {analysis['total_false_negatives']:,}")
    print("\nTop 20 occupation terms in missed physicians:")
    for term, count in analysis["occupation_terms"][:20]:
        print(f"  {term}: {count}")

    print("\nTop 20 employer terms in missed physicians:")
    for term, count in analysis["employer_terms"][:20]:
        print(f"  {term}: {count}")

    # Generate pattern suggestions
    suggestions = suggest_patterns(analysis)
    if suggestions:
        suggestions_path = output_dir / "suggested_patterns.txt"
        with open(suggestions_path, "w") as f:
            f.write("# Suggested patterns based on false negative analysis\n")
            f.write("# Review these and add appropriate ones to keyword_rules.yaml\n\n")
            for s in suggestions:
                f.write(f"{s}\n")
        print(f"\nSaved pattern suggestions to: {suggestions_path}")

    # Save sample false negative records
    fn_samples_path = output_dir / "false_negative_samples.csv"
    fn_samples = comparison_df[comparison_df["classification"] == "FN"][
        ["bonica_cid", "occupation", "employer", "rule_tier", "rule_score"]
    ].head(500)
    fn_samples.to_csv(fn_samples_path, index=False)
    print(f"Saved sample false negatives to: {fn_samples_path}")

    print("\n" + "=" * 60)
    print("Coverage diagnostics complete!")
    print("=" * 60)

    return {
        "confusion_matrix": confusion,
        "false_negative_analysis": {
            "total": analysis["total_false_negatives"],
            "top_occupation_terms": analysis["occupation_terms"][:10],
            "top_employer_terms": analysis["employer_terms"][:10],
        },
    }


def compare_definitions(
    labels_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare classification across different physician definitions.

    Requires physician_labels.parquet with columns:
    - physician_naive
    - physician_rule_label
    - physician_final

    Args:
        labels_path: Path to physician labels parquet
        output_dir: Directory for output files

    Returns:
        DataFrame with comparison statistics
    """
    labels_path = labels_path or PHYSICIAN_LABELS_PARQUET
    output_dir = output_dir or DIAGNOSTICS_DIR

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = pd.read_parquet(labels_path)

    # Column name mapping (actual -> display)
    col_mapping = {
        "physician_naive": "naive",
        "physician_rule_label": "rule",
        "physician_final": "final",
    }

    # Check which definition columns exist
    def_cols = []
    for col in col_mapping.keys():
        if col in labels.columns:
            def_cols.append(col)

    if not def_cols:
        raise ValueError("No physician definition columns found in labels file")

    # Count by definition
    results = []
    for col in def_cols:
        def_name = col_mapping.get(col, col)
        count = labels[col].sum()
        results.append({
            "definition": def_name,
            "physician_count": int(count),
            "total_donors": len(labels),
            "physician_pct": count / len(labels) * 100,
        })

    results_df = pd.DataFrame(results)

    # Save
    output_path = output_dir / "definition_comparison.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\n--- Physician Count by Definition ---")
    for _, row in results_df.iterrows():
        print(f"  {row['definition']}: {row['physician_count']:,} ({row['physician_pct']:.2f}%)")

    return results_df
