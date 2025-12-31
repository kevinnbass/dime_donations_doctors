"""
Combined physician classification using rules, linkage, and optional ML.

Combines rule-based classification with NPPES linkage results and
optionally trains a classifier on high-confidence labeled data.
"""

from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

from .config import (
    DEFAULT_PHYSICIAN_THRESHOLD,
    DIME_DONORS_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    PROCESSED_DATA_DIR,
)
from .physician_rules import PhysicianRuleClassifier, classify_naive_keywords


class PhysicianClassifier:
    """
    Combined classifier for physician identification.

    Combines:
    1. Rule-based scores from occupation/employer text
    2. NPPES linkage scores
    3. Optional ML model trained on high-confidence labels
    """

    def __init__(
        self,
        rule_classifier: Optional[PhysicianRuleClassifier] = None,
        use_ml: bool = True,
        threshold: float = DEFAULT_PHYSICIAN_THRESHOLD,
    ):
        """
        Initialize the classifier.

        Args:
            rule_classifier: Rule-based classifier instance
            use_ml: Whether to train and use ML model
            threshold: Probability threshold for hard classification
        """
        self.rule_classifier = rule_classifier or PhysicianRuleClassifier()
        self.use_ml = use_ml
        self.threshold = threshold
        self.ml_model: Optional[LogisticRegression] = None
        self.is_fitted = False

    def _get_features(self, row: dict) -> np.ndarray:
        """Extract features for ML model from a donor record."""
        features = []

        # Rule score
        rule_result = self.rule_classifier.classify(
            occupation=row.get("contributor_occupation"),
            employer=row.get("contributor_employer"),
            name=row.get("contributor_name"),
        )
        features.append(rule_result["score"])

        # Tier indicators
        features.append(1.0 if rule_result["tier"] == "tier1" else 0.0)
        features.append(1.0 if rule_result["tier"] == "tier2" else 0.0)
        features.append(1.0 if rule_result["tier"] == "tier3_exclusion" else 0.0)

        # Linkage score (if available)
        linkage_score = row.get("linkage_score", 0.0)
        features.append(linkage_score if linkage_score else 0.0)

        # Has NPPES match
        features.append(1.0 if row.get("nppes_npi") else 0.0)

        return np.array(features)

    def fit(
        self,
        donors_df: pd.DataFrame,
        linkage_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Train the classifier using high-confidence labels.

        High-confidence positive: Tier 1 rule match OR NPPES linkage
        High-confidence negative: Tier 3 exclusion

        Args:
            donors_df: DataFrame with donor information
            linkage_df: Optional DataFrame with linkage results

        Returns:
            Dictionary with training metrics
        """
        if not self.use_ml:
            self.is_fitted = True
            return {"status": "skipped", "reason": "ML disabled"}

        print("Training ML classifier on high-confidence labels...")

        # Merge linkage results if available
        if linkage_df is not None and len(linkage_df) > 0:
            # Get best match per donor
            best_matches = linkage_df.groupby("bonica_cid").first().reset_index()
            donors_df = donors_df.merge(
                best_matches[["bonica_cid", "nppes_npi"]],
                on="bonica_cid",
                how="left",
            )

        # Create labels based on high-confidence criteria
        labels = []
        features_list = []

        for _, row in donors_df.iterrows():
            rule_result = self.rule_classifier.classify(
                occupation=row.get("contributor_occupation"),
                employer=row.get("contributor_employer"),
                name=row.get("contributor_name"),
            )

            has_nppes = pd.notna(row.get("nppes_npi"))

            # High-confidence positive
            if rule_result["tier"] == "tier1" or has_nppes:
                labels.append(1)
                features_list.append(self._get_features(row.to_dict()))
            # High-confidence negative
            elif rule_result["tier"] == "tier3_exclusion":
                labels.append(0)
                features_list.append(self._get_features(row.to_dict()))
            # Ambiguous - skip for training

        if len(labels) < 100:
            print(f"Warning: Only {len(labels)} high-confidence labels. Skipping ML training.")
            self.is_fitted = True
            return {"status": "skipped", "reason": "insufficient_labels", "n_labels": len(labels)}

        X = np.array(features_list)
        y = np.array(labels)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train logistic regression
        self.ml_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self.ml_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.ml_model.predict(X_test)
        y_prob = self.ml_model.predict_proba(X_test)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )

        self.is_fitted = True

        metrics = {
            "status": "trained",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_positive": int(sum(y)),
            "n_negative": int(len(y) - sum(y)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        print(f"ML classifier trained: precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")

        return metrics

    def predict_proba(self, row: dict) -> float:
        """
        Predict physician probability for a single record.

        Args:
            row: Dictionary with donor information

        Returns:
            Probability in [0, 1]
        """
        # Rule-based score
        rule_result = self.rule_classifier.classify(
            occupation=row.get("contributor_occupation"),
            employer=row.get("contributor_employer"),
            name=row.get("contributor_name"),
        )

        # If we have a trained ML model, use it
        if self.use_ml and self.ml_model is not None:
            features = self._get_features(row).reshape(1, -1)
            ml_prob = self.ml_model.predict_proba(features)[0, 1]
            # Combine rule score and ML score
            return 0.5 * rule_result["score"] + 0.5 * ml_prob

        # Otherwise, just use rule score
        return rule_result["score"]

    def predict(self, row: dict) -> bool:
        """
        Predict whether a record is a physician.

        Args:
            row: Dictionary with donor information

        Returns:
            True if predicted physician
        """
        return self.predict_proba(row) >= self.threshold

    def classify_with_details(self, row: dict) -> dict:
        """
        Classify with full details for diagnostics.

        Returns all intermediate scores and tier information
        for analysis and debugging.

        Args:
            row: Dictionary with donor information

        Returns:
            Dictionary with:
            - rule_score: Rule-based probability
            - rule_tier: Which tier matched (tier1, tier2, tier3_exclusion, no_match)
            - rule_is_physician: Rule-based hard classification
            - rule_matched_patterns: List of pattern types that matched
            - ml_score: ML model probability (if trained)
            - has_nppes: Whether NPPES linkage exists
            - combined_score: Final combined probability
            - is_physician: Final hard classification
        """
        # Get rule classification
        rule_result = self.rule_classifier.classify(
            occupation=row.get("contributor_occupation"),
            employer=row.get("contributor_employer"),
            name=row.get("contributor_name"),
        )

        result = {
            "rule_score": rule_result["score"],
            "rule_tier": rule_result["tier"],
            "rule_is_physician": rule_result["is_physician"],
            "rule_matched_patterns": rule_result["matched_patterns"],
            "ml_score": None,
            "has_nppes": bool(row.get("nppes_npi")),
        }

        # Get ML score if available
        if self.use_ml and self.ml_model is not None:
            features = self._get_features(row).reshape(1, -1)
            result["ml_score"] = float(self.ml_model.predict_proba(features)[0, 1])
            result["combined_score"] = 0.5 * rule_result["score"] + 0.5 * result["ml_score"]
        else:
            result["combined_score"] = rule_result["score"]

        # Boost if NPPES linked
        if result["has_nppes"]:
            result["combined_score"] = max(result["combined_score"], 0.9)

        result["is_physician"] = result["combined_score"] >= self.threshold

        return result


def build_physician_labels(
    output_path: Optional[Path] = None,
    use_ml: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Build physician labels for all DIME donors.

    Creates a parquet file with probability scores and labels
    from multiple classification methods.

    Args:
        output_path: Output parquet file path
        use_ml: Whether to train and use ML classifier
        overwrite: Whether to overwrite existing file

    Returns:
        Path to output file
    """
    output_path = output_path or PHYSICIAN_LABELS_PARQUET

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return output_path

    print("Building physician labels...")

    con = duckdb.connect()
    dime_path = str(DIME_DONORS_PARQUET).replace("\\", "/")

    # Load donors
    donors_df = con.execute(f"""
        SELECT
            bonica_cid,
            contributor_name,
            contributor_occupation,
            contributor_employer,
            contributor_state,
            contributor_gender,
            contributor_cfscore
        FROM read_parquet('{dime_path}')
    """).fetchdf()

    print(f"Loaded {len(donors_df):,} donors")

    # Load linkage results if available
    linkage_df = None
    if LINKAGE_RESULTS_PARQUET.exists():
        linkage_path = str(LINKAGE_RESULTS_PARQUET).replace("\\", "/")
        linkage_df = con.execute(f"""
            SELECT bonica_cid, nppes_npi
            FROM read_parquet('{linkage_path}')
        """).fetchdf()
        print(f"Loaded {len(linkage_df):,} linkage candidates")

    con.close()

    # Initialize classifier
    classifier = PhysicianClassifier(use_ml=use_ml)
    rule_classifier = classifier.rule_classifier

    # Train if using ML
    if use_ml:
        sample_df = donors_df.sample(n=min(100000, len(donors_df)), random_state=42)
        if linkage_df is not None:
            sample_linkage = linkage_df[linkage_df["bonica_cid"].isin(sample_df["bonica_cid"])]
        else:
            sample_linkage = None
        classifier.fit(sample_df, sample_linkage)

    # Classify all donors
    print("Classifying all donors...")

    # Pre-compute NPPES linked set for O(1) lookup instead of O(n)
    nppes_linked_set = set()
    if linkage_df is not None:
        nppes_linked_set = set(linkage_df["bonica_cid"].unique())
        print(f"  NPPES linked donors: {len(nppes_linked_set):,}")

    # Process using itertuples for better performance than iterrows
    results = []
    n_donors = len(donors_df)

    for i, row in enumerate(donors_df.itertuples()):
        if i % 500000 == 0:
            print(f"  Processed {i:,} / {n_donors:,}", flush=True)

        bonica_cid = row.bonica_cid
        occupation = row.contributor_occupation if pd.notna(row.contributor_occupation) else ""
        employer = row.contributor_employer if pd.notna(row.contributor_employer) else ""
        name = row.contributor_name if pd.notna(row.contributor_name) else ""

        # Naive keyword match (baseline)
        naive_match = classify_naive_keywords(occupation)

        # Rule-based classification
        rule_result = rule_classifier.classify(
            occupation=occupation,
            employer=employer,
            name=name,
        )

        # NPPES linkage - O(1) lookup using set
        has_nppes = bonica_cid in nppes_linked_set

        # Combined probability
        if use_ml and classifier.ml_model is not None:
            row_dict = {
                "bonica_cid": bonica_cid,
                "contributor_occupation": occupation,
                "contributor_employer": employer,
                "contributor_name": name,
                "nppes_npi": "linked" if has_nppes else None,
            }
            p_physician = classifier.predict_proba(row_dict)
        else:
            # Simple combination without ML
            p_physician = rule_result["score"]
            if has_nppes:
                p_physician = max(p_physician, 0.9)

        results.append({
            "bonica_cid": bonica_cid,
            "physician_naive": naive_match,
            "physician_rule_score": rule_result["score"],
            "physician_rule_tier": rule_result["tier"],
            "physician_rule_label": rule_result["is_physician"],
            "physician_nppes_linked": has_nppes,
            "p_physician": p_physician,
            "physician_final": p_physician >= classifier.threshold,
        })

    print(f"  Processed {n_donors:,} / {n_donors:,}", flush=True)
    results_df = pd.DataFrame(results)

    # Save to parquet
    results_df.to_parquet(output_path, compression="zstd", index=False)

    print(f"Created: {output_path}")

    # Print summary
    print("\nPhysician classification summary:")
    print(f"  Naive keyword matches: {results_df['physician_naive'].sum():,}")
    print(f"  Rule-based physicians: {results_df['physician_rule_label'].sum():,}")
    print(f"  NPPES-linked: {results_df['physician_nppes_linked'].sum():,}")
    print(f"  Final physicians (p >= {classifier.threshold}): {results_df['physician_final'].sum():,}")

    return output_path


def get_physician_label_stats() -> dict:
    """Get statistics from the physician labels file."""
    if not PHYSICIAN_LABELS_PARQUET.exists():
        return {}

    con = duckdb.connect()
    path = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")

    stats = {}
    stats["n_total"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_naive"] = con.execute(
        f"SELECT SUM(CASE WHEN physician_naive THEN 1 ELSE 0 END) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_rule"] = con.execute(
        f"SELECT SUM(CASE WHEN physician_rule_label THEN 1 ELSE 0 END) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_nppes"] = con.execute(
        f"SELECT SUM(CASE WHEN physician_nppes_linked THEN 1 ELSE 0 END) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_final"] = con.execute(
        f"SELECT SUM(CASE WHEN physician_final THEN 1 ELSE 0 END) FROM read_parquet('{path}')"
    ).fetchone()[0]

    # Distribution of p_physician
    stats["p_physician_mean"] = con.execute(
        f"SELECT AVG(p_physician) FROM read_parquet('{path}')"
    ).fetchone()[0]

    con.close()
    return stats

