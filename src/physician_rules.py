"""
Rule-based physician classification from occupation/employer text.

Implements a multi-tier classification system using regex patterns
to identify physicians from DIME occupation and employer fields.
"""

import re
from typing import Optional

from .config import get_keyword_rules_config


class PhysicianRuleClassifier:
    """
    Rule-based classifier for identifying physicians from text fields.

    Uses a three-tier system:
    - Tier 1: High-confidence positive indicators (credentials, specialties)
    - Tier 2: Likely positive indicators (physician, medical context)
    - Tier 3: Exclusion patterns (PhDs, non-MD healthcare, etc.)
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the classifier.

        Args:
            config: Optional keyword rules config. If None, loads from file.
        """
        self.config = config or get_keyword_rules_config()
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns from configuration."""
        # Tier 1: High-confidence positive patterns
        self.tier1_patterns: list[re.Pattern] = []

        tier1 = self.config.get("tier1_positive", {})
        for category in ["credentials", "specialties"]:
            for pattern in tier1.get(category, []):
                try:
                    self.tier1_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    print(f"Warning: Invalid regex pattern '{pattern}': {e}")

        # Tier 2: Likely positive patterns
        self.tier2_occupation_patterns: list[re.Pattern] = []
        self.tier2_employer_patterns: list[re.Pattern] = []

        tier2 = self.config.get("tier2_positive", {})
        for pattern in tier2.get("occupation_terms", []):
            try:
                self.tier2_occupation_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern}': {e}")

        for pattern in tier2.get("employer_contexts", []):
            try:
                self.tier2_employer_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern}': {e}")

        # Tier 3: Exclusion patterns
        self.tier3_patterns: list[re.Pattern] = []

        tier3 = self.config.get("tier3_negative", {})
        for category in tier3.keys():
            for pattern in tier3.get(category, []):
                try:
                    self.tier3_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    print(f"Warning: Invalid regex pattern '{pattern}': {e}")

        # Name credential patterns
        self.name_positive_patterns: list[re.Pattern] = []
        self.name_negative_patterns: list[re.Pattern] = []

        name_patterns = self.config.get("name_credential_patterns", {})
        for pattern in name_patterns.get("positive", []):
            try:
                self.name_positive_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass
        for pattern in name_patterns.get("negative", []):
            try:
                self.name_negative_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass

        # Score assignments
        self.scores = self.config.get("scores", {
            "tier1_match": 1.0,
            "tier2_match": 0.7,
            "tier2_with_employer": 0.85,
            "tier3_exclusion": 0.0,
            "no_match": 0.3,
        })

    def _check_tier3_exclusions(
        self,
        occupation: str,
        employer: str,
        name: str,
    ) -> bool:
        """
        Check if any Tier 3 exclusion patterns match.

        Returns:
            True if an exclusion pattern matches (should NOT classify as physician)
        """
        text = f"{occupation} {employer}".lower()

        for pattern in self.tier3_patterns:
            if pattern.search(text):
                return True

        # Check name for non-MD credentials
        if name:
            for pattern in self.name_negative_patterns:
                if pattern.search(name):
                    return True

        return False

    def _check_tier1_positive(
        self,
        occupation: str,
        employer: str,
        name: str,
    ) -> bool:
        """
        Check if any Tier 1 high-confidence patterns match.

        Returns:
            True if a Tier 1 pattern matches
        """
        text = f"{occupation} {employer}".lower()

        for pattern in self.tier1_patterns:
            if pattern.search(text):
                return True

        # Check name for MD/DO credentials
        if name:
            for pattern in self.name_positive_patterns:
                if pattern.search(name):
                    return True

        return False

    def _check_tier2_positive(
        self,
        occupation: str,
        employer: str,
    ) -> tuple[bool, bool]:
        """
        Check if Tier 2 patterns match.

        Returns:
            Tuple of (occupation_match, employer_context_match)
        """
        occupation_match = False
        employer_match = False

        for pattern in self.tier2_occupation_patterns:
            if pattern.search(occupation.lower()):
                occupation_match = True
                break

        for pattern in self.tier2_employer_patterns:
            if pattern.search(employer.lower()):
                employer_match = True
                break

        return occupation_match, employer_match

    def classify(
        self,
        occupation: Optional[str] = None,
        employer: Optional[str] = None,
        name: Optional[str] = None,
    ) -> dict:
        """
        Classify whether a donor is likely a physician based on text fields.

        Args:
            occupation: Occupation field from DIME
            employer: Employer field from DIME
            name: Contributor name (may contain credentials)

        Returns:
            Dictionary with classification results:
                - score: float in [0, 1] indicating physician probability
                - tier: str indicating which tier matched
                - matched_patterns: list of pattern descriptions that matched
                - is_physician: bool hard classification (score >= 0.5)
        """
        occupation = occupation or ""
        employer = employer or ""
        name = name or ""

        # Check minimum lengths
        min_occ_len = self.config.get("min_occupation_length", 3)
        if len(occupation.strip()) < min_occ_len and len(employer.strip()) < min_occ_len:
            return {
                "score": self.scores.get("no_match", 0.3),
                "tier": "insufficient_data",
                "matched_patterns": [],
                "is_physician": False,
            }

        # Step 1: Check Tier 3 exclusions first
        if self._check_tier3_exclusions(occupation, employer, name):
            return {
                "score": self.scores.get("tier3_exclusion", 0.0),
                "tier": "tier3_exclusion",
                "matched_patterns": ["exclusion_pattern"],
                "is_physician": False,
            }

        # Step 2: Check Tier 1 high-confidence patterns
        if self._check_tier1_positive(occupation, employer, name):
            return {
                "score": self.scores.get("tier1_match", 1.0),
                "tier": "tier1",
                "matched_patterns": ["tier1_credential_or_specialty"],
                "is_physician": True,
            }

        # Step 3: Check Tier 2 patterns
        occ_match, emp_match = self._check_tier2_positive(occupation, employer)

        if occ_match and emp_match:
            return {
                "score": self.scores.get("tier2_with_employer", 0.85),
                "tier": "tier2_with_context",
                "matched_patterns": ["tier2_occupation", "tier2_employer_context"],
                "is_physician": True,
            }
        elif occ_match:
            return {
                "score": self.scores.get("tier2_match", 0.7),
                "tier": "tier2",
                "matched_patterns": ["tier2_occupation"],
                "is_physician": True,
            }

        # No match
        return {
            "score": self.scores.get("no_match", 0.3),
            "tier": "no_match",
            "matched_patterns": [],
            "is_physician": False,
        }

    def classify_batch(
        self,
        records: list[dict],
    ) -> list[dict]:
        """
        Classify multiple records.

        Args:
            records: List of dicts with 'occupation', 'employer', and optionally 'name'

        Returns:
            List of classification results
        """
        return [
            self.classify(
                occupation=r.get("occupation"),
                employer=r.get("employer"),
                name=r.get("name"),
            )
            for r in records
        ]


def classify_naive_keywords(occupation: Optional[str]) -> bool:
    """
    Naive keyword classification matching the original script.

    Uses simple 'doctor|physician' regex for baseline comparison.

    Args:
        occupation: Occupation string

    Returns:
        True if matches naive keyword pattern
    """
    if not occupation:
        return False

    pattern = r"doctor|physician"
    return bool(re.search(pattern, occupation, re.IGNORECASE))


def get_rule_classifier() -> PhysicianRuleClassifier:
    """Get a configured PhysicianRuleClassifier instance."""
    return PhysicianRuleClassifier()
