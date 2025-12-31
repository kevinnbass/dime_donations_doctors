"""
NUCC Healthcare Provider Taxonomy handling.

Parses taxonomy codes and determines physician classification based on
the NUCC Healthcare Provider Taxonomy Code Set.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import NUCC_RAW_DIR, get_taxonomy_config


class TaxonomyClassifier:
    """
    Classifies healthcare providers based on NUCC taxonomy codes.

    Uses configuration from taxonomy_physician.yaml to determine which
    taxonomy codes indicate a physician.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the taxonomy classifier.

        Args:
            config: Optional taxonomy configuration dict. If None, loads from file.
        """
        self.config = config or get_taxonomy_config()
        self._build_lookup_tables()

    def _build_lookup_tables(self) -> None:
        """Build efficient lookup structures for taxonomy classification."""
        # Set of physician taxonomy codes (full 10-character codes)
        self.physician_codes: set[str] = set()
        for entry in self.config.get("physician_codes", []):
            self.physician_codes.add(entry["code"])

        # Set of physician code prefixes (for grouping-level matching)
        self.physician_prefixes: set[str] = set()
        for name, grouping in self.config.get("physician_groupings", {}).items():
            if grouping.get("include", False):
                self.physician_prefixes.add(grouping["code_prefix"])

        # Set of excluded prefixes
        self.excluded_prefixes: set[str] = set()
        for entry in self.config.get("excluded_codes", []):
            self.excluded_prefixes.add(entry["code_prefix"])

        # Build specialty lookup (code -> specialty name)
        self.code_to_specialty: dict[str, str] = {}
        for entry in self.config.get("physician_codes", []):
            self.code_to_specialty[entry["code"]] = entry["specialty"]

        # Build specialty groupings
        self.specialty_groups: dict[str, list[str]] = self.config.get("specialty_groups", {})

        # Reverse lookup: specialty name -> group
        self.specialty_to_group: dict[str, str] = {}
        for group_name, specialties in self.specialty_groups.items():
            for specialty in specialties:
                self.specialty_to_group[specialty.lower()] = group_name

    def is_physician_code(self, taxonomy_code: str) -> bool:
        """
        Check if a taxonomy code indicates a physician.

        Args:
            taxonomy_code: 10-character NUCC taxonomy code

        Returns:
            True if the code indicates a physician (MD/DO), False otherwise
        """
        if not taxonomy_code or not isinstance(taxonomy_code, str):
            return False

        code = taxonomy_code.strip().upper()

        # Check for explicit exclusions first
        for prefix in self.excluded_prefixes:
            if code.startswith(prefix):
                return False

        # Check for exact code match
        if code in self.physician_codes:
            return True

        # Check for prefix match (Allopathic & Osteopathic Physicians = "20")
        for prefix in self.physician_prefixes:
            if code.startswith(prefix):
                return True

        return False

    def get_specialty(self, taxonomy_code: str) -> Optional[str]:
        """
        Get the specialty name for a taxonomy code.

        Args:
            taxonomy_code: 10-character NUCC taxonomy code

        Returns:
            Specialty name if known, None otherwise
        """
        if not taxonomy_code:
            return None
        return self.code_to_specialty.get(taxonomy_code.strip().upper())

    def get_specialty_group(self, specialty_or_code: str) -> Optional[str]:
        """
        Get the specialty group (e.g., 'primary_care', 'surgical') for a
        specialty name or taxonomy code.

        Args:
            specialty_or_code: Either a specialty name or taxonomy code

        Returns:
            Specialty group name if found, None otherwise
        """
        if not specialty_or_code:
            return None

        # Try direct specialty name lookup
        group = self.specialty_to_group.get(specialty_or_code.lower())
        if group:
            return group

        # Try as taxonomy code
        specialty = self.get_specialty(specialty_or_code)
        if specialty:
            return self.specialty_to_group.get(specialty.lower())

        return None

    def classify_codes(self, taxonomy_codes: list[str]) -> dict:
        """
        Classify a list of taxonomy codes for a single provider.

        Args:
            taxonomy_codes: List of taxonomy codes for a provider

        Returns:
            Dictionary with classification results:
                - is_physician: bool
                - primary_specialty: str or None
                - specialty_group: str or None
                - all_specialties: list[str]
        """
        if not taxonomy_codes:
            return {
                "is_physician": False,
                "primary_specialty": None,
                "specialty_group": None,
                "all_specialties": [],
            }

        physician_codes = []
        specialties = []

        for code in taxonomy_codes:
            if self.is_physician_code(code):
                physician_codes.append(code)
                specialty = self.get_specialty(code)
                if specialty:
                    specialties.append(specialty)

        if not physician_codes:
            return {
                "is_physician": False,
                "primary_specialty": None,
                "specialty_group": None,
                "all_specialties": [],
            }

        # Use first physician specialty as primary
        primary_specialty = specialties[0] if specialties else None
        specialty_group = self.get_specialty_group(primary_specialty) if primary_specialty else None

        return {
            "is_physician": True,
            "primary_specialty": primary_specialty,
            "specialty_group": specialty_group,
            "all_specialties": specialties,
        }


def load_nucc_taxonomy(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the NUCC taxonomy code set from CSV.

    Args:
        file_path: Path to NUCC taxonomy CSV. If None, searches in NUCC_RAW_DIR.

    Returns:
        DataFrame with taxonomy codes and descriptions
    """
    if file_path is None:
        # Look for common NUCC file names
        possible_names = [
            "nucc_taxonomy.csv",
            "nucc_taxonomy_*.csv",
            "taxonomy.csv",
        ]
        for name in possible_names:
            matches = list(NUCC_RAW_DIR.glob(name))
            if matches:
                file_path = matches[0]
                break

    if file_path is None or not file_path.exists():
        raise FileNotFoundError(
            f"NUCC taxonomy file not found in {NUCC_RAW_DIR}. "
            "Please download from https://www.nucc.org/index.php/code-sets-mainmenu-41"
        )

    df = pd.read_csv(file_path, dtype=str)

    # Standardize column names (NUCC uses various formats)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    return df
