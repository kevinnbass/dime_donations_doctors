"""
Configuration management for the DIME physician analysis pipeline.

Handles paths, constants, and loading of YAML configuration files.
"""

from pathlib import Path
from typing import Any

import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw data subdirectories
DIME_RAW_DIR = RAW_DATA_DIR / "dime"
NPPES_RAW_DIR = RAW_DATA_DIR / "nppes"
NUCC_RAW_DIR = RAW_DATA_DIR / "nucc"
AHRF_RAW_DIR = RAW_DATA_DIR / "hrsa"
CMS_RAW_DIR = RAW_DATA_DIR / "cms"
PECOS_RAW_DIR = RAW_DATA_DIR / "pecos"
BONICA_RAW_DIR = RAW_DATA_DIR / "medical_professionals" / "brr_jama-im" / "data"

# Config directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TABLES_DIR = OUTPUTS_DIR / "tables"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"

# Runs directory (for audit logging and reproducibility)
RUNS_DIR = PROJECT_ROOT / "runs"

# Processed data files
NPPES_PHYSICIANS_PARQUET = PROCESSED_DATA_DIR / "physicians_nppes.parquet"
DIME_DONORS_PARQUET = PROCESSED_DATA_DIR / "dime_donors.parquet"
DIME_RECIPIENTS_PARQUET = PROCESSED_DATA_DIR / "dime_recipients.parquet"
LINKAGE_RESULTS_PARQUET = PROCESSED_DATA_DIR / "linkage_results.parquet"
PHYSICIAN_LABELS_PARQUET = PROCESSED_DATA_DIR / "physician_labels.parquet"
DONOR_CYCLE_PANEL_PARQUET = PROCESSED_DATA_DIR / "donor_cycle_panel.parquet"
CYCLE_IDEOLOGY_PARQUET = PROCESSED_DATA_DIR / "cycle_ideology.parquet"

# CMS Medicare processed data
CMS_MEDICARE_PARQUET = PROCESSED_DATA_DIR / "cms_medicare_physicians.parquet"

# PECOS (CMS Provider Enrollment) processed data
PECOS_PHYSICIANS_PARQUET = PROCESSED_DATA_DIR / "pecos_physicians.parquet"

# AHRF state-level physician counts
AHRF_STATE_PARQUET = PROCESSED_DATA_DIR / "ahrf_state_physicians.parquet"

# Bonica reference data
BONICA_REFERENCE_PARQUET = PROCESSED_DATA_DIR / "bonica_reference.parquet"

# Probabilistic linkage data
LINKAGE_TRAINING_DIR = PROCESSED_DATA_DIR / "linkage_training"
LINKAGE_TRAINING_PAIRS_PARQUET = LINKAGE_TRAINING_DIR / "training_pairs.parquet"
NAME_FREQUENCIES_PARQUET = LINKAGE_TRAINING_DIR / "name_frequencies.parquet"
PROBABILISTIC_LINKAGE_PARQUET = PROCESSED_DATA_DIR / "probabilistic_linkage.parquet"
PHYSICIAN_EXPECTED_P_TO_REP_PARQUET = PROCESSED_DATA_DIR / "physician_expected_p_to_rep.parquet"
FIGURE1_EXTENDED_PARQUET = PROCESSED_DATA_DIR / "figure1_data_extended.parquet"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"
LINKAGE_MODEL_PATH = MODELS_DIR / "linkage_model.pkl"
LINKAGE_CALIBRATOR_PATH = MODELS_DIR / "linkage_calibrator.pkl"

# Temporary download directory
DIME_TEMP_DIR = DIME_RAW_DIR / "temp"

# Streaming ingestion paths
DIME_ITEMIZED_TMP_DIR = DIME_RAW_DIR / "itemized_tmp"
DIME_ITEMIZED_MANIFEST = DIME_RAW_DIR / "itemized_manifest.csv"
DONOR_CYCLE_PANEL_PARTITIONED_DIR = PROCESSED_DATA_DIR / "parquet" / "donor_cycle_panel"

# Cycle metadata (extracted from itemized)
DONOR_CYCLE_METADATA_PARQUET = PROCESSED_DATA_DIR / "donor_cycle_metadata.parquet"

# Precision evaluation files
LIKELY_NEGATIVES_PARQUET = PROCESSED_DATA_DIR / "likely_negatives.parquet"

# Election cycles
FIRST_CYCLE = 1980
LAST_CYCLE = 2024
CYCLES = list(range(FIRST_CYCLE, LAST_CYCLE + 1, 2))

# Default thresholds
DEFAULT_MIN_CYCLE_AMOUNT = 200  # Minimum total donation in a cycle for cycle-specific score
DEFAULT_MIN_RECIPIENTS = 3  # Minimum recipients for unshrunk cycle score
DEFAULT_PHYSICIAN_THRESHOLD = 0.5  # Probability threshold for physician classification
DEFAULT_LINKAGE_THRESHOLD = 0.85  # Minimum score for NPPES linkage match

# CMS Medicare thresholds
DEFAULT_CMS_MIN_BENEFICIARIES = 11  # CMS suppression threshold
DEFAULT_CMS_MIN_SERVICES = 25  # Minimum services for "active" clinician

# AHRF configuration
AHRF_REFERENCE_YEAR = 2023  # Most recent year in AHRF 2025 data
AHRF_COUNT_SUFFIX = "23"  # Suffix for physician count columns (e.g., md_nf_activ_23)

# Shrinkage parameters for cycle-specific ideology
SHRINKAGE_WEIGHT_SINGLE_RECIPIENT = 0.3  # Weight toward 0 for single-recipient donors
SHRINKAGE_WEIGHT_FEW_RECIPIENTS = 0.7  # Weight toward 0 for 2 recipients


def load_yaml_config(config_name: str) -> dict[str, Any]:
    """
    Load a YAML configuration file from the config directory.

    Args:
        config_name: Name of the config file (with or without .yaml extension)

    Returns:
        Dictionary containing the configuration
    """
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"

    config_path = CONFIG_DIR / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_taxonomy_config() -> dict[str, Any]:
    """Load the physician taxonomy configuration."""
    return load_yaml_config("taxonomy_physician")


def get_keyword_rules_config() -> dict[str, Any]:
    """Load the keyword rules configuration."""
    return load_yaml_config("keyword_rules")


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    dirs = [
        RAW_DATA_DIR,
        DIME_RAW_DIR,
        NPPES_RAW_DIR,
        NUCC_RAW_DIR,
        AHRF_RAW_DIR,
        CMS_RAW_DIR,
        PECOS_RAW_DIR,
        PROCESSED_DATA_DIR,
        PLOTS_DIR,
        TABLES_DIR,
        DIAGNOSTICS_DIR,
        DIME_ITEMIZED_TMP_DIR,
        DONOR_CYCLE_PANEL_PARTITIONED_DIR,
        LINKAGE_TRAINING_DIR,
        MODELS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
