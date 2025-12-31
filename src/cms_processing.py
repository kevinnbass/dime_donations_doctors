"""
CMS Medicare Physician PUF data processing.

Ingests and processes the Medicare Physician and Other Supplier PUF
for identifying active Medicare billing physicians.
"""

from pathlib import Path
from typing import Optional

import duckdb

from .config import (
    CMS_RAW_DIR,
    CMS_MEDICARE_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    DEFAULT_CMS_MIN_BENEFICIARIES,
)


def find_cms_provider_file() -> Path:
    """
    Locate the CMS Provider PUF file.

    Returns:
        Path to the CMS provider-level PUF file

    Raises:
        FileNotFoundError: If no CMS file is found
    """
    # Look for the provider-level file (not the service-level one)
    patterns = [
        "MUP_PHY_*_Prov.csv",
        "*_Prov.csv",
    ]

    for pattern in patterns:
        matches = list(CMS_RAW_DIR.glob(pattern))
        if matches:
            # Return the most recent one if multiple
            return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        f"No CMS Provider PUF file found in {CMS_RAW_DIR}. "
        "Expected file matching pattern: MUP_PHY_*_Prov.csv"
    )


def ingest_cms_medicare_physicians(
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    min_beneficiaries: int = DEFAULT_CMS_MIN_BENEFICIARIES,
) -> Path:
    """
    Ingest CMS Medicare PUF and filter to active physicians.

    Extracts individual providers with minimum beneficiary count,
    along with key utilization metrics.

    Args:
        output_path: Output parquet file path
        overwrite: Whether to overwrite existing file
        min_beneficiaries: Minimum beneficiaries (default: 11, CMS threshold)

    Returns:
        Path to the output parquet file
    """
    output_path = output_path or CMS_MEDICARE_PARQUET

    if output_path.exists() and not overwrite:
        print(f"CMS Medicare parquet already exists: {output_path}")
        return output_path

    cms_file = find_cms_provider_file()
    print(f"Ingesting CMS Medicare PUF: {cms_file.name}")

    con = duckdb.connect()

    cms_path = str(cms_file).replace("\\", "/")
    output_path_str = str(output_path).replace("\\", "/")

    # Count total records
    total = con.execute(f"""
        SELECT COUNT(*) FROM read_csv_auto('{cms_path}', header=true)
    """).fetchone()[0]
    print(f"  Total CMS records: {total:,}")

    # Ingest with filtering
    query = f"""
    COPY (
        SELECT
            Rndrng_NPI as cms_npi,
            Rndrng_Prvdr_Last_Org_Name as cms_last_name,
            Rndrng_Prvdr_First_Name as cms_first_name,
            Rndrng_Prvdr_Crdntls as cms_credentials,
            Rndrng_Prvdr_Ent_Cd as cms_entity_code,
            Rndrng_Prvdr_State_Abrvtn as cms_state,
            Rndrng_Prvdr_Zip5 as cms_zip5,
            Rndrng_Prvdr_Type as cms_specialty,
            CAST(Tot_Benes AS INTEGER) as cms_total_beneficiaries,
            CAST(Tot_Srvcs AS INTEGER) as cms_total_services,
            CAST(Tot_Mdcr_Pymt_Amt AS DOUBLE) as cms_total_payment
        FROM read_csv_auto('{cms_path}', header=true)
        WHERE Rndrng_Prvdr_Ent_Cd = 'I'  -- Individual providers only
          AND CAST(Tot_Benes AS INTEGER) >= {min_beneficiaries}
    ) TO '{output_path_str}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(query)

    # Get summary statistics
    stats = con.execute(f"""
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT cms_npi) as unique_npis,
            COUNT(DISTINCT cms_specialty) as specialties
        FROM read_parquet('{output_path_str}')
    """).fetchone()

    con.close()

    print(f"Created: {output_path}")
    print(f"  Individual providers: {stats[0]:,}")
    print(f"  Unique NPIs: {stats[1]:,}")
    print(f"  Specialties: {stats[2]}")

    return output_path


def link_nppes_to_cms(
    linkage_path: Optional[Path] = None,
    cms_path: Optional[Path] = None,
) -> dict:
    """
    Link NPPES-matched donors to CMS Medicare data.

    Uses NPI as the join key between NPPES linkage results and CMS Medicare PUF.

    Args:
        linkage_path: Path to NPPES linkage results
        cms_path: Path to processed CMS parquet

    Returns:
        Dictionary with linkage statistics
    """
    linkage_path = linkage_path or LINKAGE_RESULTS_PARQUET
    cms_path = cms_path or CMS_MEDICARE_PARQUET

    if not linkage_path.exists():
        raise FileNotFoundError(f"NPPES linkage results not found: {linkage_path}")
    if not cms_path.exists():
        raise FileNotFoundError(f"CMS Medicare parquet not found: {cms_path}")

    con = duckdb.connect()

    linkage_p = str(linkage_path).replace("\\", "/")
    cms_p = str(cms_path).replace("\\", "/")

    # Count linkage records with NPPES NPI
    n_linkage = con.execute(f"""
        SELECT COUNT(DISTINCT nppes_npi)
        FROM read_parquet('{linkage_p}')
        WHERE nppes_npi IS NOT NULL
    """).fetchone()[0]

    # Count CMS records
    n_cms = con.execute(f"""
        SELECT COUNT(DISTINCT cms_npi)
        FROM read_parquet('{cms_p}')
    """).fetchone()[0]

    # Find overlap
    n_overlap = con.execute(f"""
        SELECT COUNT(DISTINCT l.nppes_npi)
        FROM read_parquet('{linkage_p}') l
        INNER JOIN read_parquet('{cms_p}') c
            ON CAST(l.nppes_npi AS VARCHAR) = CAST(c.cms_npi AS VARCHAR)
    """).fetchone()[0]

    con.close()

    stats = {
        "nppes_linked_donors_with_npi": n_linkage,
        "cms_individual_providers": n_cms,
        "overlap_npis": n_overlap,
        "overlap_rate_of_linkage": n_overlap / n_linkage if n_linkage > 0 else 0,
        "overlap_rate_of_cms": n_overlap / n_cms if n_cms > 0 else 0,
    }

    print(f"NPPES-to-CMS linkage:")
    print(f"  NPPES-linked donors with NPI: {stats['nppes_linked_donors_with_npi']:,}")
    print(f"  CMS individual providers: {stats['cms_individual_providers']:,}")
    print(f"  Overlap (NPIs in both): {stats['overlap_npis']:,}")
    print(f"  Overlap rate (of linkage): {stats['overlap_rate_of_linkage']:.1%}")

    return stats


def add_cms_columns_to_labels(
    labels_path: Optional[Path] = None,
    linkage_path: Optional[Path] = None,
    cms_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Add CMS Medicare columns to physician labels.

    Joins physician labels with NPPES linkage and CMS data to add:
    - cms_medicare_active: Whether the donor is in CMS Medicare PUF
    - cms_specialty: Medicare specialty from CMS
    - cms_total_beneficiaries: Number of Medicare beneficiaries
    - cms_total_services: Number of Medicare services
    - cms_total_payment: Total Medicare payment

    Args:
        labels_path: Path to physician labels parquet
        linkage_path: Path to NPPES linkage results
        cms_path: Path to processed CMS parquet
        output_path: Output path (defaults to overwriting labels)
        overwrite: Whether to overwrite existing file

    Returns:
        Path to output file
    """
    labels_path = labels_path or PHYSICIAN_LABELS_PARQUET
    linkage_path = linkage_path or LINKAGE_RESULTS_PARQUET
    cms_path = cms_path or CMS_MEDICARE_PARQUET
    output_path = output_path or labels_path

    if not labels_path.exists():
        raise FileNotFoundError(f"Physician labels not found: {labels_path}")

    # If CMS or linkage doesn't exist, skip
    if not cms_path.exists():
        print(f"CMS Medicare parquet not found, skipping CMS columns")
        return labels_path
    if not linkage_path.exists():
        print(f"NPPES linkage not found, skipping CMS columns")
        return labels_path

    con = duckdb.connect()

    labels_p = str(labels_path).replace("\\", "/")
    linkage_p = str(linkage_path).replace("\\", "/")
    cms_p = str(cms_path).replace("\\", "/")

    # Use a temp file to avoid overwriting while reading
    temp_path = str(output_path).replace(".parquet", "_temp.parquet").replace("\\", "/")

    print("Adding CMS columns to physician labels...")

    query = f"""
    COPY (
        WITH linkage_best AS (
            -- Get best NPPES match per donor
            SELECT DISTINCT ON (bonica_cid)
                bonica_cid,
                nppes_npi
            FROM read_parquet('{linkage_p}')
            ORDER BY bonica_cid
        ),
        cms_lookup AS (
            SELECT DISTINCT
                cms_npi,
                cms_specialty,
                cms_total_beneficiaries,
                cms_total_services,
                cms_total_payment
            FROM read_parquet('{cms_p}')
        )
        SELECT
            l.*,
            CASE WHEN c.cms_npi IS NOT NULL THEN true ELSE false END as cms_medicare_active,
            c.cms_specialty,
            c.cms_total_beneficiaries,
            c.cms_total_services,
            c.cms_total_payment
        FROM read_parquet('{labels_p}') l
        LEFT JOIN linkage_best lb ON l.bonica_cid = lb.bonica_cid
        LEFT JOIN cms_lookup c ON CAST(lb.nppes_npi AS VARCHAR) = CAST(c.cms_npi AS VARCHAR)
    ) TO '{temp_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(query)

    # Get statistics
    stats = con.execute(f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN cms_medicare_active THEN 1 ELSE 0 END) as medicare_active,
            SUM(CASE WHEN physician_final AND cms_medicare_active THEN 1 ELSE 0 END) as physicians_medicare_active
        FROM read_parquet('{temp_path}')
    """).fetchone()

    con.close()

    # Replace original file
    import os
    import shutil
    temp_file = Path(temp_path.replace("/", "\\"))
    if output_path.exists():
        os.remove(output_path)
    shutil.move(str(temp_file), str(output_path))

    print(f"Created: {output_path}")
    print(f"  Total donors: {stats[0]:,}")
    print(f"  Medicare-active (any): {stats[1]:,}")
    print(f"  Physicians who are Medicare-active: {stats[2]:,}")

    return output_path


def get_cms_specialty_mapping() -> dict:
    """
    Map CMS specialty names to standardized specialty groups.

    Returns mapping from Rndrng_Prvdr_Type to specialty_group
    compatible with taxonomy_physician.yaml specialty_groups.
    """
    # This maps CMS Medicare specialty names to broader groups
    return {
        # Primary Care
        "Internal Medicine": "primary_care",
        "Family Practice": "primary_care",
        "General Practice": "primary_care",
        "Geriatric Medicine": "primary_care",
        "Pediatric Medicine": "primary_care",

        # Surgical
        "General Surgery": "surgical",
        "Orthopedic Surgery": "surgical",
        "Thoracic Surgery": "surgical",
        "Cardiac Surgery": "surgical",
        "Vascular Surgery": "surgical",
        "Plastic and Reconstructive Surgery": "surgical",
        "Colorectal Surgery": "surgical",
        "Hand Surgery": "surgical",

        # Medical Specialties
        "Cardiology": "medical_specialty",
        "Gastroenterology": "medical_specialty",
        "Pulmonary Disease": "medical_specialty",
        "Nephrology": "medical_specialty",
        "Rheumatology": "medical_specialty",
        "Endocrinology": "medical_specialty",
        "Hematology": "medical_specialty",
        "Hematology/Oncology": "medical_specialty",
        "Infectious Disease": "medical_specialty",
        "Allergy/Immunology": "medical_specialty",

        # Hospital-Based
        "Anesthesiology": "hospital_based",
        "Emergency Medicine": "hospital_based",
        "Critical Care (Intensivists)": "hospital_based",
        "Hospitalist": "hospital_based",
        "Pathology": "hospital_based",
        "Diagnostic Radiology": "hospital_based",
        "Interventional Radiology": "hospital_based",

        # Psychiatry/Neurology
        "Psychiatry": "psychiatry_neurology",
        "Neurology": "psychiatry_neurology",
        "Neuropsychiatry": "psychiatry_neurology",

        # OB/GYN
        "Obstetrics/Gynecology": "obgyn",
        "Gynecological/Oncology": "obgyn",

        # Ophthalmology
        "Ophthalmology": "ophthalmology",

        # Dermatology
        "Dermatology": "dermatology",

        # Other
        "Physical Medicine and Rehabilitation": "other",
        "Preventive Medicine": "other",
        "Sports Medicine": "other",
        "Pain Management": "other",
        "Radiation Oncology": "other",
        "Medical Oncology": "other",
        "Nuclear Medicine": "other",
    }


def get_cms_stats() -> dict:
    """Get statistics from the processed CMS data."""
    if not CMS_MEDICARE_PARQUET.exists():
        return {}

    con = duckdb.connect()
    path = str(CMS_MEDICARE_PARQUET).replace("\\", "/")

    stats = {}
    stats["n_total"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_unique_npis"] = con.execute(
        f"SELECT COUNT(DISTINCT cms_npi) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_specialties"] = con.execute(
        f"SELECT COUNT(DISTINCT cms_specialty) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["avg_beneficiaries"] = con.execute(
        f"SELECT AVG(cms_total_beneficiaries) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["total_payment"] = con.execute(
        f"SELECT SUM(cms_total_payment) FROM read_parquet('{path}')"
    ).fetchone()[0]

    # Top specialties
    top_specialties = con.execute(f"""
        SELECT cms_specialty, COUNT(*) as n
        FROM read_parquet('{path}')
        GROUP BY cms_specialty
        ORDER BY n DESC
        LIMIT 10
    """).fetchdf()

    stats["top_specialties"] = top_specialties.to_dict("records")

    con.close()
    return stats
