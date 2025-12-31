"""
NPPES (National Plan and Provider Enumeration System) data processing.

Handles ingestion and filtering of the NPI registry to extract physicians.
"""

from pathlib import Path
from typing import Optional

import duckdb

from .config import (
    NPPES_PHYSICIANS_PARQUET,
    NPPES_RAW_DIR,
    PROCESSED_DATA_DIR,
)
from .taxonomy import TaxonomyClassifier
from .text_normalization import (
    extract_name_parts,
    normalize_city,
    normalize_name,
    normalize_state,
    normalize_zip,
    soundex,
)


def find_nppes_file() -> Path:
    """
    Locate the NPPES NPI data file.

    Searches for the standard NPI dissemination file patterns.

    Returns:
        Path to the NPPES file

    Raises:
        FileNotFoundError: If no NPPES file is found
    """
    patterns = [
        "npidata_pfile_*.csv",
        "NPPES_Data_*.csv",
        "npidata*.csv",
        "*.csv",  # Fallback to any CSV in the directory
    ]

    for pattern in patterns:
        matches = list(NPPES_RAW_DIR.glob(pattern))
        if matches:
            # Return the largest file (likely the main data file)
            return max(matches, key=lambda p: p.stat().st_size)

    raise FileNotFoundError(
        f"No NPPES NPI data file found in {NPPES_RAW_DIR}. "
        "Download from: https://download.cms.gov/nppes/NPI_Files.html"
    )


def ingest_nppes_physicians(
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    taxonomy_classifier: Optional[TaxonomyClassifier] = None,
) -> Path:
    """
    Ingest NPPES data and filter to physicians only.

    Extracts Type 1 (individual) providers with physician taxonomy codes.

    Args:
        output_path: Output parquet file path
        overwrite: Whether to overwrite existing file
        taxonomy_classifier: Optional TaxonomyClassifier instance

    Returns:
        Path to the output parquet file
    """
    output_path = output_path or NPPES_PHYSICIANS_PARQUET

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return output_path

    input_file = find_nppes_file()
    print(f"Processing NPPES data from: {input_file}")

    # Initialize taxonomy classifier
    taxonomy = taxonomy_classifier or TaxonomyClassifier()

    # Build the physician taxonomy code condition
    # We'll use the prefix "20" for Allopathic & Osteopathic Physicians
    physician_prefix = "20"

    con = duckdb.connect()

    # First, create a view of the NPPES data with only needed columns
    # NPPES has many columns; we need to be selective
    #
    # Key columns:
    # - NPI
    # - Entity Type Code (1 = Individual, 2 = Organization)
    # - Provider Name fields
    # - Provider Address fields
    # - Healthcare Provider Taxonomy Code_1 through _15

    # Build taxonomy column list for all 15 NPPES taxonomy fields
    # FIX: Previously only checked columns 1-5, but NPPES has 15 taxonomy columns
    taxonomy_select_cols = ",\n            ".join([
        f'"Healthcare Provider Taxonomy Code_{i}" AS taxonomy_code_{i}'
        for i in range(1, 16)
    ])

    # Build WHERE clause to check ALL 15 taxonomy columns for physician prefix
    taxonomy_where_conditions = " OR ".join([
        f'"Healthcare Provider Taxonomy Code_{i}" LIKE \'{physician_prefix}%\''
        for i in range(1, 16)
    ])

    query = f"""
    COPY (
        SELECT
            NPI AS npi,
            "Provider First Name" AS first_name,
            "Provider Last Name (Legal Name)" AS last_name,
            "Provider Middle Name" AS middle_name,
            "Provider Name Suffix Text" AS name_suffix,
            "Provider Credential Text" AS credentials,
            "Provider First Line Business Practice Location Address" AS address_line1,
            "Provider Second Line Business Practice Location Address" AS address_line2,
            "Provider Business Practice Location Address City Name" AS city,
            "Provider Business Practice Location Address State Name" AS state,
            "Provider Business Practice Location Address Postal Code" AS zipcode,
            "Provider Sex Code" AS gender,
            {taxonomy_select_cols},
            "NPI Deactivation Date" AS deactivation_date,
            -- Track which taxonomy column(s) contain physician codes
            CASE
                WHEN "Healthcare Provider Taxonomy Code_1" LIKE '{physician_prefix}%' THEN 1
                WHEN "Healthcare Provider Taxonomy Code_2" LIKE '{physician_prefix}%' THEN 2
                WHEN "Healthcare Provider Taxonomy Code_3" LIKE '{physician_prefix}%' THEN 3
                WHEN "Healthcare Provider Taxonomy Code_4" LIKE '{physician_prefix}%' THEN 4
                WHEN "Healthcare Provider Taxonomy Code_5" LIKE '{physician_prefix}%' THEN 5
                WHEN "Healthcare Provider Taxonomy Code_6" LIKE '{physician_prefix}%' THEN 6
                WHEN "Healthcare Provider Taxonomy Code_7" LIKE '{physician_prefix}%' THEN 7
                WHEN "Healthcare Provider Taxonomy Code_8" LIKE '{physician_prefix}%' THEN 8
                WHEN "Healthcare Provider Taxonomy Code_9" LIKE '{physician_prefix}%' THEN 9
                WHEN "Healthcare Provider Taxonomy Code_10" LIKE '{physician_prefix}%' THEN 10
                WHEN "Healthcare Provider Taxonomy Code_11" LIKE '{physician_prefix}%' THEN 11
                WHEN "Healthcare Provider Taxonomy Code_12" LIKE '{physician_prefix}%' THEN 12
                WHEN "Healthcare Provider Taxonomy Code_13" LIKE '{physician_prefix}%' THEN 13
                WHEN "Healthcare Provider Taxonomy Code_14" LIKE '{physician_prefix}%' THEN 14
                WHEN "Healthcare Provider Taxonomy Code_15" LIKE '{physician_prefix}%' THEN 15
                ELSE NULL
            END AS first_physician_taxonomy_col
        FROM read_csv_auto(
            '{str(input_file).replace(chr(92), "/")}',
            header=true,
            all_varchar=true,
            sample_size=100000,
            ignore_errors=true
        )
        WHERE "Entity Type Code" = '1'
          AND "NPI Deactivation Date" IS NULL
          AND ({taxonomy_where_conditions})
    ) TO '{str(output_path).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    print("Filtering to physicians and converting to Parquet...")
    con.execute(query)
    con.close()

    print(f"Created: {output_path}")

    # Post-process to add normalized fields
    add_normalized_fields(output_path)

    return output_path


def add_normalized_fields(parquet_path: Path) -> None:
    """
    Add normalized name and address fields to the NPPES parquet file.

    Updates the file in place with additional columns for record linkage.
    """
    print("Adding normalized fields for record linkage...")

    con = duckdb.connect()

    # Read existing data
    df = con.execute(f"""
        SELECT * FROM read_parquet('{str(parquet_path).replace(chr(92), "/")}')
    """).fetchdf()

    con.close()

    # Add normalized fields
    df["first_name_norm"] = df["first_name"].apply(lambda x: normalize_name(x) if x else "")
    df["last_name_norm"] = df["last_name"].apply(lambda x: normalize_name(x) if x else "")
    df["city_norm"] = df["city"].apply(lambda x: normalize_city(x) if x else "")
    df["state_norm"] = df["state"].apply(lambda x: normalize_state(x) if x else "")
    df["zip5"] = df["zipcode"].apply(lambda x: normalize_zip(x) if x else "")
    df["last_name_soundex"] = df["last_name_norm"].apply(lambda x: soundex(x) if x else "")

    # Create full name for matching
    df["full_name_norm"] = df.apply(
        lambda row: normalize_name(
            f"{row['first_name'] or ''} {row['middle_name'] or ''} {row['last_name'] or ''}"
        ),
        axis=1,
    )

    # Write back
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path, compression="zstd")

    print(f"Updated {parquet_path} with normalized fields")


def get_nppes_physician_count() -> int:
    """Get the total number of physicians in the processed NPPES data."""
    if not NPPES_PHYSICIANS_PARQUET.exists():
        return 0

    con = duckdb.connect()
    result = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{str(NPPES_PHYSICIANS_PARQUET).replace(chr(92), "/")}')
    """).fetchone()
    con.close()
    return result[0] if result else 0


def get_nppes_state_counts() -> dict[str, int]:
    """
    Get physician counts by state from NPPES.

    Returns:
        Dictionary mapping state abbreviation to count
    """
    if not NPPES_PHYSICIANS_PARQUET.exists():
        return {}

    con = duckdb.connect()
    result = con.execute(f"""
        SELECT state_norm, COUNT(*) as count
        FROM read_parquet('{str(NPPES_PHYSICIANS_PARQUET).replace(chr(92), "/")}')
        WHERE state_norm IS NOT NULL AND state_norm != ''
        GROUP BY state_norm
        ORDER BY count DESC
    """).fetchall()
    con.close()

    return {row[0]: row[1] for row in result}


def get_nppes_gender_counts() -> dict[str, int]:
    """
    Get physician counts by gender from NPPES.

    Returns:
        Dictionary mapping gender code to count
    """
    if not NPPES_PHYSICIANS_PARQUET.exists():
        return {}

    con = duckdb.connect()
    result = con.execute(f"""
        SELECT gender, COUNT(*) as count
        FROM read_parquet('{str(NPPES_PHYSICIANS_PARQUET).replace(chr(92), "/")}')
        WHERE gender IS NOT NULL
        GROUP BY gender
    """).fetchall()
    con.close()

    return {row[0]: row[1] for row in result}


def load_nppes_for_linkage(
    states: Optional[list[str]] = None,
) -> "duckdb.DuckDBPyRelation":
    """
    Load NPPES physician data for record linkage.

    Args:
        states: Optional list of state abbreviations to filter to

    Returns:
        DuckDB relation with NPPES data
    """
    if not NPPES_PHYSICIANS_PARQUET.exists():
        raise FileNotFoundError(
            f"NPPES physicians parquet not found: {NPPES_PHYSICIANS_PARQUET}. "
            "Run the ingestion step first."
        )

    con = duckdb.connect()

    query = f"""
        SELECT *
        FROM read_parquet('{str(NPPES_PHYSICIANS_PARQUET).replace(chr(92), "/")}')
    """

    if states:
        states_str = ", ".join([f"'{s}'" for s in states])
        query += f" WHERE state_norm IN ({states_str})"

    return con.execute(query)


def get_physician_specialty_from_taxonomy(row: dict) -> Optional[str]:
    """
    Extract primary physician specialty from taxonomy codes.

    Checks all 15 taxonomy columns and returns the specialty classification
    for the first physician taxonomy code found.

    Args:
        row: Dictionary with taxonomy_code_1 through taxonomy_code_15

    Returns:
        Specialty classification string or None
    """
    # Specialty mapping for common physician taxonomy prefixes
    # Based on NUCC taxonomy: https://taxonomy.nucc.org/
    specialty_prefixes = {
        "207K": "Allergy & Immunology",
        "207L": "Anesthesiology",
        "207N": "Dermatology",
        "207P": "Emergency Medicine",
        "207Q": "Family Medicine",
        "207R": "Internal Medicine",
        "207S": "Colon & Rectal Surgery",
        "207T": "Neurological Surgery",
        "207U": "Nuclear Medicine",
        "207V": "Obstetrics & Gynecology",
        "207W": "Ophthalmology",
        "207X": "Orthopaedic Surgery",
        "207Y": "Otolaryngology",
        "207Z": "Pathology",
        "208": "Pediatrics",
        "208100": "Physical Medicine & Rehabilitation",
        "2082": "Plastic Surgery",
        "2083": "Preventive Medicine",
        "2084": "Psychiatry & Neurology",
        "2085": "Radiology",
        "2086": "Surgery",
        "2088": "Urology",
        "208D": "General Practice",
        "208U": "Clinical Pharmacology",
        "208V": "Pain Medicine",
        "209": "Podiatric Medicine",  # Not strictly physician but related
    }

    # Check all 15 taxonomy columns
    for i in range(1, 16):
        col = f"taxonomy_code_{i}"
        code = row.get(col)

        if not code or not str(code).startswith("20"):
            continue

        code_str = str(code)

        # Try to match increasingly specific prefixes
        for prefix_len in [6, 5, 4, 3]:
            if len(code_str) >= prefix_len:
                prefix = code_str[:prefix_len]
                if prefix in specialty_prefixes:
                    return specialty_prefixes[prefix]

    return "Other/Unknown"


def get_nppes_specialty_counts() -> dict[str, int]:
    """
    Get physician counts by specialty from NPPES.

    Returns:
        Dictionary mapping specialty to count
    """
    if not NPPES_PHYSICIANS_PARQUET.exists():
        return {}

    con = duckdb.connect()
    path = str(NPPES_PHYSICIANS_PARQUET).replace("\\", "/")

    # Get all rows and classify specialty
    df = con.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()
    con.close()

    # Apply specialty classification
    df["specialty"] = df.apply(
        lambda row: get_physician_specialty_from_taxonomy(row.to_dict()),
        axis=1,
    )

    return df["specialty"].value_counts().to_dict()


def get_nppes_taxonomy_column_stats() -> dict:
    """
    Get statistics on which taxonomy columns contain physician codes.

    Returns:
        Dictionary with counts of physicians found in each taxonomy column
    """
    if not NPPES_PHYSICIANS_PARQUET.exists():
        return {}

    con = duckdb.connect()
    path = str(NPPES_PHYSICIANS_PARQUET).replace("\\", "/")

    stats = {}

    # Count physicians by first taxonomy column with physician code
    result = con.execute(f"""
        SELECT first_physician_taxonomy_col, COUNT(*) as count
        FROM read_parquet('{path}')
        WHERE first_physician_taxonomy_col IS NOT NULL
        GROUP BY first_physician_taxonomy_col
        ORDER BY first_physician_taxonomy_col
    """).fetchall()

    stats["by_first_column"] = {row[0]: row[1] for row in result}

    # Count total physicians with codes in each column
    for i in range(1, 16):
        col_count = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{path}')
            WHERE taxonomy_code_{i} LIKE '20%'
        """).fetchone()[0]
        stats[f"column_{i}_count"] = col_count

    con.close()
    return stats
