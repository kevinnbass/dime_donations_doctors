"""
PECOS (CMS Provider Enrollment) data processing.

Ingests the CMS "Doctors and Clinicians National Downloadable File" to extract
graduation year and other provider attributes for physician analysis.

Data source: https://data.cms.gov/provider-data/dataset/mj5m-pzi6
"""

from pathlib import Path
from typing import Optional

import duckdb

from .config import (
    PECOS_RAW_DIR,
    PECOS_PHYSICIANS_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    NPPES_PHYSICIANS_PARQUET,
)


def find_pecos_file() -> Path:
    """
    Locate the PECOS Doctors and Clinicians file.

    Returns:
        Path to the PECOS data file

    Raises:
        FileNotFoundError: If no PECOS file is found
    """
    # Look for the Doctors and Clinicians National Downloadable File
    patterns = [
        "DAC_NationalDownloadableFile.csv",
        "DAC_NationalDownloadableFile*.csv",
        "Doctors_and_Clinicians*.csv",
        "*.csv",
    ]

    for pattern in patterns:
        matches = list(PECOS_RAW_DIR.glob(pattern))
        if matches:
            # Return the most recent one if multiple
            return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        f"No PECOS file found in {PECOS_RAW_DIR}.\n"
        "Please download the 'Doctors and Clinicians National Downloadable File' from:\n"
        "  https://data.cms.gov/provider-data/dataset/mj5m-pzi6\n"
        "Expected file: DAC_NationalDownloadableFile.csv"
    )


def ingest_pecos_physicians(
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Ingest PECOS Doctors and Clinicians file and extract physician data.

    Extracts NPI, graduation year, gender, and other key attributes.

    Args:
        output_path: Output parquet file path
        overwrite: Whether to overwrite existing file

    Returns:
        Path to the output parquet file
    """
    output_path = output_path or PECOS_PHYSICIANS_PARQUET

    if output_path.exists() and not overwrite:
        print(f"PECOS physicians parquet already exists: {output_path}")
        return output_path

    pecos_file = find_pecos_file()
    print(f"Ingesting PECOS Doctors and Clinicians: {pecos_file.name}")

    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")

    pecos_path = str(pecos_file).replace("\\", "/")
    output_path_str = str(output_path).replace("\\", "/")

    # First, inspect the columns to understand the schema
    cols = con.execute(f"""
        SELECT column_name
        FROM (DESCRIBE SELECT * FROM read_csv_auto('{pecos_path}', header=true, sample_size=1000))
    """).fetchall()
    col_names = [c[0] for c in cols]
    print(f"  Found {len(col_names)} columns")

    # The PECOS file uses these column names (from CMS data dictionary):
    # - NPI: National Provider Identifier
    # - Grd_yr: Graduation year
    # - gndr: Gender (M/F)
    # - Cred: Credentials
    # - pri_spec: Primary specialty
    # - st: State
    # - zip: ZIP code

    # Build the query based on available columns
    # Handle potential column name variations
    npi_col = next((c for c in col_names if c.upper() == 'NPI'), None)
    grad_col = next((c for c in col_names if c.upper() in ('GRD_YR', 'GRAD_YR', 'GRADUATION_YEAR')), None)
    gender_col = next((c for c in col_names if c.upper() in ('GNDR', 'GENDER', 'SEX')), None)
    cred_col = next((c for c in col_names if c.upper() in ('CRED', 'CREDENTIALS')), None)
    spec_col = next((c for c in col_names if c.upper() in ('PRI_SPEC', 'PRIMARY_SPECIALTY', 'SPECIALTY')), None)
    state_col = next((c for c in col_names if c.upper() in ('ST', 'STATE')), None)
    zip_col = next((c for c in col_names if c.upper() in ('ZIP', 'ZIPCODE', 'ZIP_CODE')), None)
    first_name_col = next((c for c in col_names if c.upper() in ('FRST_NM', 'FIRST_NAME')), None)
    last_name_col = next((c for c in col_names if c.upper() in ('LST_NM', 'LAST_NAME')), None)

    if not npi_col:
        raise ValueError(f"Could not find NPI column. Available columns: {col_names[:20]}...")

    print(f"  NPI column: {npi_col}")
    print(f"  Graduation year column: {grad_col}")
    print(f"  Gender column: {gender_col}")

    # Build SELECT clause dynamically
    select_parts = [f'"{npi_col}" as pecos_npi']

    if grad_col:
        select_parts.append(f'TRY_CAST("{grad_col}" AS INTEGER) as graduation_year')
    else:
        select_parts.append('NULL as graduation_year')
        print("  WARNING: No graduation year column found!")

    if gender_col:
        select_parts.append(f'"{gender_col}" as pecos_gender')
    else:
        select_parts.append('NULL as pecos_gender')

    if cred_col:
        select_parts.append(f'"{cred_col}" as pecos_credentials')
    else:
        select_parts.append('NULL as pecos_credentials')

    if spec_col:
        select_parts.append(f'"{spec_col}" as pecos_specialty')
    else:
        select_parts.append('NULL as pecos_specialty')

    if state_col:
        select_parts.append(f'"{state_col}" as pecos_state')
    else:
        select_parts.append('NULL as pecos_state')

    if zip_col:
        select_parts.append(f'SUBSTRING("{zip_col}", 1, 5) as pecos_zip5')
    else:
        select_parts.append('NULL as pecos_zip5')

    if first_name_col:
        select_parts.append(f'"{first_name_col}" as pecos_first_name')
    else:
        select_parts.append('NULL as pecos_first_name')

    if last_name_col:
        select_parts.append(f'"{last_name_col}" as pecos_last_name')
    else:
        select_parts.append('NULL as pecos_last_name')

    select_clause = ",\n            ".join(select_parts)

    # Use COPY for efficient streaming write
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = f"""
    COPY (
        SELECT DISTINCT
            {select_clause}
        FROM read_csv_auto('{pecos_path}', header=true, sample_size=100000, all_varchar=true)
        WHERE "{npi_col}" IS NOT NULL
    ) TO '{output_path_str}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(query)

    # Get summary statistics
    stats = con.execute(f"""
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT pecos_npi) as unique_npis,
            COUNT(graduation_year) as has_grad_year,
            MIN(graduation_year) as min_grad_year,
            MAX(graduation_year) as max_grad_year,
            COUNT(pecos_gender) as has_gender
        FROM read_parquet('{output_path_str}')
    """).fetchone()

    con.close()

    print(f"Created: {output_path}")
    print(f"  Total records: {stats[0]:,}")
    print(f"  Unique NPIs: {stats[1]:,}")
    print(f"  With graduation year: {stats[2]:,} ({100*stats[2]/stats[0]:.1f}%)")
    if stats[3] and stats[4]:
        print(f"  Graduation year range: {stats[3]} - {stats[4]}")
    print(f"  With gender: {stats[5]:,}")

    return output_path


def link_nppes_to_pecos(
    nppes_path: Optional[Path] = None,
    pecos_path: Optional[Path] = None,
) -> dict:
    """
    Link NPPES physicians to PECOS data via NPI.

    Args:
        nppes_path: Path to NPPES physicians parquet
        pecos_path: Path to PECOS physicians parquet

    Returns:
        Dictionary with linkage statistics
    """
    nppes_path = nppes_path or NPPES_PHYSICIANS_PARQUET
    pecos_path = pecos_path or PECOS_PHYSICIANS_PARQUET

    if not nppes_path.exists():
        raise FileNotFoundError(f"NPPES physicians not found: {nppes_path}")
    if not pecos_path.exists():
        raise FileNotFoundError(f"PECOS physicians not found: {pecos_path}")

    con = duckdb.connect()

    nppes_p = str(nppes_path).replace("\\", "/")
    pecos_p = str(pecos_path).replace("\\", "/")

    # Count NPPES records
    n_nppes = con.execute(f"""
        SELECT COUNT(DISTINCT npi) FROM read_parquet('{nppes_p}')
    """).fetchone()[0]

    # Count PECOS records
    n_pecos = con.execute(f"""
        SELECT COUNT(DISTINCT pecos_npi) FROM read_parquet('{pecos_p}')
    """).fetchone()[0]

    # Count PECOS records with graduation year
    n_pecos_grad = con.execute(f"""
        SELECT COUNT(DISTINCT pecos_npi)
        FROM read_parquet('{pecos_p}')
        WHERE graduation_year IS NOT NULL
    """).fetchone()[0]

    # Find overlap
    n_overlap = con.execute(f"""
        SELECT COUNT(DISTINCT n.npi)
        FROM read_parquet('{nppes_p}') n
        INNER JOIN read_parquet('{pecos_p}') p
            ON CAST(n.npi AS VARCHAR) = CAST(p.pecos_npi AS VARCHAR)
    """).fetchone()[0]

    # Find overlap with graduation year
    n_overlap_grad = con.execute(f"""
        SELECT COUNT(DISTINCT n.npi)
        FROM read_parquet('{nppes_p}') n
        INNER JOIN read_parquet('{pecos_p}') p
            ON CAST(n.npi AS VARCHAR) = CAST(p.pecos_npi AS VARCHAR)
        WHERE p.graduation_year IS NOT NULL
    """).fetchone()[0]

    con.close()

    stats = {
        "nppes_physicians": n_nppes,
        "pecos_providers": n_pecos,
        "pecos_with_grad_year": n_pecos_grad,
        "overlap_npis": n_overlap,
        "overlap_with_grad_year": n_overlap_grad,
        "overlap_rate_of_nppes": n_overlap / n_nppes if n_nppes > 0 else 0,
        "grad_year_rate_of_overlap": n_overlap_grad / n_overlap if n_overlap > 0 else 0,
    }

    print("NPPES-to-PECOS linkage:")
    print(f"  NPPES physicians: {stats['nppes_physicians']:,}")
    print(f"  PECOS providers: {stats['pecos_providers']:,}")
    print(f"  PECOS with graduation year: {stats['pecos_with_grad_year']:,}")
    print(f"  Overlap (NPIs in both): {stats['overlap_npis']:,} ({stats['overlap_rate_of_nppes']:.1%})")
    print(f"  Overlap with grad year: {stats['overlap_with_grad_year']:,} ({stats['grad_year_rate_of_overlap']:.1%} of overlap)")

    return stats


def add_pecos_columns_to_labels(
    labels_path: Optional[Path] = None,
    linkage_path: Optional[Path] = None,
    pecos_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Add PECOS columns to physician labels.

    Joins physician labels with NPPES linkage and PECOS data to add:
    - graduation_year: Medical school graduation year
    - pecos_gender: Gender from PECOS (as backup/validation)
    - pecos_specialty: Primary specialty from PECOS

    Args:
        labels_path: Path to physician labels parquet
        linkage_path: Path to NPPES linkage results
        pecos_path: Path to processed PECOS parquet
        output_path: Output path (defaults to overwriting labels)
        overwrite: Whether to overwrite existing file

    Returns:
        Path to output file
    """
    labels_path = labels_path or PHYSICIAN_LABELS_PARQUET
    linkage_path = linkage_path or LINKAGE_RESULTS_PARQUET
    pecos_path = pecos_path or PECOS_PHYSICIANS_PARQUET
    output_path = output_path or labels_path

    if not labels_path.exists():
        raise FileNotFoundError(f"Physician labels not found: {labels_path}")

    if not pecos_path.exists():
        print(f"PECOS parquet not found, skipping PECOS columns")
        return labels_path
    if not linkage_path.exists():
        print(f"NPPES linkage not found, skipping PECOS columns")
        return labels_path

    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")

    labels_p = str(labels_path).replace("\\", "/")
    linkage_p = str(linkage_path).replace("\\", "/")
    pecos_p = str(pecos_path).replace("\\", "/")

    # Use a temp file to avoid overwriting while reading
    temp_path = str(output_path).replace(".parquet", "_temp.parquet").replace("\\", "/")

    print("Adding PECOS columns to physician labels...")

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
        pecos_lookup AS (
            SELECT DISTINCT ON (pecos_npi)
                pecos_npi,
                graduation_year,
                pecos_gender,
                pecos_specialty
            FROM read_parquet('{pecos_p}')
            WHERE pecos_npi IS NOT NULL
            ORDER BY pecos_npi
        )
        SELECT
            l.*,
            p.graduation_year,
            p.pecos_gender,
            p.pecos_specialty
        FROM read_parquet('{labels_p}') l
        LEFT JOIN linkage_best lb ON l.bonica_cid = lb.bonica_cid
        LEFT JOIN pecos_lookup p ON CAST(lb.nppes_npi AS VARCHAR) = CAST(p.pecos_npi AS VARCHAR)
    ) TO '{temp_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(query)

    # Get statistics
    stats = con.execute(f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN graduation_year IS NOT NULL THEN 1 ELSE 0 END) as has_grad_year,
            SUM(CASE WHEN physician_final AND graduation_year IS NOT NULL THEN 1 ELSE 0 END) as physicians_with_grad_year,
            MIN(graduation_year) as min_year,
            MAX(graduation_year) as max_year
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
    print(f"  With graduation year: {stats[1]:,} ({100*stats[1]/stats[0]:.1f}%)")
    print(f"  Physicians with grad year: {stats[2]:,}")
    if stats[3] and stats[4]:
        print(f"  Graduation year range: {stats[3]} - {stats[4]}")

    return output_path


def get_pecos_stats() -> dict:
    """Get statistics from the processed PECOS data."""
    if not PECOS_PHYSICIANS_PARQUET.exists():
        return {}

    con = duckdb.connect()
    path = str(PECOS_PHYSICIANS_PARQUET).replace("\\", "/")

    stats = {}
    stats["n_total"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_unique_npis"] = con.execute(
        f"SELECT COUNT(DISTINCT pecos_npi) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_with_grad_year"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}') WHERE graduation_year IS NOT NULL"
    ).fetchone()[0]

    grad_range = con.execute(f"""
        SELECT MIN(graduation_year), MAX(graduation_year)
        FROM read_parquet('{path}')
        WHERE graduation_year IS NOT NULL
    """).fetchone()
    stats["min_grad_year"] = grad_range[0]
    stats["max_grad_year"] = grad_range[1]

    # Gender distribution
    gender_dist = con.execute(f"""
        SELECT pecos_gender, COUNT(*) as n
        FROM read_parquet('{path}')
        WHERE pecos_gender IS NOT NULL
        GROUP BY pecos_gender
        ORDER BY n DESC
    """).fetchdf()
    stats["gender_distribution"] = gender_dist.to_dict("records")

    # Graduation year distribution by decade
    grad_decades = con.execute(f"""
        SELECT
            (graduation_year / 10) * 10 as decade,
            COUNT(*) as n
        FROM read_parquet('{path}')
        WHERE graduation_year IS NOT NULL
        GROUP BY decade
        ORDER BY decade
    """).fetchdf()
    stats["graduation_decades"] = grad_decades.to_dict("records")

    con.close()
    return stats
