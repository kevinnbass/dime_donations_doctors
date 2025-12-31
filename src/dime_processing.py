"""
DIME data processing module.

Handles ingestion and transformation of DIME campaign finance data
using DuckDB for efficient querying of large files.
"""

import gzip
import shutil
from pathlib import Path
from typing import Optional

import duckdb

from .config import (
    CYCLES,
    DIME_DONORS_PARQUET,
    DIME_RAW_DIR,
    PROCESSED_DATA_DIR,
)


def find_dime_contributors_file() -> Path:
    """
    Locate the DIME contributors file in the raw data directory.

    Searches for common file patterns.

    Returns:
        Path to the DIME contributors file

    Raises:
        FileNotFoundError: If no DIME file is found
    """
    patterns = [
        "dime_contributors_*.csv",
        "dime_contributors_*.csv.gz",
        "contributors*.csv",
        "contributors*.csv.gz",
    ]

    for pattern in patterns:
        matches = list(DIME_RAW_DIR.glob(pattern))
        if matches:
            # Return the most recent file
            return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        f"No DIME contributors file found in {DIME_RAW_DIR}. "
        "Expected file matching pattern: dime_contributors_*.csv[.gz]"
    )


def find_dime_recipients_file() -> Path:
    """
    Locate the DIME recipients file in the raw data directory.

    Returns:
        Path to the DIME recipients file

    Raises:
        FileNotFoundError: If no DIME recipients file is found
    """
    patterns = [
        "dime_recipients_*.csv",
        "dime_recipients_*.csv.gz",
        "dime_recipients.csv",
        "dime_recipients.csv.gz",
        "recipients*.csv",
        "recipients*.csv.gz",
    ]

    for pattern in patterns:
        matches = list(DIME_RAW_DIR.glob(pattern))
        if matches:
            return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        f"No DIME recipients file found in {DIME_RAW_DIR}. "
        "Expected file matching pattern: dime_recipients_*.csv[.gz]"
    )


def find_dime_sqlite_file() -> Path:
    """
    Locate the DIME SQLite database file.

    Returns:
        Path to the DIME SQLite file

    Raises:
        FileNotFoundError: If no SQLite file is found
    """
    patterns = [
        "dime_v4.sqlite3",
        "dime_v4.sqlite3.gz",
        "dime*.sqlite3",
        "dime*.sqlite3.gz",
        "dime*.db",
    ]

    for pattern in patterns:
        matches = list(DIME_RAW_DIR.glob(pattern))
        if matches:
            return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        f"No DIME SQLite file found in {DIME_RAW_DIR}. "
        "Expected file matching pattern: dime_v4.sqlite3[.gz]"
    )


def decompress_gzip(gz_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Decompress a gzip file if needed.

    Args:
        gz_path: Path to the .gz file
        output_path: Optional output path. If None, removes .gz extension.

    Returns:
        Path to the decompressed file
    """
    if not gz_path.suffix == ".gz":
        return gz_path

    if output_path is None:
        output_path = gz_path.with_suffix("")

    if output_path.exists():
        return output_path

    print(f"Decompressing {gz_path.name}...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return output_path


def ingest_dime_contributors(
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Ingest DIME contributors CSV to Parquet format.

    Filters to individual contributors only and selects relevant columns.
    Uses DuckDB for efficient processing of large files.

    Args:
        output_path: Output parquet file path. If None, uses default.
        overwrite: Whether to overwrite existing output file.

    Returns:
        Path to the output parquet file
    """
    output_path = output_path or DIME_DONORS_PARQUET

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return output_path

    # Find and prepare input file
    input_file = find_dime_contributors_file()
    if input_file.suffix == ".gz":
        input_file = decompress_gzip(input_file)

    print(f"Processing DIME contributors from: {input_file}")

    # Build column list for amount columns
    amount_cols = ", ".join([f'"amount.{year}"' for year in CYCLES])

    # Use DuckDB to read and transform
    con = duckdb.connect()

    query = f"""
    COPY (
        SELECT
            "bonica.cid" AS bonica_cid,
            "contributor.type" AS contributor_type,
            "most.recent.contributor.name" AS contributor_name,
            "most.recent.contributor.address" AS contributor_address,
            "most.recent.contributor.city" AS contributor_city,
            "most.recent.contributor.state" AS contributor_state,
            "most.recent.contributor.zipcode" AS contributor_zipcode,
            "most.recent.contributor.occupation" AS contributor_occupation,
            "most.recent.contributor.employer" AS contributor_employer,
            "contributor.gender" AS contributor_gender,
            "contributor.cfscore" AS contributor_cfscore,
            "is.projected" AS is_projected,
            "first_cycle_active" AS first_cycle_active,
            "last_cycle_active" AS last_cycle_active,
            {amount_cols}
        FROM read_csv_auto('{str(input_file).replace(chr(92), "/")}',
                          header=true,
                          all_varchar=false,
                          sample_size=100000,
                          ignore_errors=true)
        WHERE "contributor.type" = 'I'
    ) TO '{str(output_path).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    print("Converting to Parquet format...")
    con.execute(query)
    con.close()

    print(f"Created: {output_path}")
    return output_path


def ingest_dime_recipients(
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Ingest DIME recipients CSV to Parquet format.

    Args:
        output_path: Output parquet file path. If None, uses default.
        overwrite: Whether to overwrite existing output file.

    Returns:
        Path to the output parquet file
    """
    output_path = output_path or (PROCESSED_DATA_DIR / "dime_recipients.parquet")

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return output_path

    input_file = find_dime_recipients_file()
    if input_file.suffix == ".gz":
        input_file = decompress_gzip(input_file)

    print(f"Processing DIME recipients from: {input_file}")

    con = duckdb.connect()

    query = f"""
    COPY (
        SELECT
            "bonica.rid" AS bonica_rid,
            "name" AS recipient_name,
            "party" AS recipient_party,
            "recipient.type" AS recipient_type,
            "seat" AS seat,
            "state" AS state,
            "recipient.cfscore" AS recipient_cfscore,
            "dwnom1" AS dwnom1
        FROM read_csv_auto('{str(input_file).replace(chr(92), "/")}',
                          header=true,
                          all_varchar=true,
                          sample_size=100000,
                          ignore_errors=true)
    ) TO '{str(output_path).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(query)
    con.close()

    print(f"Created: {output_path}")
    return output_path


def get_dime_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Get a DuckDB connection with DIME data registered.

    Returns:
        DuckDB connection with 'donors' and 'recipients' views
    """
    con = duckdb.connect()

    # Register donors parquet
    if DIME_DONORS_PARQUET.exists():
        con.execute(f"""
            CREATE VIEW donors AS
            SELECT * FROM read_parquet('{str(DIME_DONORS_PARQUET).replace(chr(92), "/")}')
        """)

    # Register recipients parquet
    recipients_path = PROCESSED_DATA_DIR / "dime_recipients.parquet"
    if recipients_path.exists():
        con.execute(f"""
            CREATE VIEW recipients AS
            SELECT * FROM read_parquet('{str(recipients_path).replace(chr(92), "/")}')
        """)

    return con


def get_sqlite_connection() -> duckdb.DuckDBPyConnection:
    """
    Get a DuckDB connection to the DIME SQLite database for contribution records.

    Uses DuckDB's SQLite scanner extension.

    Returns:
        DuckDB connection with SQLite attached
    """
    sqlite_file = find_dime_sqlite_file()

    # Decompress if needed
    if sqlite_file.suffix == ".gz":
        sqlite_file = decompress_gzip(sqlite_file)

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{str(sqlite_file).replace(chr(92), '/')}' AS dime (TYPE SQLITE, READ_ONLY)")

    return con


def query_donors(
    query: str,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Execute a query against the DIME donors data.

    Args:
        query: SQL query string. Use 'donors' as the table name.
        con: Optional existing connection. If None, creates one.

    Returns:
        DuckDB relation with query results
    """
    should_close = con is None
    con = con or get_dime_duckdb_connection()

    result = con.execute(query)

    if should_close:
        # Return result and close handled by caller
        pass

    return result


def get_donor_count() -> int:
    """Get the total number of individual donors in the processed data."""
    con = get_dime_duckdb_connection()
    result = con.execute("SELECT COUNT(*) FROM donors").fetchone()
    con.close()
    return result[0] if result else 0


def get_cycle_donor_counts() -> dict[int, int]:
    """
    Get the number of donors who donated in each cycle.

    Returns:
        Dictionary mapping cycle year to donor count
    """
    con = get_dime_duckdb_connection()

    counts = {}
    for year in CYCLES:
        col = f'"amount.{year}"'
        query = f"SELECT COUNT(*) FROM donors WHERE {col} > 0"
        result = con.execute(query).fetchone()
        counts[year] = result[0] if result else 0

    con.close()
    return counts
