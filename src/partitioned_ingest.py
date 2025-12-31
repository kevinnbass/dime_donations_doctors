"""
Partitioned ingestion module for large DIME itemized contribution files.

Revised approach after both DuckDB and Polars failed with gzipped files:
- Uses PyArrow streaming for partitioning (true streaming with gzip)
- Uses DuckDB for aggregation per partition (efficient SQL)
- Joins with recipients during aggregation, not partitioning

Key approach:
1. Partition CSV -> Parquet by hash(donor_id) % N using PyArrow streaming
2. Aggregate each partition with DuckDB (includes recipients join)
3. Delete partition files immediately after processing
4. Combine results into final output

Resource guarantees:
- RAM: ~1-2GB (PyArrow streaming + DuckDB per-partition)
- Temp Disk: ~20-25GB (partitions, deleted as processed)
- Determinism: ORDER BY for tie-breaks
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

from .config import (
    DIME_ITEMIZED_TMP_DIR,
    DIME_RECIPIENTS_PARQUET,
    DONOR_CYCLE_PANEL_PARTITIONED_DIR,
)

logger = logging.getLogger(__name__)

# Configuration
NUM_PARTITIONS = 128  # Number of hash partitions
PYARROW_BATCH_SIZE = 100_000  # Rows per batch - PyArrow streaming
MEMORY_LIMIT = "4GB"  # DuckDB memory limit for aggregation


def get_temp_partition_dir(cycle: int) -> Path:
    """Get temp directory for partition files."""
    return DIME_ITEMIZED_TMP_DIR / f"parts_{cycle}"


def get_temp_donors_dir(cycle: int) -> Path:
    """Get temp directory for aggregated donor files."""
    return DIME_ITEMIZED_TMP_DIR / f"donors_{cycle}"


def detect_csv_columns_pyarrow(csv_path: Path) -> dict:
    """
    Detect column names from CSV file using PyArrow.

    Returns dict with keys: cid, rid, amount, occupation, employer, state, zip, city
    """
    # Read just the first row to get column names
    read_options = pv.ReadOptions(block_size=1024*1024)  # 1MB block
    parse_options = pv.ParseOptions()
    convert_options = pv.ConvertOptions(strings_can_be_null=True)

    reader = pv.open_csv(
        csv_path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    # Get first batch just to read schema
    first_batch = reader.read_next_batch()
    available_cols = first_batch.schema.names
    logger.info(f"Available columns: {available_cols[:10]}...")

    cols = {}

    # Find key columns
    for col in available_cols:
        col_lower = col.lower()
        if "bonica" in col_lower and "cid" in col_lower:
            cols['cid'] = col
        elif "bonica" in col_lower and "rid" in col_lower:
            cols['rid'] = col
        elif col_lower in ("amount", "transaction.amount", "transaction_amount"):
            cols['amount'] = col
        elif "occupation" in col_lower:
            cols.setdefault('occupation', col)
        elif "employer" in col_lower:
            cols.setdefault('employer', col)
        elif "state" in col_lower and "contributor" in col_lower:
            cols.setdefault('state', col)
        elif "zip" in col_lower:
            cols.setdefault('zip', col)
        elif "city" in col_lower:
            cols.setdefault('city', col)

    # Fallbacks
    if 'cid' not in cols:
        cols['cid'] = next((c for c in available_cols if 'cid' in c.lower()), None)
    if 'rid' not in cols:
        cols['rid'] = next((c for c in available_cols if 'rid' in c.lower()), None)
    if 'amount' not in cols:
        cols['amount'] = next((c for c in available_cols if 'amount' in c.lower()), None)

    return cols


def hash_string(s) -> int:
    """Simple deterministic hash for partitioning."""
    if s is None:
        return 0
    # Convert to string if not already
    s = str(s)
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h


def partition_csv_with_pyarrow(
    csv_path: Path,
    partition_dir: Path,
    cols: dict,
    num_partitions: int = NUM_PARTITIONS,
    batch_size: int = PYARROW_BATCH_SIZE,
) -> int:
    """
    Partition CSV to Parquet files using PyArrow streaming.

    Uses PyArrow's true streaming CSV reader which handles gzip efficiently.
    Each batch is immediately written to partition files.

    Returns:
        Number of rows processed
    """
    # Ensure clean output directory
    if partition_dir.exists():
        shutil.rmtree(partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Create all partition directories upfront
    for i in range(num_partitions):
        (partition_dir / f"part={i}").mkdir(exist_ok=True)

    logger.info(f"Partitioning CSV into {num_partitions} partitions using PyArrow...")

    cid_col = cols.get('cid')
    rid_col = cols.get('rid')
    amount_col = cols.get('amount')
    occ_col = cols.get('occupation')
    emp_col = cols.get('employer')
    state_col = cols.get('state')
    zip_col = cols.get('zip')
    city_col = cols.get('city')

    # Columns to read (only what we need)
    use_cols = [c for c in [cid_col, rid_col, amount_col, occ_col, emp_col, state_col, zip_col, city_col] if c]

    # Set up PyArrow streaming reader
    read_options = pv.ReadOptions(
        column_names=None,  # Read header from file
        block_size=10 * 1024 * 1024,  # 10MB block for streaming
    )
    parse_options = pv.ParseOptions(
        invalid_row_handler=lambda x: 'skip',  # Skip bad rows
    )
    convert_options = pv.ConvertOptions(
        include_columns=use_cols,
        strings_can_be_null=True,
    )

    reader = pv.open_csv(
        csv_path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    # Output schema (standardized column names)
    output_schema = pa.schema([
        ('cid', pa.string()),
        ('rid', pa.string()),
        ('amount', pa.float64()),
        ('occupation', pa.string()),
        ('employer', pa.string()),
        ('state', pa.string()),
        ('zip5', pa.string()),
        ('city', pa.string()),
    ])

    # Track partition writers
    partition_writers = {}
    partition_counts = [0] * num_partitions

    total_rows = 0
    valid_rows = 0
    batch_num = 0

    try:
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            except Exception as e:
                logger.warning(f"Error reading batch: {e}")
                continue

            batch_rows = batch.num_rows
            total_rows += batch_rows
            batch_num += 1

            # Convert to Python for processing (necessary for partitioning)
            cid_array = batch.column(cid_col).to_pylist() if cid_col in batch.schema.names else [None] * batch_rows
            rid_array = batch.column(rid_col).to_pylist() if rid_col and rid_col in batch.schema.names else [None] * batch_rows
            amount_array = batch.column(amount_col).to_pylist() if amount_col in batch.schema.names else [None] * batch_rows
            occ_array = batch.column(occ_col).to_pylist() if occ_col and occ_col in batch.schema.names else [None] * batch_rows
            emp_array = batch.column(emp_col).to_pylist() if emp_col and emp_col in batch.schema.names else [None] * batch_rows
            state_array = batch.column(state_col).to_pylist() if state_col and state_col in batch.schema.names else [None] * batch_rows
            zip_array = batch.column(zip_col).to_pylist() if zip_col and zip_col in batch.schema.names else [None] * batch_rows
            city_array = batch.column(city_col).to_pylist() if city_col and city_col in batch.schema.names else [None] * batch_rows

            # Partition data
            partition_buffers = {i: {'cid': [], 'rid': [], 'amount': [], 'occupation': [], 'employer': [], 'state': [], 'zip5': [], 'city': []} for i in range(num_partitions)}

            for i in range(batch_rows):
                cid = cid_array[i]
                if cid is None:
                    continue

                # Parse amount
                try:
                    amount = float(amount_array[i]) if amount_array[i] is not None else None
                except (ValueError, TypeError):
                    amount = None

                if amount is None or amount <= 0:
                    continue

                # Compute partition
                part_num = hash_string(cid) % num_partitions

                # Helper to convert to string and handle types
                def to_str(val):
                    if val is None:
                        return None
                    return str(val).strip() if str(val).strip() else None

                def get_zip5(val):
                    if val is None:
                        return None
                    s = str(val).strip()
                    if len(s) >= 5:
                        return s[:5]
                    return s if s else None

                # Add to partition buffer
                buf = partition_buffers[part_num]
                buf['cid'].append(to_str(cid))
                buf['rid'].append(to_str(rid_array[i]))
                buf['amount'].append(amount)
                buf['occupation'].append(to_str(occ_array[i]))
                buf['employer'].append(to_str(emp_array[i]))
                buf['state'].append(to_str(state_array[i]))
                buf['zip5'].append(get_zip5(zip_array[i]))
                buf['city'].append(to_str(city_array[i]))

            # Write partition buffers to files
            for part_num, buf in partition_buffers.items():
                if len(buf['cid']) == 0:
                    continue

                # Create PyArrow table
                table = pa.table({
                    'cid': pa.array(buf['cid'], type=pa.string()),
                    'rid': pa.array(buf['rid'], type=pa.string()),
                    'amount': pa.array(buf['amount'], type=pa.float64()),
                    'occupation': pa.array(buf['occupation'], type=pa.string()),
                    'employer': pa.array(buf['employer'], type=pa.string()),
                    'state': pa.array(buf['state'], type=pa.string()),
                    'zip5': pa.array(buf['zip5'], type=pa.string()),
                    'city': pa.array(buf['city'], type=pa.string()),
                })

                # Write to partition file
                part_dir = partition_dir / f"part={part_num}"
                part_file = part_dir / f"data_{batch_num}.parquet"
                pq.write_table(table, part_file, compression='zstd')

                valid_rows += len(buf['cid'])
                partition_counts[part_num] += len(buf['cid'])

            # Log progress
            if batch_num % 50 == 0:
                logger.info(f"  Batch {batch_num}: {total_rows:,} rows read, {valid_rows:,} valid rows written")

            # Release memory
            del batch
            del partition_buffers

    except Exception as e:
        logger.error(f"Error during partitioning: {e}")
        import traceback
        traceback.print_exc()
        raise

    logger.info(f"Partitioned {valid_rows:,} valid rows into {num_partitions} partitions")
    return valid_rows



def partition_csv_with_duckdb(
    csv_path: Path,
    partition_dir: Path,
    temp_dir: Path,
    num_partitions: int = NUM_PARTITIONS,
) -> int:
    """
    Partition CSV to Parquet files using DuckDB native partitioning.

    MUCH faster than PyArrow for uncompressed CSV files because:
    - DuckDB reads CSV in parallel (multiple threads)
    - DuckDB writes partitions in parallel

    Only works well with uncompressed CSV files (gzip is still sequential).
    """
    if partition_dir.exists():
        shutil.rmtree(partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Partitioning CSV into {num_partitions} partitions using DuckDB (parallel)...")

    csv_path_str = str(csv_path).replace("\\", "/")
    partition_dir_str = str(partition_dir).replace("\\", "/")
    temp_dir_str = str(temp_dir).replace("\\", "/")

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{temp_dir_str}'")

    partition_query = f"""
        COPY (
            SELECT
                CAST("bonica.cid" AS VARCHAR) AS cid,
                CAST("bonica.rid" AS VARCHAR) AS rid,
                TRY_CAST(amount AS DOUBLE) AS amount,
                NULLIF(TRIM("contributor.occupation"), '') AS occupation,
                NULLIF(TRIM("contributor.employer"), '') AS employer,
                NULLIF(TRIM("contributor.state"), '') AS state,
                NULLIF(SUBSTRING(TRIM(CAST("contributor.zipcode" AS VARCHAR)), 1, 5), '') AS zip5,
                NULLIF(TRIM("contributor.city"), '') AS city,
                ABS(hash(CAST("bonica.cid" AS VARCHAR))) % {num_partitions} AS part
            FROM read_csv('{csv_path_str}',
                header=true,
                ignore_errors=true,
                parallel=true
            )
            WHERE "bonica.cid" IS NOT NULL
              AND TRY_CAST(amount AS DOUBLE) > 0
        ) TO '{partition_dir_str}'
        (FORMAT parquet, PARTITION_BY (part), COMPRESSION zstd)
    """

    con.execute(partition_query)

    total_rows = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{partition_dir_str}/*/*.parquet')
    """).fetchone()[0]

    con.close()

    logger.info(f"Partitioned {total_rows:,} valid rows into {num_partitions} partitions using DuckDB")
    return total_rows




def partition_csv_with_duckdb(
    csv_path: Path,
    partition_dir: Path,
    temp_dir: Path,
    num_partitions: int = NUM_PARTITIONS,
) -> int:
    """
    Partition CSV to Parquet files using DuckDB native partitioning.

    MUCH faster than PyArrow for uncompressed CSV files because:
    - DuckDB reads CSV in parallel (multiple threads)
    - DuckDB writes partitions in parallel

    Only works well with uncompressed CSV files (gzip is still sequential).
    """
    if partition_dir.exists():
        shutil.rmtree(partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Partitioning CSV into {num_partitions} partitions using DuckDB (parallel)...")

    csv_path_str = str(csv_path).replace("\\", "/")
    partition_dir_str = str(partition_dir).replace("\\", "/")
    temp_dir_str = str(temp_dir).replace("\\", "/")

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{temp_dir_str}'")

    partition_query = f"""
        COPY (
            SELECT
                CAST("bonica.cid" AS VARCHAR) AS cid,
                CAST("bonica.rid" AS VARCHAR) AS rid,
                TRY_CAST(amount AS DOUBLE) AS amount,
                NULLIF(TRIM("contributor.occupation"), '') AS occupation,
                NULLIF(TRIM("contributor.employer"), '') AS employer,
                NULLIF(TRIM("contributor.state"), '') AS state,
                NULLIF(SUBSTRING(TRIM(CAST("contributor.zipcode" AS VARCHAR)), 1, 5), '') AS zip5,
                NULLIF(TRIM("contributor.city"), '') AS city,
                ABS(hash(CAST("bonica.cid" AS VARCHAR))) % {num_partitions} AS part
            FROM read_csv('{csv_path_str}',
                header=true,
                ignore_errors=true,
                parallel=true
            )
            WHERE "bonica.cid" IS NOT NULL
              AND TRY_CAST(amount AS DOUBLE) > 0
        ) TO '{partition_dir_str}'
        (FORMAT parquet, PARTITION_BY (part), COMPRESSION zstd)
    """

    con.execute(partition_query)

    total_rows = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{partition_dir_str}/*/*.parquet')
    """).fetchone()[0]

    con.close()

    logger.info(f"Partitioned {total_rows:,} valid rows into {num_partitions} partitions using DuckDB")
    return total_rows


def aggregate_partition(
    con: duckdb.DuckDBPyConnection,
    partition_path: Path,
    output_path: Path,
    recipients_path: Path,
    cycle: int,
) -> int:
    """
    Aggregate a single partition to donor-level statistics.

    Joins with recipients lookup during aggregation (per-partition, ~7M rows).

    Uses:
    - weighted averages for revealed scores
    - first(value ORDER BY amt DESC, value ASC) for deterministic modal values

    Returns:
        Number of donors in this partition
    """
    partition_path_str = str(partition_path).replace("\\", "/")
    output_path_str = str(output_path).replace("\\", "/")
    recipients_path_str = str(recipients_path).replace("\\", "/")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The aggregation query for one partition
    # Join with recipients here (per-partition = bounded memory)
    agg_query = f"""
        COPY (
            WITH raw AS (
                SELECT * FROM read_parquet('{partition_path_str}/*.parquet')
            ),
            base AS (
                SELECT
                    r.cid,
                    r.rid,
                    r.amount,
                    r.occupation,
                    r.employer,
                    r.state,
                    r.zip5,
                    r.city,
                    TRY_CAST(rec.recipient_cfscore AS DOUBLE) AS cfscore,
                    CASE
                        WHEN rec.recipient_party IN ('100', 'DEM', 'D') THEN 1
                        WHEN rec.recipient_party IN ('200', 'REP', 'R') THEN -1
                        ELSE 0
                    END AS party_score
                FROM raw r
                LEFT JOIN read_parquet('{recipients_path_str}') rec
                    ON r.rid = rec.bonica_rid
            ),
            -- Numeric aggregates
            num AS (
                SELECT
                    cid,
                    {cycle} AS cycle,
                    SUM(amount) AS total_amount,
                    COUNT(*) AS n_contributions,
                    COUNT(DISTINCT rid) AS n_unique_recipients,
                    SUM(CASE WHEN cfscore IS NOT NULL THEN amount * cfscore ELSE 0 END) /
                        NULLIF(SUM(CASE WHEN cfscore IS NOT NULL THEN amount ELSE 0 END), 0)
                        AS revealed_cfscore_cycle,
                    SUM(amount * party_score) /
                        NULLIF(SUM(CASE WHEN party_score != 0 THEN amount ELSE 0 END), 0)
                        AS revealed_party_cycle
                FROM base
                GROUP BY cid
            ),
            -- Modal occupation (highest total amount, alphabetical tie-break)
            occ_sums AS (
                SELECT cid, occupation, SUM(amount) AS amt
                FROM base WHERE occupation IS NOT NULL
                GROUP BY cid, occupation
            ),
            occ_mode AS (
                SELECT cid, FIRST(occupation ORDER BY amt DESC, occupation ASC) AS occupation_cycle
                FROM occ_sums GROUP BY cid
            ),
            -- Modal employer
            emp_sums AS (
                SELECT cid, employer, SUM(amount) AS amt
                FROM base WHERE employer IS NOT NULL
                GROUP BY cid, employer
            ),
            emp_mode AS (
                SELECT cid, FIRST(employer ORDER BY amt DESC, employer ASC) AS employer_cycle
                FROM emp_sums GROUP BY cid
            ),
            -- Modal state
            state_sums AS (
                SELECT cid, state, SUM(amount) AS amt
                FROM base WHERE state IS NOT NULL
                GROUP BY cid, state
            ),
            state_mode AS (
                SELECT cid, FIRST(state ORDER BY amt DESC, state ASC) AS state_cycle
                FROM state_sums GROUP BY cid
            ),
            -- Modal ZIP
            zip_sums AS (
                SELECT cid, zip5, SUM(amount) AS amt
                FROM base WHERE zip5 IS NOT NULL
                GROUP BY cid, zip5
            ),
            zip_mode AS (
                SELECT cid, FIRST(zip5 ORDER BY amt DESC, zip5 ASC) AS zip5_cycle
                FROM zip_sums GROUP BY cid
            ),
            -- Modal city
            city_sums AS (
                SELECT cid, city, SUM(amount) AS amt
                FROM base WHERE city IS NOT NULL
                GROUP BY cid, city
            ),
            city_mode AS (
                SELECT cid, FIRST(city ORDER BY amt DESC, city ASC) AS city_cycle
                FROM city_sums GROUP BY cid
            )
            SELECT
                num.cid AS bonica_cid,
                num.cycle,
                num.total_amount,
                num.n_contributions,
                num.n_unique_recipients,
                num.revealed_cfscore_cycle,
                num.revealed_party_cycle,
                occ_mode.occupation_cycle,
                emp_mode.employer_cycle,
                state_mode.state_cycle,
                zip_mode.zip5_cycle,
                city_mode.city_cycle
            FROM num
            LEFT JOIN occ_mode ON num.cid = occ_mode.cid
            LEFT JOIN emp_mode ON num.cid = emp_mode.cid
            LEFT JOIN state_mode ON num.cid = state_mode.cid
            LEFT JOIN zip_mode ON num.cid = zip_mode.cid
            LEFT JOIN city_mode ON num.cid = city_mode.cid
        ) TO '{output_path_str}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(agg_query)

    # Count donors written
    donor_count = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{output_path_str}')
    """).fetchone()[0]

    return donor_count


def process_cycle_partitioned(
    cycle: int,
    csv_path: Path,
    recipients_path: Path = None,
    output_dir: Path = None,
    overwrite: bool = False,
    temp_dir: Path = None,
) -> dict:
    """
    Process a large cycle using PyArrow partitioning + DuckDB aggregation.

    This approach:
    1. Partitions CSV -> Parquet using PyArrow streaming (memory-bounded)
    2. Aggregates each partition with DuckDB (includes recipients join)
    3. Deletes partition files immediately after processing
    4. Combines results into final output

    Args:
        cycle: Election cycle year
        csv_path: Path to downloaded CSV file (can be .gz)
        recipients_path: Path to recipients Parquet
        output_dir: Base directory for partitioned output
        overwrite: Whether to overwrite existing partition
        temp_dir: Directory for DuckDB temp files

    Returns:
        Dictionary with processing statistics
    """
    output_dir = output_dir or DONOR_CYCLE_PANEL_PARTITIONED_DIR
    recipients_path = recipients_path or DIME_RECIPIENTS_PARQUET
    temp_dir = temp_dir or (DIME_ITEMIZED_TMP_DIR / "duckdb_temp")

    partition_dir = get_temp_partition_dir(cycle)
    donors_dir = get_temp_donors_dir(cycle)
    final_output_dir = output_dir / f"cycle={cycle}"
    manifest_path = output_dir / f"cycle_{cycle}_manifest.json"

    # Check for uncompressed CSV (prefer it over gzip for speed)
    if str(csv_path).endswith('.gz'):
        uncompressed_csv = Path(str(csv_path)[:-3])
        if uncompressed_csv.exists():
            logger.info(f"Found uncompressed CSV: {uncompressed_csv.name}, using it for parallel reading")
            csv_path = uncompressed_csv

    use_duckdb_partitioning = not str(csv_path).endswith('.gz')

    # Check if already exists
    if final_output_dir.exists() and not overwrite:
        parquet_files = list(final_output_dir.glob("*.parquet"))
        if parquet_files:
            logger.info(f"Cycle {cycle} already exists, skipping")
            return {"cycle": cycle, "status": "skipped"}

    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any existing output
    if final_output_dir.exists():
        shutil.rmtree(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    method = "DuckDB parallel" if use_duckdb_partitioning else "PyArrow streaming"
    logger.info(f"Processing cycle {cycle} from {csv_path.name} ({method})")

    try:
        # Step 1: Detect columns
        cols = detect_csv_columns_pyarrow(csv_path)
        logger.info(f"Detected columns: {cols}")

        if not cols.get('cid') or not cols.get('amount'):
            raise ValueError(f"Missing required columns. Found: {cols}")

        # Step 2: Partition CSV -> Parquet (DuckDB for uncompressed, PyArrow for gzip)
        if use_duckdb_partitioning:
            n_raw_rows = partition_csv_with_duckdb(csv_path, partition_dir, temp_dir)
        else:
            n_raw_rows = partition_csv_with_pyarrow(csv_path, partition_dir, cols)

        # Step 3: Aggregate each partition with DuckDB
        logger.info("Aggregating partitions with DuckDB...")

        # Create DuckDB connection for aggregation
        con = duckdb.connect()
        temp_dir_str = str(temp_dir).replace("\\", "/")
        con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
        con.execute(f"SET temp_directory = '{temp_dir_str}'")

        total_donors = 0
        partitions_processed = 0

        # Get list of partition directories that actually have data
        partition_dirs = sorted([
            d for d in partition_dir.iterdir()
            if d.is_dir() and d.name.startswith("part=")
        ])

        for part_dir in partition_dirs:
            part_num = part_dir.name.split("=")[1]
            output_file = donors_dir / f"part_{part_num}.parquet"

            try:
                donor_count = aggregate_partition(
                    con, part_dir, output_file, recipients_path, cycle
                )
                total_donors += donor_count
            except Exception as e:
                logger.warning(f"Error aggregating partition {part_num}: {e}")

            partitions_processed += 1

            # Delete partition immediately to save disk
            shutil.rmtree(part_dir)

            if partitions_processed % 20 == 0:
                logger.info(f"  Aggregated {partitions_processed}/{len(partition_dirs)} partitions, {total_donors:,} donors")

        # Clean up empty partition directory
        if partition_dir.exists():
            shutil.rmtree(partition_dir)

        logger.info(f"Aggregated {total_donors:,} donors from {partitions_processed} partitions")

        # Step 4: Combine into final output
        logger.info("Combining donor files into final output...")
        donors_dir_str = str(donors_dir).replace("\\", "/")
        final_path = final_output_dir / "part-0.parquet"
        final_path_str = str(final_path).replace("\\", "/")

        con.execute(f"""
            COPY (
                SELECT * FROM read_parquet('{donors_dir_str}/*.parquet')
            ) TO '{final_path_str}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # Verify final count
        final_count = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{final_path_str}')
        """).fetchone()[0]

        # Clean up donor temp files
        if donors_dir.exists():
            shutil.rmtree(donors_dir)

        con.close()

        # Write manifest
        manifest_data = {
            "cycle": cycle,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "input_file": str(csv_path),
            "output_dir": str(final_output_dir),
            "processing_method": "pyarrow_partitioned",
            "stats": {
                "n_raw_contributions": n_raw_rows,
                "n_donors": final_count,
                "n_partitions": partitions_processed,
            },
            "columns_used": cols,
            "config": {
                "memory_limit": MEMORY_LIMIT,
                "num_partitions": NUM_PARTITIONS,
                "batch_size": PYARROW_BATCH_SIZE,
            },
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Cycle {cycle} complete: {final_count:,} donors written")

        return {
            "cycle": cycle,
            "status": "success",
            "row_count": final_count,
            "stats": manifest_data["stats"],
        }

    except Exception as e:
        logger.error(f"Error processing cycle {cycle}: {e}")
        import traceback
        traceback.print_exc()

        # Clean up partial outputs
        for d in [partition_dir, donors_dir, final_output_dir]:
            if d.exists():
                shutil.rmtree(d)

        return {
            "cycle": cycle,
            "status": "error",
            "error": str(e),
        }
