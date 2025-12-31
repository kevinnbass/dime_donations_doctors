#!/usr/bin/env python3
"""Script to update partitioned_ingest.py with DuckDB parallel partitioning support."""

import re

# Read the original file
with open('partitioned_ingest.py', 'r', encoding='utf-8') as f:
    content = f.read()

# New function to add before aggregate_partition
NEW_FUNCTION = '''
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

    csv_path_str = str(csv_path).replace("\\\\", "/")
    partition_dir_str = str(partition_dir).replace("\\\\", "/")
    temp_dir_str = str(temp_dir).replace("\\\\", "/")

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


'''

# Insert the new function before aggregate_partition
marker = 'def aggregate_partition('
if marker in content:
    content = content.replace(marker, NEW_FUNCTION + marker)
    print("Added partition_csv_with_duckdb function")
else:
    print("ERROR: Could not find aggregate_partition marker")
    exit(1)

# Now update process_cycle_partitioned to prefer uncompressed CSV and use DuckDB
OLD_PROCESS_START = '''    partition_dir = get_temp_partition_dir(cycle)
    donors_dir = get_temp_donors_dir(cycle)
    final_output_dir = output_dir / f"cycle={cycle}"
    manifest_path = output_dir / f"cycle_{cycle}_manifest.json"

    # Check if already exists'''

NEW_PROCESS_START = '''    partition_dir = get_temp_partition_dir(cycle)
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

    # Check if already exists'''

if OLD_PROCESS_START in content:
    content = content.replace(OLD_PROCESS_START, NEW_PROCESS_START)
    print("Updated process_cycle_partitioned to detect uncompressed CSV")
else:
    print("WARNING: Could not find process_cycle_partitioned marker")

# Update the partitioning step to use DuckDB for uncompressed CSV
OLD_PARTITION_STEP = '''        # Step 2: Partition CSV -> Parquet using PyArrow
        n_raw_rows = partition_csv_with_pyarrow(csv_path, partition_dir, cols)'''

NEW_PARTITION_STEP = '''        # Step 2: Partition CSV -> Parquet (DuckDB for uncompressed, PyArrow for gzip)
        if use_duckdb_partitioning:
            n_raw_rows = partition_csv_with_duckdb(csv_path, partition_dir, temp_dir)
        else:
            n_raw_rows = partition_csv_with_pyarrow(csv_path, partition_dir, cols)'''

if OLD_PARTITION_STEP in content:
    content = content.replace(OLD_PARTITION_STEP, NEW_PARTITION_STEP)
    print("Updated partitioning step to use DuckDB for uncompressed CSV")
else:
    print("WARNING: Could not find partitioning step marker")

# Update the log message
OLD_LOG = '''    logger.info(f"Processing cycle {cycle} from {csv_path.name} (partitioned)")'''
NEW_LOG = '''    method = "DuckDB parallel" if use_duckdb_partitioning else "PyArrow streaming"
    logger.info(f"Processing cycle {cycle} from {csv_path.name} ({method})")'''

if OLD_LOG in content:
    content = content.replace(OLD_LOG, NEW_LOG)
    print("Updated log message")
else:
    print("WARNING: Could not find log message marker")

# Write the updated file
with open('partitioned_ingest.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done! Updated partitioned_ingest.py")
