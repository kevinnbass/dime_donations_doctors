"""
Ideology computation for donor-cycle panel.

Computes both static (DIME CFScore) and cycle-specific
revealed ideology scores for donors.

Dual ideology measures:
- revealed_cfscore_cycle: Weighted mean of recipient CFScores (may have look-ahead bias)
- revealed_party_cycle: Weighted mean of party scores (+1 DEM, -1 REP, 0 other) - no look-ahead

External recipient ideology override:
- Config file: config/recipient_ideology_override.yaml
- Allows plugging in cycle-specific ideology (e.g., DW-NOMINATE)
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import yaml

from .config import (
    CONFIG_DIR,
    CYCLES,
    CYCLE_IDEOLOGY_PARQUET,
    DEFAULT_MIN_CYCLE_AMOUNT,
    DEFAULT_MIN_RECIPIENTS,
    DIME_DONORS_PARQUET,
    DIME_RAW_DIR,
    DIME_RECIPIENTS_PARQUET,
    DIME_TEMP_DIR,
    DONOR_CYCLE_PANEL_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    PROCESSED_DATA_DIR,
    SHRINKAGE_WEIGHT_FEW_RECIPIENTS,
    SHRINKAGE_WEIGHT_SINGLE_RECIPIENT,
)
from .dime_processing import find_dime_sqlite_file, decompress_gzip

logger = logging.getLogger(__name__)

# Recipient ideology override config path
RECIPIENT_IDEOLOGY_OVERRIDE_CONFIG = CONFIG_DIR / "recipient_ideology_override.yaml"


def compute_party_score(party: str) -> float:
    """
    Convert party affiliation to numeric score.

    Args:
        party: Party string (e.g., '100' for DEM, '200' for REP)

    Returns:
        +1.0 for Democrat, -1.0 for Republican, 0.0 for other/unknown
    """
    if party is None:
        return 0.0

    party_str = str(party).upper().strip()

    # DIME uses numeric codes: 100 = DEM, 200 = REP
    if party_str in ("100", "DEM", "D", "DEMOCRAT", "DEMOCRATIC"):
        return 1.0
    elif party_str in ("200", "REP", "R", "REPUBLICAN"):
        return -1.0
    else:
        return 0.0


def load_recipient_ideology_override() -> Optional[pd.DataFrame]:
    """
    Load external cycle-specific recipient ideology if configured.

    Config file format (config/recipient_ideology_override.yaml):
        enabled: true
        source: path/to/mapping.parquet
        format: parquet
        columns:
          recipient_id: bonica_rid
          cycle: cycle
          ideology: dw_nominate_score

    Returns:
        DataFrame with (bonica_rid, cycle, ideology_override) or None if not configured
    """
    if not RECIPIENT_IDEOLOGY_OVERRIDE_CONFIG.exists():
        return None

    try:
        with open(RECIPIENT_IDEOLOGY_OVERRIDE_CONFIG, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Error loading recipient ideology config: {e}")
        return None

    if not config or not config.get("enabled", False):
        return None

    source_path = Path(config.get("source", ""))
    if not source_path.exists():
        logger.warning(f"Recipient ideology source not found: {source_path}")
        return None

    file_format = config.get("format", "parquet").lower()
    col_config = config.get("columns", {})

    recipient_col = col_config.get("recipient_id", "bonica_rid")
    cycle_col = col_config.get("cycle", "cycle")
    ideology_col = col_config.get("ideology", "ideology")

    try:
        if file_format == "parquet":
            df = pd.read_parquet(source_path)
        elif file_format == "csv":
            df = pd.read_csv(source_path)
        else:
            logger.warning(f"Unknown format for recipient ideology: {file_format}")
            return None

        # Rename columns
        df = df.rename(columns={
            recipient_col: "bonica_rid",
            cycle_col: "cycle",
            ideology_col: "ideology_override",
        })

        return df[["bonica_rid", "cycle", "ideology_override"]]

    except Exception as e:
        logger.warning(f"Error loading recipient ideology override: {e}")
        return None


def compute_shrinkage_weight(n_recipients: int) -> float:
    """
    Compute shrinkage weight based on number of recipients.

    For donors with few recipients, shrink the cycle-specific
    score toward 0 (moderate).

    Args:
        n_recipients: Number of unique recipients in cycle

    Returns:
        Weight in [0, 1] for the observed score (remainder toward 0)
    """
    if n_recipients >= DEFAULT_MIN_RECIPIENTS:
        return 1.0
    elif n_recipients == 2:
        return SHRINKAGE_WEIGHT_FEW_RECIPIENTS
    elif n_recipients == 1:
        return SHRINKAGE_WEIGHT_SINGLE_RECIPIENT
    else:
        return 0.0


def compute_cycle_ideology(
    contributions_df: pd.DataFrame,
    recipients_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cycle-specific revealed ideology from contribution records.

    Computes DUAL ideology measures:
    - revealed_cfscore_cycle: Weighted mean of recipient CFScores
      (may have look-ahead bias since CFScores are estimated across years)
    - revealed_party_cycle: Weighted mean of party scores (+1 DEM, -1 REP, 0 other)
      (NO look-ahead bias since party is known at contribution time)

    Args:
        contributions_df: Contributions with columns:
            - bonica_cid: donor ID
            - bonica_rid: recipient ID
            - cycle: election cycle
            - amount: contribution amount
        recipients_df: Recipients with columns:
            - bonica_rid: recipient ID
            - recipient_cfscore: recipient CFScore
            - recipient_party (optional): party affiliation

    Returns:
        DataFrame with donor-cycle ideology scores including both CFScore and party measures
    """
    # Check if party column is available
    has_party = "recipient_party" in recipients_df.columns

    # Columns to select from recipients
    recipient_cols = ["bonica_rid", "recipient_cfscore"]
    if has_party:
        recipient_cols.append("recipient_party")

    # Merge to get recipient data
    merged = contributions_df.merge(
        recipients_df[recipient_cols],
        on="bonica_rid",
        how="left",
    )

    # Compute party score if party column exists
    if has_party:
        merged["party_score"] = merged["recipient_party"].apply(compute_party_score)
    else:
        # Try to infer from CFScore sign if party not available
        # CFScore > 0 = more conservative (REP), < 0 = more liberal (DEM)
        merged["party_score"] = np.where(
            merged["recipient_cfscore"].isna(),
            0.0,
            np.where(merged["recipient_cfscore"] > 0.5, -1.0,  # Conservative -> REP
                     np.where(merged["recipient_cfscore"] < -0.5, 1.0,  # Liberal -> DEM
                              0.0))  # Moderate -> unknown
        )

    # Filter to contributions with valid amounts
    merged_valid = merged[merged["amount"] > 0].copy()

    # Create separate dataframes for CFScore and party calculations
    # CFScore calculation requires non-null CFScores
    merged_cfscore = merged_valid[merged_valid["recipient_cfscore"].notna()].copy()

    # Party calculation can use all contributions (party_score is always defined)
    merged_party = merged_valid.copy()

    # Group by donor-cycle for CFScore-based ideology
    if len(merged_cfscore) > 0:
        grouped_cfscore = merged_cfscore.groupby(["bonica_cid", "cycle"]).apply(
            lambda g: pd.Series({
                "amount_total_cfscore": g["amount"].sum(),
                "n_recipients_cfscore": g["bonica_rid"].nunique(),
                "n_contributions_cfscore": len(g),
                "revealed_cfscore_cycle": np.average(
                    g["recipient_cfscore"], weights=g["amount"]
                ),
            }),
            include_groups=False,
        ).reset_index()
    else:
        grouped_cfscore = pd.DataFrame(columns=[
            "bonica_cid", "cycle", "amount_total_cfscore",
            "n_recipients_cfscore", "n_contributions_cfscore", "revealed_cfscore_cycle"
        ])

    # Group by donor-cycle for party-based ideology
    if len(merged_party) > 0:
        grouped_party = merged_party.groupby(["bonica_cid", "cycle"]).apply(
            lambda g: pd.Series({
                "amount_total": g["amount"].sum(),
                "n_recipients": g["bonica_rid"].nunique(),
                "n_contributions": len(g),
                "revealed_party_cycle": np.average(
                    g["party_score"], weights=g["amount"]
                ),
            }),
            include_groups=False,
        ).reset_index()
    else:
        grouped_party = pd.DataFrame(columns=[
            "bonica_cid", "cycle", "amount_total",
            "n_recipients", "n_contributions", "revealed_party_cycle"
        ])

    # Merge the two results
    if len(grouped_cfscore) > 0 and len(grouped_party) > 0:
        grouped = grouped_party.merge(
            grouped_cfscore[["bonica_cid", "cycle", "revealed_cfscore_cycle",
                            "n_recipients_cfscore", "n_contributions_cfscore"]],
            on=["bonica_cid", "cycle"],
            how="outer",
        )
    elif len(grouped_cfscore) > 0:
        grouped = grouped_cfscore.rename(columns={
            "amount_total_cfscore": "amount_total",
            "n_recipients_cfscore": "n_recipients",
            "n_contributions_cfscore": "n_contributions",
        })
        grouped["revealed_party_cycle"] = np.nan
    elif len(grouped_party) > 0:
        grouped = grouped_party.copy()
        grouped["revealed_cfscore_cycle"] = np.nan
        grouped["n_recipients_cfscore"] = np.nan
        grouped["n_contributions_cfscore"] = np.nan
    else:
        return pd.DataFrame()

    # Fill NaN for amounts/counts where only one measure was computed
    for col in ["amount_total", "n_recipients", "n_contributions"]:
        if col not in grouped.columns:
            grouped[col] = np.nan

    # Apply shrinkage for low-n donors (based on CFScore recipients count)
    n_recip_col = "n_recipients_cfscore" if "n_recipients_cfscore" in grouped.columns else "n_recipients"
    grouped["shrinkage_weight"] = grouped[n_recip_col].fillna(0).astype(int).apply(compute_shrinkage_weight)
    grouped["cfscore_cycle_shrunk"] = grouped["revealed_cfscore_cycle"] * grouped["shrinkage_weight"]
    grouped["party_cycle_shrunk"] = grouped["revealed_party_cycle"] * grouped["shrinkage_weight"]

    # Also keep the legacy column name for backward compatibility
    grouped["cfscore_cycle"] = grouped["revealed_cfscore_cycle"]

    return grouped


def build_donor_cycle_panel(
    output_path: Optional[Path] = None,
    min_amount: float = DEFAULT_MIN_CYCLE_AMOUNT,
    overwrite: bool = False,
) -> Path:
    """
    Build the complete donor-cycle panel with ideology scores.

    Creates a panel with:
    - Static CFScore (from DIME aggregate)
    - Cycle-specific revealed ideology (from contribution records)
    - Physician classification labels

    Args:
        output_path: Output parquet file path
        min_amount: Minimum total amount in cycle for inclusion
        overwrite: Whether to overwrite existing file

    Returns:
        Path to output file
    """
    output_path = output_path or DONOR_CYCLE_PANEL_PARQUET

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return output_path

    print("Building donor-cycle panel...")

    con = duckdb.connect()

    # Load donors with static CFScore
    dime_path = str(DIME_DONORS_PARQUET).replace("\\", "/")
    donors_df = con.execute(f"""
        SELECT
            bonica_cid,
            contributor_state,
            contributor_gender,
            contributor_cfscore as cfscore_static
        FROM read_parquet('{dime_path}')
    """).fetchdf()

    print(f"Loaded {len(donors_df):,} donors")

    # Load physician labels
    if PHYSICIAN_LABELS_PARQUET.exists():
        labels_path = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
        labels_df = con.execute(f"""
            SELECT bonica_cid, p_physician, physician_final, physician_naive, physician_rule_label
            FROM read_parquet('{labels_path}')
        """).fetchdf()
        print(f"Loaded {len(labels_df):,} physician labels")
    else:
        print("Warning: Physician labels not found. Run physician labeling first.")
        labels_df = None

    # Build cycle participation from the aggregate file amount columns
    print("Building cycle participation from aggregate amounts...")

    cycle_records = []
    amount_cols = [f'"amount.{year}"' for year in CYCLES]

    # Read and unpivot the amount columns
    for year in CYCLES:
        col = f"amount.{year}"
        cycle_df = con.execute(f"""
            SELECT
                bonica_cid,
                {year} as cycle,
                "{col}" as amount_total
            FROM read_parquet('{dime_path}')
            WHERE "{col}" > {min_amount}
        """).fetchdf()
        cycle_records.append(cycle_df)

    cycle_participation = pd.concat(cycle_records, ignore_index=True)
    print(f"Found {len(cycle_participation):,} donor-cycle records")

    # Try to compute cycle-specific ideology from SQLite if available
    try:
        sqlite_file = find_dime_sqlite_file()
        if sqlite_file.suffix == ".gz":
            sqlite_file = decompress_gzip(sqlite_file)

        print(f"Loading contribution records from {sqlite_file.name}...")

        # Connect to SQLite via DuckDB
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute(f"ATTACH '{str(sqlite_file).replace(chr(92), '/')}' AS dime_db (TYPE SQLITE, READ_ONLY)")

        # Check available tables
        tables = con.execute("SELECT name FROM dime_db.sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]
        print(f"Available tables: {table_names}")

        # Look for contribution table
        contrib_table = None
        for name in ["contributions", "contribs", "contribution", "contribDB"]:
            if name in table_names:
                contrib_table = name
                break

        if contrib_table:
            print(f"Using contribution table: {contrib_table}")

            # Load recipients
            recipients_path = PROCESSED_DATA_DIR / "dime_recipients.parquet"
            if recipients_path.exists():
                recipients_df = con.execute(f"""
                    SELECT bonica_rid, recipient_cfscore
                    FROM read_parquet('{str(recipients_path).replace(chr(92), "/")}')
                    WHERE recipient_cfscore IS NOT NULL
                """).fetchdf()

                # Query contributions
                contribs_df = con.execute(f"""
                    SELECT
                        bonica_cid,
                        bonica_rid,
                        cycle,
                        amount
                    FROM dime_db.{contrib_table}
                    WHERE amount > 0
                      AND cycle >= {CYCLES[0]}
                      AND cycle <= {CYCLES[-1]}
                """).fetchdf()

                print(f"Loaded {len(contribs_df):,} contribution records")

                # Compute cycle-specific ideology
                cycle_ideology = compute_cycle_ideology(contribs_df, recipients_df)
                print(f"Computed cycle ideology for {len(cycle_ideology):,} donor-cycles")
            else:
                print("Recipients file not found. Skipping cycle-specific ideology.")
                cycle_ideology = None
        else:
            print("Contribution table not found in SQLite. Skipping cycle-specific ideology.")
            cycle_ideology = None

    except FileNotFoundError:
        print("SQLite file not found. Using aggregate amounts only.")
        cycle_ideology = None
    except Exception as e:
        print(f"Error loading SQLite: {e}. Using aggregate amounts only.")
        cycle_ideology = None

    con.close()

    # Merge everything together
    print("Merging panel data...")

    panel = cycle_participation.merge(
        donors_df,
        on="bonica_cid",
        how="left",
    )

    if labels_df is not None:
        panel = panel.merge(
            labels_df,
            on="bonica_cid",
            how="left",
        )
    else:
        panel["p_physician"] = np.nan
        panel["physician_final"] = False
        panel["physician_naive"] = False
        panel["physician_rule_label"] = False

    if cycle_ideology is not None:
        panel = panel.merge(
            cycle_ideology[["bonica_cid", "cycle", "cfscore_cycle", "cfscore_cycle_shrunk", "n_recipients"]],
            on=["bonica_cid", "cycle"],
            how="left",
        )
    else:
        panel["cfscore_cycle"] = np.nan
        panel["cfscore_cycle_shrunk"] = np.nan
        panel["n_recipients"] = np.nan

    # Save to parquet
    panel.to_parquet(output_path, compression="zstd", index=False)

    print(f"Created: {output_path}")
    print(f"  Total donor-cycle records: {len(panel):,}")
    print(f"  Unique donors: {panel['bonica_cid'].nunique():,}")
    print(f"  Cycles: {sorted(panel['cycle'].unique())}")

    return output_path


def get_panel_stats() -> dict:
    """Get statistics from the donor-cycle panel."""
    if not DONOR_CYCLE_PANEL_PARQUET.exists():
        return {}

    con = duckdb.connect()
    path = str(DONOR_CYCLE_PANEL_PARQUET).replace("\\", "/")

    stats = {}
    stats["n_records"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_donors"] = con.execute(
        f"SELECT COUNT(DISTINCT bonica_cid) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_physician_records"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}') WHERE physician_final = true"
    ).fetchone()[0]

    # Records with cycle-specific ideology
    stats["n_with_cycle_ideology"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}') WHERE cfscore_cycle IS NOT NULL"
    ).fetchone()[0]

    con.close()
    return stats


def get_physician_cycle_data(
    physician_definition: str = "final",
    cycles: Optional[list[int]] = None,
    include_party: bool = True,
) -> pd.DataFrame:
    """
    Get physician donor data by cycle for a given physician definition.

    Args:
        physician_definition: One of 'naive', 'rule', 'final'
        cycles: Optional list of cycles to include
        include_party: Whether to include party-based ideology columns

    Returns:
        DataFrame with cycle, cfscore, party ideology, and counts
    """
    if not DONOR_CYCLE_PANEL_PARQUET.exists():
        raise FileNotFoundError("Donor-cycle panel not found. Run build_donor_cycle_panel first.")

    con = duckdb.connect()
    path = str(DONOR_CYCLE_PANEL_PARQUET).replace("\\", "/")

    # Map definition to column
    col_map = {
        "naive": "physician_naive",
        "rule": "physician_rule_label",
        "final": "physician_final",
    }
    physician_col = col_map.get(physician_definition, "physician_final")

    # Build query with optional party columns
    party_cols = ""
    if include_party:
        party_cols = """,
            revealed_party_cycle,
            party_cycle_shrunk"""

    # Build query
    cycle_filter = ""
    if cycles:
        cycle_list = ", ".join(str(c) for c in cycles)
        cycle_filter = f"AND cycle IN ({cycle_list})"

    query = f"""
        SELECT
            cycle,
            cfscore_static,
            cfscore_cycle,
            cfscore_cycle_shrunk,
            revealed_cfscore_cycle,
            amount_total,
            p_physician{party_cols}
        FROM read_parquet('{path}')
        WHERE {physician_col} = true
          AND cfscore_static IS NOT NULL
          {cycle_filter}
    """

    result = con.execute(query).fetchdf()
    con.close()

    return result


def build_cycle_ideology_streaming(
    output_path: Optional[Path] = None,
    cycles: Optional[list[int]] = None,
    overwrite: bool = False,
    keep_files: bool = False,
) -> Path:
    """
    Build cycle-specific ideology by streaming through cycle contribution files.

    Downloads each cycle file, processes it, extracts ideology scores,
    then deletes the file to minimize disk usage.

    Args:
        output_path: Output parquet file for cycle ideology
        cycles: Optional list of cycles to process (default: all)
        overwrite: Whether to overwrite existing output
        keep_files: Keep downloaded files instead of deleting

    Returns:
        Path to cycle ideology parquet file
    """
    from .download import (
        download_cycle_file,
        download_recipients,
        delete_cycle_file,
        get_available_cycles,
    )
    from .dime_processing import ingest_dime_recipients

    output_path = output_path or CYCLE_IDEOLOGY_PARQUET

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return output_path

    cycles_to_process = cycles or CYCLES
    available = get_available_cycles()
    cycles_to_process = [c for c in cycles_to_process if c in available]

    if not cycles_to_process:
        raise ValueError("No cycles available for download")

    print(f"Will process {len(cycles_to_process)} cycles: {cycles_to_process[0]}..{cycles_to_process[-1]}")

    # Ensure recipients file is available
    if not DIME_RECIPIENTS_PARQUET.exists():
        print("\n--- Downloading and processing recipients file ---")
        try:
            download_recipients()
            ingest_dime_recipients()
        except Exception as e:
            raise RuntimeError(f"Failed to get recipients file: {e}")

    # Load recipients once
    con = duckdb.connect()
    recipients_path = str(DIME_RECIPIENTS_PARQUET).replace("\\", "/")
    recipients_df = con.execute(f"""
        SELECT bonica_rid, recipient_cfscore
        FROM read_parquet('{recipients_path}')
        WHERE recipient_cfscore IS NOT NULL
    """).fetchdf()
    print(f"Loaded {len(recipients_df):,} recipients with CFScores")
    con.close()

    # Process each cycle
    all_results = []
    DIME_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for i, year in enumerate(cycles_to_process):
        print(f"\n--- Processing cycle {year} ({i+1}/{len(cycles_to_process)}) ---")

        try:
            # Download cycle file
            cycle_file = download_cycle_file(year, output_dir=DIME_TEMP_DIR)

            # Process with DuckDB
            con = duckdb.connect()
            cycle_path = str(cycle_file).replace("\\", "/")

            print(f"  Loading contributions...")
            contribs_df = con.execute(f"""
                SELECT
                    bonica_cid,
                    bonica_rid,
                    {year} as cycle,
                    amount
                FROM read_csv_auto('{cycle_path}',
                                   header=true,
                                   compression='gzip',
                                   sample_size=100000)
                WHERE amount > 0
            """).fetchdf()

            con.close()

            print(f"  Loaded {len(contribs_df):,} contributions")

            if len(contribs_df) > 0:
                # Compute cycle ideology
                cycle_ideology = compute_cycle_ideology(contribs_df, recipients_df)
                cycle_ideology["cycle"] = year
                all_results.append(cycle_ideology)
                print(f"  Computed ideology for {len(cycle_ideology):,} donors")

            # Delete cycle file to save space
            if not keep_files:
                delete_cycle_file(year, output_dir=DIME_TEMP_DIR)

        except Exception as e:
            print(f"  Error processing cycle {year}: {e}")
            continue

    if not all_results:
        raise RuntimeError("No cycle data was processed successfully")

    # Combine all results
    print("\n--- Combining results ---")
    combined = pd.concat(all_results, ignore_index=True)
    print(f"Total donor-cycle records: {len(combined):,}")

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, compression="zstd", index=False)
    print(f"Created: {output_path}")

    return output_path


def build_donor_cycle_panel_with_streaming(
    output_path: Optional[Path] = None,
    min_amount: float = DEFAULT_MIN_CYCLE_AMOUNT,
    download_cycles: bool = False,
    cycles: Optional[list[int]] = None,
    overwrite: bool = False,
) -> Path:
    """
    Build donor-cycle panel, optionally downloading cycle files for ideology.

    This is an enhanced version of build_donor_cycle_panel that can
    automatically download and process cycle contribution files.

    Args:
        output_path: Output parquet file path
        min_amount: Minimum total amount in cycle for inclusion
        download_cycles: Whether to download cycle files for cycle-specific ideology
        cycles: Optional list of cycles to process (e.g., [2020, 2022, 2024])
        overwrite: Whether to overwrite existing files

    Returns:
        Path to output file
    """
    output_path = output_path or DONOR_CYCLE_PANEL_PARQUET

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return output_path

    # If download_cycles is requested, first build cycle ideology
    if download_cycles:
        print("=" * 60)
        print("PHASE 1: Building cycle-specific ideology from downloads")
        print("=" * 60)

        if not CYCLE_IDEOLOGY_PARQUET.exists() or overwrite:
            build_cycle_ideology_streaming(cycles=cycles, overwrite=overwrite)
        else:
            print(f"Using existing cycle ideology: {CYCLE_IDEOLOGY_PARQUET}")

    print("\n" + "=" * 60)
    print("PHASE 2: Building donor-cycle panel")
    print("=" * 60)

    con = duckdb.connect()

    # Load donors with static CFScore
    dime_path = str(DIME_DONORS_PARQUET).replace("\\", "/")
    donors_df = con.execute(f"""
        SELECT
            bonica_cid,
            contributor_state,
            contributor_gender,
            contributor_cfscore as cfscore_static
        FROM read_parquet('{dime_path}')
    """).fetchdf()

    print(f"Loaded {len(donors_df):,} donors")

    # Load physician labels
    if PHYSICIAN_LABELS_PARQUET.exists():
        labels_path = str(PHYSICIAN_LABELS_PARQUET).replace("\\", "/")
        labels_df = con.execute(f"""
            SELECT bonica_cid, p_physician, physician_final, physician_naive, physician_rule_label
            FROM read_parquet('{labels_path}')
        """).fetchdf()
        print(f"Loaded {len(labels_df):,} physician labels")
    else:
        print("Warning: Physician labels not found.")
        labels_df = None

    # Build cycle participation from aggregate amounts
    print("Building cycle participation...")
    cycle_records = []
    for year in CYCLES:
        col = f"amount.{year}"
        cycle_df = con.execute(f"""
            SELECT
                bonica_cid,
                {year} as cycle,
                "{col}" as amount_total
            FROM read_parquet('{dime_path}')
            WHERE "{col}" > {min_amount}
        """).fetchdf()
        cycle_records.append(cycle_df)

    cycle_participation = pd.concat(cycle_records, ignore_index=True)
    print(f"Found {len(cycle_participation):,} donor-cycle records")

    # Load cycle ideology if available
    if CYCLE_IDEOLOGY_PARQUET.exists():
        ideology_path = str(CYCLE_IDEOLOGY_PARQUET).replace("\\", "/")
        cycle_ideology = con.execute(f"""
            SELECT bonica_cid, cycle, cfscore_cycle, cfscore_cycle_shrunk, n_recipients
            FROM read_parquet('{ideology_path}')
        """).fetchdf()
        print(f"Loaded {len(cycle_ideology):,} cycle ideology records")
    else:
        cycle_ideology = None

    con.close()

    # Merge everything
    print("Merging panel data...")
    panel = cycle_participation.merge(donors_df, on="bonica_cid", how="left")

    if labels_df is not None:
        panel = panel.merge(labels_df, on="bonica_cid", how="left")
    else:
        panel["p_physician"] = np.nan
        panel["physician_final"] = False
        panel["physician_naive"] = False
        panel["physician_rule_label"] = False

    if cycle_ideology is not None:
        panel = panel.merge(
            cycle_ideology,
            on=["bonica_cid", "cycle"],
            how="left",
        )
    else:
        panel["cfscore_cycle"] = np.nan
        panel["cfscore_cycle_shrunk"] = np.nan
        panel["n_recipients"] = np.nan

    # Save
    panel.to_parquet(output_path, compression="zstd", index=False)

    print(f"\nCreated: {output_path}")
    print(f"  Total records: {len(panel):,}")
    print(f"  Unique donors: {panel['bonica_cid'].nunique():,}")
    print(f"  With cycle ideology: {panel['cfscore_cycle'].notna().sum():,}")

    return output_path
