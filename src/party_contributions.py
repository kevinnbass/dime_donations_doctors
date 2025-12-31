"""
Party Contributions Module.

Computes p.to.rep (percentage of contributions to Republicans) for each donor
using itemized contribution data.

This replicates the methodology from Bonica et al. (2019):
    p.to.rep = Σ(amount to Republicans) / Σ(amount to Republicans + Democrats)

Party codes in DIME:
    - '100' = Democrat
    - '200' = Republican
"""

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from .config import (
    DIME_ITEMIZED_TMP_DIR,
    DIME_RECIPIENTS_PARQUET,
    PROCESSED_DATA_DIR,
)


# Default cycles for Bonica replication (1992-2016)
BONICA_CYCLES = list(range(1992, 2017, 2))

# All available cycles
ALL_CYCLES = list(range(1980, 2025, 2))


def get_available_itemized_cycles() -> list:
    """Get list of cycles with downloaded itemized data."""
    available = []
    for f in DIME_ITEMIZED_TMP_DIR.glob("contribDB_*.csv.gz"):
        try:
            cycle = int(f.stem.replace("contribDB_", ""))
            available.append(cycle)
        except ValueError:
            pass
    return sorted(available)


def compute_donor_party_shares(
    cycles: Optional[list] = None,
    output_path: Optional[Path] = None,
    chunk_size: int = 5,
) -> pd.DataFrame:
    """
    Compute p.to.rep for all donors across specified cycles.

    This replicates Bonica's methodology:
    1. Sum contributions to Republican recipients (party='200')
    2. Sum contributions to Democratic recipients (party='100')
    3. Compute p.to.rep = rep_dollars / (rep_dollars + dem_dollars)

    Args:
        cycles: List of election cycles to include. Defaults to available cycles.
        output_path: Optional path to save results as parquet.
        chunk_size: Number of cycles to process at once (memory management).

    Returns:
        DataFrame with columns:
        - bonica_cid: Donor ID
        - rep_dollars: Total $ to Republicans
        - dem_dollars: Total $ to Democrats
        - total_partisan: Total $ to R or D
        - p_to_rep: Percentage to Republicans (0-1)
        - n_cycles: Number of cycles with contributions
    """
    # Get available cycles
    available = get_available_itemized_cycles()
    if not available:
        raise FileNotFoundError(
            f"No itemized files found in {DIME_ITEMIZED_TMP_DIR}\n"
            "Run: python scripts/download_itemized.py --all"
        )

    if cycles is None:
        cycles = available
    else:
        # Filter to available
        missing = [c for c in cycles if c not in available]
        if missing:
            print(f"Warning: Missing cycles: {missing}")
        cycles = [c for c in cycles if c in available]

    if not cycles:
        raise ValueError("No valid cycles to process")

    print(f"Processing {len(cycles)} cycles: {min(cycles)}-{max(cycles)}")
    print(f"Recipient data: {DIME_RECIPIENTS_PARQUET}")

    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")

    # Process in chunks to manage memory
    all_results = []

    for i in range(0, len(cycles), chunk_size):
        chunk_cycles = cycles[i:i + chunk_size]
        print(f"\nProcessing chunk: {chunk_cycles}")

        # Build list of files for this chunk
        files = [
            str(DIME_ITEMIZED_TMP_DIR / f"contribDB_{c}.csv.gz").replace("\\", "/")
            for c in chunk_cycles
        ]
        files_pattern = "', '".join(files)

        # Query to compute party shares
        query = f"""
        WITH contribs AS (
            SELECT
                CAST("bonica.cid" AS VARCHAR) as bonica_cid,
                CAST("bonica.rid" AS VARCHAR) as bonica_rid,
                CAST("recipient.party" AS VARCHAR) as recipient_party,
                CAST(amount AS DOUBLE) as amount,
                cycle
            FROM read_csv(
                ['{files_pattern}'],
                all_varchar=true,
                ignore_errors=true
            )
            WHERE TRY_CAST(amount AS DOUBLE) > 0
        ),
        -- Join with recipients table to fill missing party info
        contribs_with_party AS (
            SELECT
                c.bonica_cid,
                c.amount,
                c.cycle,
                COALESCE(c.recipient_party, r.recipient_party) as party
            FROM contribs c
            LEFT JOIN read_parquet('{str(DIME_RECIPIENTS_PARQUET).replace(chr(92), "/")}') r
                ON c.bonica_rid = r.bonica_rid
            WHERE c.bonica_cid IS NOT NULL
              AND c.bonica_cid != ''
        ),
        -- Aggregate by donor
        donor_party_totals AS (
            SELECT
                bonica_cid,
                SUM(CASE WHEN party = '200' THEN amount ELSE 0 END) as rep_dollars,
                SUM(CASE WHEN party = '100' THEN amount ELSE 0 END) as dem_dollars,
                SUM(CASE WHEN party IN ('100', '200') THEN amount ELSE 0 END) as total_partisan,
                COUNT(DISTINCT cycle) as n_cycles
            FROM contribs_with_party
            GROUP BY bonica_cid
            HAVING total_partisan > 0
        )
        SELECT
            bonica_cid,
            rep_dollars,
            dem_dollars,
            total_partisan,
            n_cycles,
            rep_dollars * 1.0 / total_partisan as p_to_rep
        FROM donor_party_totals
        """

        chunk_result = con.execute(query).fetchdf()
        print(f"  Chunk donors: {len(chunk_result):,}")
        all_results.append(chunk_result)

    con.close()

    # Combine chunks - need to re-aggregate across chunks
    if len(all_results) == 1:
        result = all_results[0]
    else:
        print("\nCombining chunks...")
        combined = pd.concat(all_results, ignore_index=True)

        # Re-aggregate by donor (sum across chunks)
        result = combined.groupby('bonica_cid').agg({
            'rep_dollars': 'sum',
            'dem_dollars': 'sum',
            'total_partisan': 'sum',
            'n_cycles': 'sum',
        }).reset_index()

        # Recompute p_to_rep
        result['p_to_rep'] = result['rep_dollars'] / result['total_partisan']

    print(f"\nTotal donors: {len(result):,}")
    print(f"Mean p.to.rep: {result['p_to_rep'].mean():.3f}")
    print(f"Median p.to.rep: {result['p_to_rep'].median():.3f}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")

    return result


def validate_against_bonica(
    our_data: pd.DataFrame,
    bonica_path: Optional[Path] = None,
) -> dict:
    """
    Validate our computed p.to.rep against Bonica's pre-computed values.

    Args:
        our_data: DataFrame from compute_donor_party_shares
        bonica_path: Path to Bonica's data (npi_dime_signatories_merged.csv)

    Returns:
        Dictionary with validation statistics
    """
    if bonica_path is None:
        bonica_path = (
            PROCESSED_DATA_DIR.parent / "raw" / "bonica_replication" /
            "npi_dime_signatories_merged.csv"
        )

    if not bonica_path.exists():
        raise FileNotFoundError(f"Bonica data not found: {bonica_path}")

    # Load Bonica data
    bonica = pd.read_csv(bonica_path)
    bonica['dime_cid_clean'] = bonica['dime.cid'].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    bonica_valid = bonica[
        bonica['p.to.rep'].notna() & bonica['dime_cid_clean'].notna()
    ].copy()

    # Merge with our data
    merged = bonica_valid.merge(
        our_data,
        left_on='dime_cid_clean',
        right_on='bonica_cid',
        how='inner'
    )

    if len(merged) < 100:
        print(f"Warning: Only {len(merged)} matched donors")
        return {'n_matched': len(merged), 'valid': False}

    # Compute validation statistics
    correlation = merged['p.to.rep'].corr(merged['p_to_rep'])
    rmse = ((merged['p.to.rep'] - merged['p_to_rep']) ** 2).mean() ** 0.5
    mean_diff = merged['p_to_rep'].mean() - merged['p.to.rep'].mean()

    stats = {
        'n_bonica': len(bonica_valid),
        'n_ours': len(our_data),
        'n_matched': len(merged),
        'bonica_mean': merged['p.to.rep'].mean(),
        'our_mean': merged['p_to_rep'].mean(),
        'mean_diff': mean_diff,
        'correlation': correlation,
        'rmse': rmse,
        'valid': correlation > 0.95,  # High bar for same-cycle validation
    }

    print("\n=== Validation Results ===")
    print(f"Matched donors: {stats['n_matched']:,}")
    print(f"Bonica mean: {stats['bonica_mean']:.3f}")
    print(f"Our mean: {stats['our_mean']:.3f}")
    print(f"Difference: {stats['mean_diff']:+.3f}")
    print(f"Correlation: {stats['correlation']:.3f}")
    print(f"RMSE: {stats['rmse']:.3f}")
    print(f"Valid: {stats['valid']}")

    return stats


def compute_party_shares_by_period(
    periods: dict,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Compute p.to.rep for multiple time periods.

    Args:
        periods: Dict mapping period names to cycle lists
            e.g., {'bonica': [1992,1994,...,2016], 'update': [2018,2020,...]}
        output_dir: Directory to save period-specific results

    Returns:
        Dict mapping period names to DataFrames
    """
    results = {}

    for name, cycles in periods.items():
        print(f"\n{'='*60}")
        print(f"Computing p.to.rep for period: {name}")
        print(f"Cycles: {min(cycles)}-{max(cycles)}")
        print(f"{'='*60}")

        output_path = None
        if output_dir:
            output_path = output_dir / f"party_shares_{name}.parquet"

        results[name] = compute_donor_party_shares(
            cycles=cycles,
            output_path=output_path,
        )

    return results
