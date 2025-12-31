"""
Record linkage between DIME donors and NPPES physicians.

Uses blocking strategies and fuzzy string matching to identify
DIME donors who are registered physicians in NPPES.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

import duckdb
from rapidfuzz import fuzz

from .config import (
    DEFAULT_LINKAGE_THRESHOLD,
    DIME_DONORS_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    NPPES_PHYSICIANS_PARQUET,
)
from .text_normalization import (
    extract_name_parts,
    get_zip_prefix,
    normalize_city,
    normalize_name,
    normalize_state,
    normalize_zip,
    soundex,
)


@dataclass
class LinkageCandidate:
    """A candidate match between a DIME donor and NPPES physician."""

    dime_id: str
    nppes_npi: str
    name_score: float
    city_score: float
    combined_score: float
    match_details: dict


def compute_name_similarity(
    dime_first: str,
    dime_last: str,
    nppes_first: str,
    nppes_last: str,
) -> float:
    """
    Compute name similarity score between DIME and NPPES records.

    Uses weighted combination of first and last name similarity.

    Args:
        dime_first: DIME first name (normalized)
        dime_last: DIME last name (normalized)
        nppes_first: NPPES first name (normalized)
        nppes_last: NPPES last name (normalized)

    Returns:
        Similarity score in [0, 1]
    """
    if not dime_last or not nppes_last:
        return 0.0

    # Last name is more important
    last_score = fuzz.ratio(dime_last, nppes_last) / 100.0

    # First name comparison
    if dime_first and nppes_first:
        first_score = fuzz.ratio(dime_first, nppes_first) / 100.0
        # Also check for initial match
        if len(dime_first) == 1 or len(nppes_first) == 1:
            if dime_first[0] == nppes_first[0]:
                first_score = max(first_score, 0.7)
    else:
        first_score = 0.0

    # Weight: 60% last name, 40% first name
    return 0.6 * last_score + 0.4 * first_score


def compute_location_similarity(
    dime_city: str,
    dime_state: str,
    dime_zip: str,
    nppes_city: str,
    nppes_state: str,
    nppes_zip: str,
) -> float:
    """
    Compute location similarity score.

    Args:
        dime_*: DIME location fields
        nppes_*: NPPES location fields

    Returns:
        Similarity score in [0, 1]
    """
    score = 0.0

    # State match (must match for high score)
    if dime_state and nppes_state:
        if dime_state.upper() == nppes_state.upper():
            score += 0.3
        else:
            return 0.0  # State mismatch is disqualifying

    # ZIP code match
    if dime_zip and nppes_zip:
        dime_zip5 = normalize_zip(dime_zip)
        nppes_zip5 = normalize_zip(nppes_zip)
        if dime_zip5 == nppes_zip5:
            score += 0.4
        elif dime_zip5[:3] == nppes_zip5[:3]:
            score += 0.2

    # City match
    if dime_city and nppes_city:
        city_score = fuzz.ratio(
            normalize_city(dime_city),
            normalize_city(nppes_city),
        ) / 100.0
        score += 0.3 * city_score

    return min(score, 1.0)


def compute_match_score(
    dime_record: dict,
    nppes_record: dict,
) -> LinkageCandidate:
    """
    Compute overall match score between a DIME donor and NPPES physician.

    Args:
        dime_record: DIME donor record
        nppes_record: NPPES physician record

    Returns:
        LinkageCandidate with scores
    """
    # Extract name parts from DIME
    dime_name_parts = extract_name_parts(dime_record.get("contributor_name", ""))
    dime_first = dime_name_parts["first"]
    dime_last = dime_name_parts["last"]

    # NPPES names are already separated
    nppes_first = normalize_name(nppes_record.get("first_name_norm", ""))
    nppes_last = normalize_name(nppes_record.get("last_name_norm", ""))

    # Name similarity
    name_score = compute_name_similarity(
        dime_first, dime_last, nppes_first, nppes_last
    )

    # Location similarity
    location_score = compute_location_similarity(
        dime_record.get("contributor_city", ""),
        dime_record.get("contributor_state", ""),
        dime_record.get("contributor_zipcode", ""),
        nppes_record.get("city", ""),
        nppes_record.get("state_norm", ""),
        nppes_record.get("zip5", ""),
    )

    # Combined score (weighted)
    combined_score = 0.7 * name_score + 0.3 * location_score

    return LinkageCandidate(
        dime_id=dime_record.get("bonica_cid", ""),
        nppes_npi=nppes_record.get("npi", ""),
        name_score=name_score,
        city_score=location_score,
        combined_score=combined_score,
        match_details={
            "dime_name": dime_record.get("contributor_name", ""),
            "nppes_name": f"{nppes_record.get('first_name', '')} {nppes_record.get('last_name', '')}",
            "dime_location": f"{dime_record.get('contributor_city', '')}, {dime_record.get('contributor_state', '')}",
            "nppes_location": f"{nppes_record.get('city', '')}, {nppes_record.get('state_norm', '')}",
        },
    )


class RecordLinker:
    """
    Links DIME donors to NPPES physicians using blocking and fuzzy matching.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_LINKAGE_THRESHOLD,
        block_on_state: bool = True,
        block_on_zip_prefix: bool = True,
        block_on_soundex: bool = True,
    ):
        """
        Initialize the record linker.

        Args:
            threshold: Minimum combined score for a match
            block_on_state: Whether to require state match for blocking
            block_on_zip_prefix: Whether to use ZIP prefix for blocking
            block_on_soundex: Whether to use last name soundex for blocking
        """
        self.threshold = threshold
        self.block_on_state = block_on_state
        self.block_on_zip_prefix = block_on_zip_prefix
        self.block_on_soundex = block_on_soundex

    def find_matches(
        self,
        dime_record: dict,
        nppes_df,
    ) -> list[LinkageCandidate]:
        """
        Find NPPES matches for a single DIME record.

        Args:
            dime_record: DIME donor record
            nppes_df: NPPES DataFrame or DuckDB relation

        Returns:
            List of candidates above threshold, sorted by score descending
        """
        # Extract blocking keys from DIME record
        dime_state = normalize_state(dime_record.get("contributor_state", ""))
        dime_zip = normalize_zip(dime_record.get("contributor_zipcode", ""))
        dime_zip_prefix = get_zip_prefix(dime_zip, 3)

        name_parts = extract_name_parts(dime_record.get("contributor_name", ""))
        dime_last_soundex = soundex(name_parts["last"])

        # Build blocking query
        candidates = []

        # For each NPPES record matching blocking criteria
        for _, nppes_row in nppes_df.iterrows():
            # Check blocking criteria
            if self.block_on_state:
                if nppes_row.get("state_norm", "") != dime_state:
                    continue

            if self.block_on_zip_prefix and dime_zip_prefix:
                nppes_zip = nppes_row.get("zip5", "")
                if not nppes_zip.startswith(dime_zip_prefix):
                    continue

            if self.block_on_soundex and dime_last_soundex:
                if nppes_row.get("last_name_soundex", "") != dime_last_soundex:
                    continue

            # Compute match score
            candidate = compute_match_score(dime_record, nppes_row.to_dict())

            if candidate.combined_score >= self.threshold:
                candidates.append(candidate)

        # Sort by score descending
        candidates.sort(key=lambda c: c.combined_score, reverse=True)

        return candidates


def run_linkage_pipeline(
    output_path=None,
    threshold: float = DEFAULT_LINKAGE_THRESHOLD,
    sample_size: Optional[int] = None,
    overwrite: bool = False,
) -> dict:
    """
    Run the full linkage pipeline between DIME and NPPES.

    This uses SQL-based blocking for efficiency.

    Args:
        output_path: Output parquet file path
        threshold: Minimum match score
        sample_size: Optional limit for testing
        overwrite: Whether to overwrite existing output

    Returns:
        Dictionary with statistics
    """
    output_path = output_path or LINKAGE_RESULTS_PARQUET

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return {"status": "skipped", "path": str(output_path)}

    print("Running DIME-NPPES record linkage...")

    con = duckdb.connect()

    # Load data
    dime_path = str(DIME_DONORS_PARQUET).replace("\\", "/")
    nppes_path = str(NPPES_PHYSICIANS_PARQUET).replace("\\", "/")

    # Add normalized fields to DIME data for blocking
    # Pre-filter to potential physicians to reduce join size dramatically
    physician_keywords = (
        "LOWER(contributor_occupation) LIKE '%physician%' OR "
        "LOWER(contributor_occupation) LIKE '%doctor%' OR "
        "LOWER(contributor_occupation) LIKE '%surgeon%' OR "
        "LOWER(contributor_occupation) LIKE '%md%' OR "
        "LOWER(contributor_occupation) LIKE '%m.d.%' OR "
        "LOWER(contributor_occupation) LIKE '%medical%' OR "
        "LOWER(contributor_occupation) LIKE '%pediatr%' OR "
        "LOWER(contributor_occupation) LIKE '%cardiolog%' OR "
        "LOWER(contributor_occupation) LIKE '%oncolog%' OR "
        "LOWER(contributor_occupation) LIKE '%radiolog%' OR "
        "LOWER(contributor_occupation) LIKE '%anesthes%' OR "
        "LOWER(contributor_occupation) LIKE '%dermatolog%' OR "
        "LOWER(contributor_occupation) LIKE '%neurolog%' OR "
        "LOWER(contributor_occupation) LIKE '%psychiatr%' OR "
        "LOWER(contributor_occupation) LIKE '%patholog%' OR "
        "LOWER(contributor_occupation) LIKE '%orthoped%' OR "
        "LOWER(contributor_occupation) LIKE '%ophthalm%' OR "
        "LOWER(contributor_occupation) LIKE '%urolog%' OR "
        "LOWER(contributor_occupation) LIKE '%gastroenter%' OR "
        "LOWER(contributor_occupation) LIKE '%pulmonolog%' OR "
        "LOWER(contributor_occupation) LIKE '%internist%' OR "
        "LOWER(contributor_occupation) LIKE '%obstetr%' OR "
        "LOWER(contributor_occupation) LIKE '%gynecolog%' OR "
        "LOWER(contributor_employer) LIKE '%hospital%' OR "
        "LOWER(contributor_employer) LIKE '%clinic%' OR "
        "LOWER(contributor_employer) LIKE '%medical%'"
    )

    print("Preparing DIME data for linkage (pre-filtering to potential physicians)...")
    con.execute(f"""
        CREATE TABLE dime_for_linkage AS
        SELECT
            bonica_cid,
            contributor_name,
            contributor_city,
            contributor_state,
            contributor_zipcode,
            contributor_occupation,
            contributor_employer,
            UPPER(SUBSTRING(contributor_state, 1, 2)) as state_norm,
            SUBSTRING(REGEXP_REPLACE(CAST(contributor_zipcode AS VARCHAR), '[^0-9]', '', 'g'), 1, 5) as zip5,
            SUBSTRING(REGEXP_REPLACE(CAST(contributor_zipcode AS VARCHAR), '[^0-9]', '', 'g'), 1, 3) as zip3,
            -- Extract first letter of last name for blocking
            UPPER(SUBSTRING(REGEXP_EXTRACT(contributor_name, '([A-Za-z]+)[^A-Za-z]*$'), 1, 1)) as last_initial
        FROM read_parquet('{dime_path}')
        WHERE ({physician_keywords})
        {"LIMIT " + str(sample_size) if sample_size else ""}
    """)

    print("Preparing NPPES data for linkage...")
    con.execute(f"""
        CREATE TABLE nppes_for_linkage AS
        SELECT
            npi,
            first_name,
            last_name,
            first_name_norm,
            last_name_norm,
            city,
            city_norm,
            state_norm,
            zip5,
            SUBSTRING(zip5, 1, 3) as zip3,
            last_name_soundex
        FROM read_parquet('{nppes_path}')
    """)

    # Perform blocked join
    # This is a simplified version - for production, you'd want more sophisticated blocking
    print("Performing blocked join...")
    con.execute(f"""
        CREATE TABLE linkage_candidates AS
        SELECT
            d.bonica_cid,
            n.npi as nppes_npi,
            d.contributor_name as dime_name,
            n.first_name || ' ' || n.last_name as nppes_name,
            d.contributor_city as dime_city,
            n.city as nppes_city,
            d.state_norm as dime_state,
            n.state_norm as nppes_state,
            d.zip5 as dime_zip,
            n.zip5 as nppes_zip
        FROM dime_for_linkage d
        JOIN nppes_for_linkage n
            ON d.state_norm = n.state_norm
            AND d.zip5 = n.zip5
            AND UPPER(SUBSTRING(n.last_name, 1, 1)) = d.last_initial
    """)

    # Export results
    print("Exporting linkage candidates...")
    con.execute(f"""
        COPY linkage_candidates
        TO '{str(output_path).replace(chr(92), "/")}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Get statistics
    stats = con.execute("SELECT COUNT(*) FROM linkage_candidates").fetchone()
    n_candidates = stats[0] if stats else 0

    con.close()

    print(f"Created {output_path} with {n_candidates:,} linkage candidates")

    return {
        "status": "completed",
        "path": str(output_path),
        "n_candidates": n_candidates,
    }


def get_linkage_stats() -> dict:
    """Get statistics from the linkage results."""
    if not LINKAGE_RESULTS_PARQUET.exists():
        return {}

    con = duckdb.connect()
    path = str(LINKAGE_RESULTS_PARQUET).replace("\\", "/")

    stats = {}
    stats["n_candidates"] = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_unique_dime"] = con.execute(
        f"SELECT COUNT(DISTINCT bonica_cid) FROM read_parquet('{path}')"
    ).fetchone()[0]

    stats["n_unique_nppes"] = con.execute(
        f"SELECT COUNT(DISTINCT nppes_npi) FROM read_parquet('{path}')"
    ).fetchone()[0]

    con.close()
    return stats
