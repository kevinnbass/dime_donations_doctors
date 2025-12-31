"""
Fast DuckDB-based physician classification.

This module provides a high-performance alternative to row-by-row classification.
"""

import time
from pathlib import Path

import duckdb

from .config import (
    DEFAULT_PHYSICIAN_THRESHOLD,
    DIME_DONORS_PARQUET,
    LINKAGE_RESULTS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
)


def build_physician_labels_fast(
    output_path=None,
    threshold: float = DEFAULT_PHYSICIAN_THRESHOLD,
) -> Path:
    """
    Build physician labels using pure DuckDB SQL for maximum speed.

    This replaces the slow row-by-row Python iteration with vectorized
    SQL operations that run 50-100x faster.
    """
    start_time = time.time()

    output_path = output_path or PHYSICIAN_LABELS_PARQUET

    print("Building physician labels (FAST DuckDB mode)...")

    con = duckdb.connect()

    # Get file paths
    dime_path = str(DIME_DONORS_PARQUET).replace("\\", "/")
    linkage_path = str(LINKAGE_RESULTS_PARQUET).replace("\\", "/")
    output_path_str = str(output_path).replace("\\", "/")

    # Check if linkage results exist
    has_linkage = LINKAGE_RESULTS_PARQUET.exists()
    print(f"  NPPES linkage available: {has_linkage}")

    # Count donors
    n_donors = con.execute(f"SELECT COUNT(*) FROM read_parquet('{dime_path}')").fetchone()[0]
    print(f"  Total donors: {n_donors:,}")

    # Build the SQL query components
    linkage_join = ""
    linkage_select = "false as has_nppes"
    if has_linkage:
        n_linked = con.execute(f"SELECT COUNT(DISTINCT bonica_cid) FROM read_parquet('{linkage_path}')").fetchone()[0]
        print(f"  NPPES linked donors: {n_linked:,}")
        linkage_join = f"""
        LEFT JOIN (
            SELECT DISTINCT bonica_cid
            FROM read_parquet('{linkage_path}')
        ) l ON d.bonica_cid = l.bonica_cid
        """
        linkage_select = "(l.bonica_cid IS NOT NULL) as has_nppes"

    # Tier 1 specialties pattern
    tier1_specialties = (
        "surgeon|cardiologist|neurologist|oncologist|dermatologist|radiologist|"
        "anesthesiologist|pathologist|psychiatrist|urologist|ophthalmologist|"
        "gastroenterologist|pulmonologist|nephrologist|rheumatologist|endocrinologist|"
        "hematologist|hospitalist|intensivist|internist|obstetrician|gynecologist|"
        "neonatologist|geriatrician|allergist|physiatrist|otolaryngologist|hepatologist"
    )

    # Tier 1 credentials pattern
    tier1_credentials = "m\\.?d\\.?|d\\.?o\\.?|mbbs|facp|facs|facc|facep|faap|facog|diplomate"

    # Tier 2 terms pattern
    tier2_terms = (
        "physician|medical doctor|family practice|family medicine|internal medicine|"
        "primary care|pediatrician|resident physician|chief resident|attending|"
        "board certified|private practice|medical fellow"
    )

    # Tier 3 exclusions
    tier3_doctorates = "ph\\.?d|ed\\.?d|psy\\.?d"
    tier3_healthcare = (
        "chiropract|dentist|dental|dds|dmd|podiatr|optometr|veterinar|pharmacist|"
        "pharm\\.?d|nurse practitioner|aprn|physician assistant|pa-c|psycholog|lcsw|"
        "physical therap|occupational therap|paramedic|emt|medical assistant"
    )
    tier3_occupations = (
        "lawyer|attorney|engineer|accountant|teacher|software|programmer|realtor|"
        "salesperson|writer|author|artist|musician|banker|homemaker"
    )

    # Full classification query
    query = f"""
    COPY (
        WITH donors_with_features AS (
            SELECT
                d.bonica_cid,
                LOWER(COALESCE(d.contributor_occupation, '')) as occ,
                LOWER(COALESCE(d.contributor_employer, '')) as emp,
                COALESCE(d.contributor_name, '') as name,
                LOWER(COALESCE(d.contributor_occupation, '') || ' ' || COALESCE(d.contributor_employer, '')) as occ_emp,
                {linkage_select}
            FROM read_parquet('{dime_path}') d
            {linkage_join}
        ),
        classified AS (
            SELECT
                bonica_cid, occ, emp, name, has_nppes,

                -- Naive keyword match
                (occ LIKE '%physician%' OR occ LIKE '%doctor%' OR occ LIKE '%surgeon%') as physician_naive,

                -- Tier 1: High-confidence patterns
                (
                    regexp_matches(occ_emp, '{tier1_specialties}')
                    OR regexp_matches(occ_emp, '{tier1_credentials}')
                    OR regexp_matches(LOWER(name), ',\\s*m\\.?d|,\\s*d\\.?o|,\\s*facp|,\\s*facs')
                    OR regexp_matches(occ_emp, 'chief medical officer|medical director|chief of surgery|chief of medicine')
                ) as tier1_match,

                -- Tier 2: Likely positive
                (
                    regexp_matches(occ_emp, '{tier2_terms}')
                ) as tier2_match,

                -- Tier 3: Exclusions
                (
                    regexp_matches(occ_emp, '{tier3_doctorates}')
                    OR regexp_matches(occ_emp, '{tier3_healthcare}')
                    OR regexp_matches(occ_emp, '{tier3_occupations}')
                    OR regexp_matches(LOWER(name), ',\\s*ph\\.?d|,\\s*d\\.?d\\.?s|,\\s*j\\.?d|,\\s*esq')
                ) as tier3_exclusion

            FROM donors_with_features
        )
        SELECT
            bonica_cid,
            physician_naive,
            (tier1_match AND NOT tier3_exclusion) as physician_rule_label,
            CASE
                WHEN tier3_exclusion THEN 'tier3_exclusion'
                WHEN tier1_match THEN 'tier1'
                WHEN tier2_match THEN 'tier2'
                ELSE 'no_match'
            END as physician_rule_tier,
            CASE
                WHEN tier3_exclusion THEN 0.0
                WHEN tier1_match THEN 1.0
                WHEN tier2_match THEN 0.7
                ELSE 0.3
            END::DOUBLE as physician_rule_score,
            has_nppes as physician_nppes_linked,
            CASE
                WHEN tier3_exclusion THEN 0.0
                WHEN tier1_match THEN 1.0
                WHEN has_nppes THEN 0.9
                WHEN tier2_match THEN 0.7
                ELSE 0.3
            END::DOUBLE as p_physician,
            CASE
                WHEN tier3_exclusion THEN false
                WHEN tier1_match OR has_nppes THEN true
                WHEN tier2_match THEN true
                ELSE false
            END as physician_final
        FROM classified
    ) TO '{output_path_str}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    print("  Running classification query...")
    con.execute(query)

    # Get summary statistics
    print("  Computing statistics...")
    stats = con.execute(f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN physician_naive THEN 1 ELSE 0 END) as naive_matches,
            SUM(CASE WHEN physician_rule_label THEN 1 ELSE 0 END) as rule_physicians,
            SUM(CASE WHEN physician_nppes_linked THEN 1 ELSE 0 END) as nppes_linked,
            SUM(CASE WHEN physician_final THEN 1 ELSE 0 END) as final_physicians
        FROM read_parquet('{output_path_str}')
    """).fetchone()

    con.close()

    elapsed = time.time() - start_time

    print(f"\nCreated: {output_path}")
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"\nPhysician classification summary:")
    print(f"  Total donors: {stats[0]:,}")
    print(f"  Naive keyword matches: {stats[1]:,}")
    print(f"  Rule-based physicians: {stats[2]:,}")
    print(f"  NPPES-linked: {stats[3]:,}")
    print(f"  Final physicians: {stats[4]:,}")

    return output_path
