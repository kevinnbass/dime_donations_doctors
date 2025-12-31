#!/usr/bin/env python3
"""
Step 2: Build Physician Labels

Runs the physician identification pipeline:
1. Rule-based classification from occupation/employer text
2. NPPES record linkage
3. Optional ML classifier
4. Optional coverage diagnostics

Usage:
    python scripts/02_build_physician_labels.py [OPTIONS]

Options:
    --skip-linkage      Skip NPPES record linkage
    --no-ml             Disable ML classifier training
    --run-diagnostics   Run coverage diagnostics comparing rules vs NPPES
    --overwrite         Overwrite existing files
"""

import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ensure_directories, DIME_DONORS_PARQUET, NPPES_PHYSICIANS_PARQUET
from src.record_linkage import run_linkage_pipeline
from src.physician_classifier import build_physician_labels, get_physician_label_stats
from src.diagnostics import generate_coverage_report, compare_definitions


@click.command()
@click.option("--skip-linkage", is_flag=True, help="Skip NPPES record linkage")
@click.option("--no-ml", is_flag=True, help="Disable ML classifier training")
@click.option("--run-diagnostics", is_flag=True, help="Run coverage diagnostics comparing rules vs NPPES")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
@click.option("--sample-size", type=int, default=None, help="Sample size for testing")
def main(skip_linkage: bool, no_ml: bool, run_diagnostics: bool, overwrite: bool, sample_size: int):
    """Build physician classification labels for DIME donors."""
    print("=" * 60)
    print("DIME Physician Analysis - Physician Labeling")
    print("=" * 60)

    ensure_directories()

    # Check prerequisites
    if not DIME_DONORS_PARQUET.exists():
        print("Error: DIME donors parquet not found.")
        print("Run 01_ingest_data.py first.")
        sys.exit(1)

    # Step 1: Record linkage (if not skipped and NPPES is available)
    if not skip_linkage:
        if NPPES_PHYSICIANS_PARQUET.exists():
            print("\n--- Step 1: NPPES Record Linkage ---")
            run_linkage_pipeline(
                overwrite=overwrite,
                sample_size=sample_size,
            )
        else:
            print("\nSkipping record linkage: NPPES data not found.")
    else:
        print("\nSkipping record linkage: --skip-linkage flag set")

    # Step 2: Build labels
    print("\n--- Step 2: Building Physician Labels ---")
    build_physician_labels(
        use_ml=not no_ml,
        overwrite=overwrite,
    )

    # Print summary stats
    stats = get_physician_label_stats()
    if stats:
        print("\n--- Label Statistics ---")
        print(f"  Total donors:       {stats.get('n_total', 0):,}")
        print(f"  Naive matches:      {stats.get('n_naive', 0):,}")
        print(f"  Rule matches:       {stats.get('n_rule', 0):,}")
        print(f"  NPPES linked:       {stats.get('n_nppes', 0):,}")
        print(f"  Final physicians:   {stats.get('n_final', 0):,}")

    # Step 3: Run diagnostics (if requested)
    if run_diagnostics:
        print("\n--- Step 3: Coverage Diagnostics ---")
        try:
            report = generate_coverage_report()
            print("\nDiagnostics complete. See outputs/diagnostics/ for detailed reports.")

            # Also compare definitions
            print("\n--- Definition Comparison ---")
            compare_definitions()
        except Exception as e:
            print(f"\nWarning: Diagnostics failed: {e}")
            print("This may occur if NPPES linkage was skipped.")

    print("\n" + "=" * 60)
    print("Physician labeling complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
