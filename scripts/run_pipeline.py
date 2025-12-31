#!/usr/bin/env python3
"""
Run Full Analysis Pipeline.

Executes all scripts in order to replicate the physician political contributions
analysis from raw DIME data.

PREREQUISITES:
1. Download DIME data from https://data.stanford.edu/dime
2. Place itemized contribution files in data/raw/dime/
3. (Optional) Download CMS Medicare data for ground-truth validation
4. (Optional) Download NPPES data for physician registry matching

Usage:
    python scripts/run_pipeline.py              # Run full pipeline
    python scripts/run_pipeline.py --skip-to 34 # Skip to script 34
    python scripts/run_pipeline.py --only 34    # Run only script 34
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# Pipeline steps in execution order
PIPELINE_STEPS = [
    ("00_streaming_ingest.py", "Download and process DIME contribution data"),
    ("01_ingest_data.py", "Ingest NPPES and taxonomy data"),
    ("02_build_physician_labels.py", "Build physician classification labels"),
    ("02b_add_cms_linkage.py", "Add CMS Medicare linkage (optional)"),
    ("02c_add_pecos_linkage.py", "Add PECOS enrollment data (optional)"),
    ("03_build_donor_panel.py", "Build donor-cycle panel"),
    ("34_yearly_contributions_by_pool.py", "Analyze yearly contributions by pool"),
    ("35_academic_physicians.py", "Analyze academic physicians"),
]

# Optional steps that can be skipped without breaking pipeline
OPTIONAL_STEPS = ["02b_add_cms_linkage.py", "02c_add_pecos_linkage.py"]


def run_script(script_name: str, scripts_dir: Path) -> bool:
    """Run a single script and return success status."""
    script_path = scripts_dir / script_name

    if not script_path.exists():
        print(f"  WARNING: Script not found: {script_path}")
        return False

    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=scripts_dir.parent,  # Run from project root
            check=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: Script failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the full physician contributions analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_pipeline.py              # Run full pipeline
    python scripts/run_pipeline.py --skip-to 34 # Skip to script 34
    python scripts/run_pipeline.py --only 34    # Run only script 34
    python scripts/run_pipeline.py --list       # List all steps
        """
    )
    parser.add_argument(
        "--skip-to",
        type=str,
        help="Skip to a specific script number (e.g., '34')"
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Run only a specific script (e.g., '34')"
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional steps (CMS, PECOS linkage)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all pipeline steps and exit"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next step even if current step fails"
    )

    args = parser.parse_args()

    # List steps and exit
    if args.list:
        print("\nPipeline Steps:")
        print("-" * 70)
        for script, description in PIPELINE_STEPS:
            optional = " (optional)" if script in OPTIONAL_STEPS else ""
            print(f"  {script:<35} - {description}{optional}")
        return

    # Determine scripts directory
    scripts_dir = Path(__file__).parent

    # Filter steps based on arguments
    steps_to_run = PIPELINE_STEPS.copy()

    if args.only:
        # Run only specific script
        steps_to_run = [(s, d) for s, d in PIPELINE_STEPS if args.only in s]
        if not steps_to_run:
            print(f"ERROR: No script found matching '{args.only}'")
            sys.exit(1)

    elif args.skip_to:
        # Skip to specific script
        found = False
        filtered = []
        for script, desc in PIPELINE_STEPS:
            if args.skip_to in script:
                found = True
            if found:
                filtered.append((script, desc))

        if not filtered:
            print(f"ERROR: No script found matching '{args.skip_to}'")
            sys.exit(1)
        steps_to_run = filtered

    if args.skip_optional:
        steps_to_run = [(s, d) for s, d in steps_to_run if s not in OPTIONAL_STEPS]

    # Print header
    print("\n" + "=" * 70)
    print("PHYSICIAN POLITICAL CONTRIBUTIONS - FULL PIPELINE")
    print("=" * 70)
    print(f"\nSteps to run: {len(steps_to_run)}")
    for script, _ in steps_to_run:
        print(f"  - {script}")
    print()

    # Run pipeline
    successful = 0
    failed = 0
    skipped = 0

    for script, description in steps_to_run:
        is_optional = script in OPTIONAL_STEPS

        success = run_script(script, scripts_dir)

        if success:
            successful += 1
            print(f"\n  SUCCESS: {script}")
        else:
            if is_optional:
                print(f"\n  SKIPPED (optional): {script}")
                skipped += 1
            elif args.continue_on_error:
                print(f"\n  FAILED (continuing): {script}")
                failed += 1
            else:
                print(f"\n  FAILED: {script}")
                print("\nPipeline stopped due to error.")
                print("Use --continue-on-error to continue past failures.")
                sys.exit(1)

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
