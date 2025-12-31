"""
Unified pipeline for DIME physician analysis.

This module provides:
1. Pipeline stage definitions
2. Dependency tracking
3. Output staleness checking
4. Manifest generation for reproducibility

Default behavior: Skip stages whose expected outputs already exist
Use --force to recompute everything from scratch
"""

import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .config import (
    CYCLES,
    DIAGNOSTICS_DIR,
    DIME_DONORS_PARQUET,
    DIME_ITEMIZED_MANIFEST,
    DIME_RECIPIENTS_PARQUET,
    DONOR_CYCLE_METADATA_PARQUET,
    DONOR_CYCLE_PANEL_PARQUET,
    DONOR_CYCLE_PANEL_PARTITIONED_DIR,
    LINKAGE_RESULTS_PARQUET,
    NPPES_PHYSICIANS_PARQUET,
    PHYSICIAN_LABELS_PARQUET,
    PLOTS_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    TABLES_DIR,
    ensure_directories,
)
from . import run_manager

logger = logging.getLogger(__name__)


def get_git_info() -> dict:
    """Get current git commit info."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ) != 0

        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def compute_file_hash(path: Path) -> Optional[str]:
    """Compute SHA256 hash of a file."""
    if not path.exists():
        return None

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_output_manifest(
    output_path: Path,
    inputs: list[Path],
    parameters: dict,
    row_count: Optional[int] = None,
) -> Path:
    """
    Generate a manifest file for an output artifact.

    Args:
        output_path: Path to the output file/directory
        inputs: List of input file paths
        parameters: Parameters used to generate the output
        row_count: Optional row count for tabular outputs

    Returns:
        Path to the manifest file
    """
    manifest = {
        "output_path": str(output_path),
        "created_at": datetime.now().isoformat(),
        "inputs": [
            {
                "path": str(p),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat() if p.exists() else None,
                "sha256": compute_file_hash(p) if p.is_file() else None,
            }
            for p in inputs
        ],
        "code_version": get_git_info(),
        "parameters": parameters,
    }

    if row_count is not None:
        manifest["row_count"] = row_count

    manifest_path = output_path.parent / f"{output_path.name}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def check_output_stale(output_path: Path, manifest_path: Optional[Path] = None) -> bool:
    """
    Check if an output is stale (needs recomputation).

    Args:
        output_path: Path to the output file/directory
        manifest_path: Optional path to manifest (default: output_path + .manifest.json)

    Returns:
        True if output needs recomputation
    """
    if not output_path.exists():
        return True

    manifest_path = manifest_path or output_path.parent / f"{output_path.name}.manifest.json"
    if not manifest_path.exists():
        # No manifest = assume stale
        return True

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except Exception:
        return True

    # Check if any input is newer than output
    output_mtime = output_path.stat().st_mtime

    for input_info in manifest.get("inputs", []):
        input_path = Path(input_info.get("path", ""))
        if input_path.exists() and input_path.stat().st_mtime > output_mtime:
            return True

    return False


# Pipeline stage definitions
# NOTE: Script names must match actual files in scripts/
PIPELINE_STAGES = {
    "manifest": {
        "name": "Build DIME Manifest",
        "script": "scripts/build_dime_manifest.py",
        "outputs": [DIME_ITEMIZED_MANIFEST],
        "dependencies": [],
        "no_cycles_arg": True,  # This script doesn't accept --cycles
    },
    "data_ingest": {
        "name": "Data Ingestion (DIME + NPPES)",
        "script": "scripts/01_ingest_data.py",
        "outputs": [DIME_DONORS_PARQUET, DIME_RECIPIENTS_PARQUET, NPPES_PHYSICIANS_PARQUET],
        "dependencies": [],
        "no_cycles_arg": True,
    },
    "ingest": {
        "name": "Streaming Ingestion",
        "script": "scripts/00_streaming_ingest.py",
        "outputs": [DONOR_CYCLE_PANEL_PARTITIONED_DIR],
        "dependencies": ["manifest"],
    },
    "physician_id": {
        "name": "Physician Classification + Linkage",
        "script": "scripts/02_build_physician_labels.py",
        "outputs": [PHYSICIAN_LABELS_PARQUET, LINKAGE_RESULTS_PARQUET],
        "dependencies": ["data_ingest"],
        "no_cycles_arg": True,
    },
    "panel": {
        "name": "Build Donor-Cycle Panel",
        "script": "scripts/03_build_donor_panel.py",
        "outputs": [DONOR_CYCLE_PANEL_PARQUET],
        "dependencies": ["physician_id", "data_ingest"],
    },
    "lookahead": {
        "name": "Lookahead Fix",
        "script": "scripts/07_fix_lookahead.py",
        "outputs": [DONOR_CYCLE_METADATA_PARQUET],
        "dependencies": ["ingest"],
        "no_cycles_arg": True,
    },
    "coverage": {
        "name": "Coverage Audits",
        "script": "scripts/08_coverage_audit.py",
        "outputs": [DIAGNOSTICS_DIR / "itemized_coverage_by_cycle.csv"],
        "dependencies": ["ingest", "data_ingest"],
        "no_cycles_arg": True,
    },
    "representativeness": {
        "name": "Representativeness",
        "script": "scripts/09_representativeness.py",
        "outputs": [DIAGNOSTICS_DIR / "intensity_stratified_stats.csv"],
        "dependencies": ["panel", "physician_id", "data_ingest"],
        "no_cycles_arg": True,
    },
    "precision": {
        "name": "Precision Evaluation",
        "script": "scripts/10_precision_eval.py",
        "outputs": [DIAGNOSTICS_DIR / "calibration_report.txt"],
        "dependencies": ["physician_id", "data_ingest"],
        "no_cycles_arg": True,
    },
    "plots": {
        "name": "Generate Plots",
        "script": "scripts/04_generate_outputs.py",
        "outputs": [PLOTS_DIR],
        "dependencies": ["panel"],
        "no_cycles_arg": True,
    },
}


def get_stage_status(stage_id: str) -> dict:
    """Get the status of a pipeline stage."""
    stage = PIPELINE_STAGES.get(stage_id)
    if not stage:
        return {"exists": False, "stale": True, "error": "Unknown stage"}

    outputs = stage.get("outputs", [])

    if not outputs:
        return {"exists": False, "stale": True}

    all_exist = all(p.exists() for p in outputs)
    any_stale = any(check_output_stale(p) for p in outputs)

    return {
        "exists": all_exist,
        "stale": any_stale,
        "outputs": [str(p) for p in outputs],
    }


def should_run_stage(stage_id: str, force: bool = False, rerun_stages: Optional[list[str]] = None) -> bool:
    """
    Determine if a stage should run.

    Args:
        stage_id: Stage identifier
        force: Force rerun regardless of status
        rerun_stages: List of stages to explicitly rerun

    Returns:
        True if stage should run
    """
    if force:
        return True

    if rerun_stages and stage_id in rerun_stages:
        return True

    status = get_stage_status(stage_id)
    return not status["exists"] or status["stale"]


def run_stage(stage_id: str, force: bool = False, **kwargs) -> dict:
    """
    Run a single pipeline stage.

    Args:
        stage_id: Stage identifier
        force: Force rerun
        **kwargs: Additional arguments to pass to stage script

    Returns:
        Result dictionary with status and outputs
    """
    stage = PIPELINE_STAGES.get(stage_id)
    if not stage:
        return {"status": "error", "error": f"Unknown stage: {stage_id}"}

    script_path = PROJECT_ROOT / stage["script"]
    if not script_path.exists():
        return {"status": "error", "error": f"Script not found: {script_path}"}

    logger.info(f"Running stage: {stage['name']}")

    # Build command
    cmd = ["python", str(script_path)]
    if force:
        cmd.append("--overwrite")  # Most scripts use --overwrite instead of --force

    # Skip --cycles for scripts that don't support it
    no_cycles = stage.get("no_cycles_arg", False)

    for key, value in kwargs.items():
        if key == "cycles" and no_cycles:
            continue  # Skip cycles arg for scripts that don't support it
        cmd.append(f"--{key.replace('_', '-')}")
        if value is not True:
            cmd.append(str(value))

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "error": result.stderr,
                "stdout": result.stdout,
            }

        return {
            "status": "success",
            "stdout": result.stdout,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_pipeline(
    force: bool = False,
    rerun_stages: Optional[list[str]] = None,
    cycles: Optional[list[int]] = None,
    skip_stages: Optional[list[str]] = None,
    description: str = "",
    seed: Optional[int] = 42,
) -> dict:
    """
    Run the full pipeline with forensic audit tracking.

    Args:
        force: Force rerun all stages
        rerun_stages: List of specific stages to rerun
        cycles: Election cycles to process
        skip_stages: Stages to skip
        description: Description for the run manifest
        seed: Random seed for reproducibility (None to skip seeding)

    Returns:
        Dictionary with results for each stage
    """
    ensure_directories()

    # Create run with forensic tracking
    run_id = run_manager.create_run(
        description=description or "DIME physician analysis pipeline",
        seed=seed,
    )

    skip_stages = skip_stages or []
    results = {}
    any_errors = False

    # Determine execution order (topological sort based on dependencies)
    execution_order = [
        "manifest",
        "data_ingest",
        "ingest",
        "physician_id",
        "panel",
        "lookahead",
        "coverage",
        "representativeness",
        "precision",
        "plots",
    ]

    for stage_id in execution_order:
        stage = PIPELINE_STAGES.get(stage_id)
        if not stage:
            continue

        if stage_id in skip_stages:
            logger.info(f"Skipping stage: {stage_id}")
            results[stage_id] = {"status": "skipped"}
            continue

        if not should_run_stage(stage_id, force, rerun_stages):
            logger.info(f"Stage already complete: {stage_id}")
            results[stage_id] = {"status": "cached"}
            continue

        # Check dependencies
        deps_satisfied = all(
            results.get(dep, {}).get("status") in ["success", "cached", "skipped"]
            for dep in stage.get("dependencies", [])
        )

        if not deps_satisfied:
            logger.warning(f"Dependencies not satisfied for: {stage_id}")
            results[stage_id] = {"status": "skipped", "reason": "dependencies not met"}
            continue

        # Log step start with run_manager
        run_manager.log_step_start(
            run_id,
            stage["name"],
            parameters={
                "force": force or (stage_id in (rerun_stages or [])),
                "cycles": cycles,
            },
        )

        # Run stage
        kwargs = {}
        if cycles:
            kwargs["cycles"] = ",".join(str(c) for c in cycles)

        result = run_stage(stage_id, force=force or (stage_id in (rerun_stages or [])), **kwargs)
        results[stage_id] = result

        if result["status"] == "success":
            # Log step completion and output files
            run_manager.log_step_complete(run_id, stage["name"])

            # Log output files with SHA256 checksums
            for output_path in stage.get("outputs", []):
                if output_path.exists():
                    run_manager.log_output_file(
                        run_id,
                        f"{stage_id}:{output_path.name}",
                        output_path,
                    )
        else:
            # Log error
            error_msg = result.get("error", "Unknown error")
            run_manager.log_step_error(run_id, stage["name"], error_msg)
            any_errors = True
            logger.error(f"Stage failed: {stage_id}")
            # Continue with other stages (some may not depend on failed stage)

    # Finalize run with forensic checksums
    final_status = "completed" if not any_errors else "completed_with_errors"
    run_manager.finalize_run(run_id, status=final_status)

    return results


def get_pipeline_status() -> dict:
    """Get status of all pipeline stages."""
    status = {}
    for stage_id in PIPELINE_STAGES:
        status[stage_id] = get_stage_status(stage_id)
    return status
