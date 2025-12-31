"""
Run Manager for DIME Physician Analysis Pipeline.

Provides audit logging, reproducibility tracking, and run management.
Each run creates a dated folder with full manifest of what was executed.
"""

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .config import PROJECT_ROOT, PROCESSED_DATA_DIR, RAW_DATA_DIR

# Runs directory
RUNS_DIR = PROJECT_ROOT / "runs"

# Global seed tracking for reproducibility
_GLOBAL_SEED: Optional[int] = None
_SEED_LOG: list[dict] = []


def capture_environment() -> dict:
    """Capture Python version, key package versions, and OS info for reproducibility."""
    env_info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
    }

    # Capture key package versions
    packages = {}
    for pkg_name in ["pandas", "numpy", "duckdb", "pyarrow", "polars", "scikit-learn"]:
        try:
            mod = __import__(pkg_name.replace("-", "_"))
            packages[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg_name] = "not installed"

    env_info["packages"] = packages
    return env_info


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file for forensic integrity."""
    if not filepath.exists():
        return "file_not_found"

    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def set_global_seed(seed: int = 42) -> int:
    """
    Set and track global random seed for reproducibility.

    Sets seeds for: random, numpy (if available).
    Returns the seed used.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    random.seed(seed)
    if HAS_NUMPY:
        np.random.seed(seed)

    _SEED_LOG.append({
        "seed": seed,
        "set_at": datetime.now().isoformat(),
        "numpy_available": HAS_NUMPY,
    })

    print(f"Global seed set to: {seed}")
    return seed


def get_seed_log() -> list[dict]:
    """Get the log of all seeds set during this session."""
    return _SEED_LOG.copy()


def get_current_seed() -> Optional[int]:
    """Get the current global seed, or None if not set."""
    return _GLOBAL_SEED


def get_git_info() -> dict:
    """Get current git commit and branch info."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(status)

        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def compute_file_checksum(filepath: Path) -> str:
    """Compute MD5 checksum of a file."""
    if not filepath.exists():
        return "file_not_found"

    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_info(filepath: Path, include_sha256: bool = True) -> dict:
    """Get file info including size and checksums (MD5 + optional SHA256).

    For directories, returns file count and total size instead of checksums.
    """
    if not filepath.exists():
        return {"exists": False}

    # Handle directories differently
    if filepath.is_dir():
        files = list(filepath.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        return {
            "exists": True,
            "path": str(filepath),
            "is_directory": True,
            "file_count": file_count,
            "size_bytes": total_size,
            "size_human": _human_readable_size(total_size),
            "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
        }

    info = {
        "exists": True,
        "path": str(filepath),
        "is_directory": False,
        "size_bytes": filepath.stat().st_size,
        "size_human": _human_readable_size(filepath.stat().st_size),
        "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
        "checksum_md5": compute_file_checksum(filepath),
    }

    if include_sha256:
        info["checksum_sha256"] = compute_sha256(filepath)

    return info


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def create_run(description: str = "", seed: Optional[int] = 42) -> str:
    """
    Create a new run folder with unique ID.

    Args:
        description: Human-readable description of the run
        seed: Random seed for reproducibility (None to skip seeding)

    Returns:
        Run ID in format YYYY-MM-DD_NNN
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")

    # Find next run number for today
    existing = list(RUNS_DIR.glob(f"{today}_*"))
    if existing:
        nums = [int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()]
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1

    run_id = f"{today}_{next_num:03d}"
    run_dir = RUNS_DIR / run_id

    # Create directory structure
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "config_snapshot").mkdir()
    (run_dir / "metrics").mkdir()

    # Set global seed if provided
    if seed is not None:
        set_global_seed(seed)

    # Copy config files
    config_dir = PROJECT_ROOT / "config"
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            shutil.copy(config_file, run_dir / "config_snapshot" / config_file.name)

    # Initialize manifest with environment and seed tracking
    manifest = {
        "run_id": run_id,
        "description": description,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "status": "running",
        "git": get_git_info(),
        "environment": capture_environment(),
        "seed": seed,
        "seed_log": [],
        "parameters": {},
        "steps": [],
        "input_files": {},
        "output_files": {},
        "cleanup_performed": [],
        "errors": [],
    }

    save_manifest(run_id, manifest)

    print(f"Created run: {run_id}")
    print(f"Run directory: {run_dir}")
    if seed is not None:
        print(f"Random seed: {seed}")

    return run_id


def get_run_dir(run_id: str) -> Path:
    """Get the directory for a run."""
    return RUNS_DIR / run_id


def load_manifest(run_id: str) -> dict:
    """Load the manifest for a run."""
    manifest_path = get_run_dir(run_id) / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def save_manifest(run_id: str, manifest: dict) -> None:
    """Save the manifest for a run."""
    manifest_path = get_run_dir(run_id) / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def log_step_start(run_id: str, step_name: str, parameters: dict = None) -> None:
    """Log the start of a pipeline step."""
    manifest = load_manifest(run_id)

    step_info = {
        "name": step_name,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "status": "running",
        "parameters": parameters or {},
        "duration_seconds": None,
    }

    manifest["steps"].append(step_info)
    manifest["parameters"].update(parameters or {})
    save_manifest(run_id, manifest)

    print(f"[{run_id}] Starting step: {step_name}")


def log_step_complete(run_id: str, step_name: str, metrics: dict = None) -> None:
    """Log the completion of a pipeline step."""
    manifest = load_manifest(run_id)

    for step in manifest["steps"]:
        if step["name"] == step_name and step["status"] == "running":
            step["completed_at"] = datetime.now().isoformat()
            step["status"] = "completed"

            # Calculate duration
            start = datetime.fromisoformat(step["started_at"])
            end = datetime.fromisoformat(step["completed_at"])
            step["duration_seconds"] = (end - start).total_seconds()

            if metrics:
                step["metrics"] = metrics
            break

    save_manifest(run_id, manifest)

    print(f"[{run_id}] Completed step: {step_name}")


def log_step_error(run_id: str, step_name: str, error: str) -> None:
    """Log an error during a pipeline step."""
    manifest = load_manifest(run_id)

    for step in manifest["steps"]:
        if step["name"] == step_name and step["status"] == "running":
            step["completed_at"] = datetime.now().isoformat()
            step["status"] = "error"
            step["error"] = error
            break

    manifest["errors"].append({
        "step": step_name,
        "time": datetime.now().isoformat(),
        "error": error,
    })

    save_manifest(run_id, manifest)

    print(f"[{run_id}] ERROR in step {step_name}: {error}")


def log_input_file(run_id: str, name: str, filepath: Path) -> None:
    """Log an input file with its metadata."""
    manifest = load_manifest(run_id)
    manifest["input_files"][name] = get_file_info(filepath)
    save_manifest(run_id, manifest)


def log_output_file(run_id: str, name: str, filepath: Path) -> None:
    """Log an output file with its metadata."""
    manifest = load_manifest(run_id)
    manifest["output_files"][name] = get_file_info(filepath)
    save_manifest(run_id, manifest)


def log_cleanup(run_id: str, filepath: Path, reason: str = "") -> None:
    """Log a file that was cleaned up (deleted)."""
    manifest = load_manifest(run_id)

    file_info = get_file_info(filepath) if filepath.exists() else {"path": str(filepath)}
    file_info["deleted_at"] = datetime.now().isoformat()
    file_info["reason"] = reason

    manifest["cleanup_performed"].append(file_info)
    save_manifest(run_id, manifest)

    print(f"[{run_id}] Logged cleanup: {filepath}")


def cleanup_raw_file(run_id: str, filepath: Path, reason: str = "") -> bool:
    """Delete a raw file and log the cleanup."""
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return False

    # Log before deleting
    log_cleanup(run_id, filepath, reason)

    # Delete the file
    try:
        if filepath.is_dir():
            shutil.rmtree(filepath)
        else:
            filepath.unlink()
        print(f"Deleted: {filepath}")
        return True
    except Exception as e:
        print(f"Failed to delete {filepath}: {e}")
        return False


def finalize_run(run_id: str, status: str = "completed") -> None:
    """Finalize a run and save final manifest with forensic checksums."""
    manifest = load_manifest(run_id)

    manifest["completed_at"] = datetime.now().isoformat()
    manifest["status"] = status

    # Save seed log for reproducibility audit
    manifest["seed_log"] = get_seed_log()

    # Calculate total duration
    if manifest.get("started_at"):
        start = datetime.fromisoformat(manifest["started_at"])
        end = datetime.fromisoformat(manifest["completed_at"])
        manifest["total_duration_seconds"] = (end - start).total_seconds()

    # Save checksums of all output files (both MD5 and SHA256)
    checksums = {
        "md5": {},
        "sha256": {},
    }
    for name, info in manifest.get("output_files", {}).items():
        if info.get("exists"):
            if info.get("checksum_md5"):
                checksums["md5"][name] = info["checksum_md5"]
            if info.get("checksum_sha256"):
                checksums["sha256"][name] = info["checksum_sha256"]

    checksums_path = get_run_dir(run_id) / "checksums.json"
    with open(checksums_path, "w") as f:
        json.dump(checksums, f, indent=2)

    save_manifest(run_id, manifest)

    print(f"\n{'='*60}")
    print(f"Run {run_id} finalized: {status}")
    print(f"Duration: {manifest.get('total_duration_seconds', 0):.1f} seconds")
    print(f"Steps completed: {len([s for s in manifest['steps'] if s['status'] == 'completed'])}")
    print(f"Files cleaned up: {len(manifest['cleanup_performed'])}")
    print(f"Seed used: {manifest.get('seed', 'not set')}")
    print(f"{'='*60}")


def list_runs() -> list[dict]:
    """List all runs with basic info."""
    if not RUNS_DIR.exists():
        return []

    runs = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if run_dir.is_dir():
            manifest = load_manifest(run_dir.name)
            runs.append({
                "run_id": run_dir.name,
                "status": manifest.get("status", "unknown"),
                "started_at": manifest.get("started_at"),
                "completed_at": manifest.get("completed_at"),
                "description": manifest.get("description", ""),
            })

    return runs


def print_run_summary(run_id: str) -> None:
    """Print a summary of a run."""
    manifest = load_manifest(run_id)

    if not manifest:
        print(f"Run not found: {run_id}")
        return

    print(f"\n{'='*60}")
    print(f"Run: {run_id}")
    print(f"Status: {manifest.get('status')}")
    print(f"Started: {manifest.get('started_at')}")
    print(f"Completed: {manifest.get('completed_at')}")
    print(f"Git: {manifest.get('git', {}).get('commit')} ({manifest.get('git', {}).get('branch')})")

    print(f"\nSteps:")
    for step in manifest.get("steps", []):
        status_icon = "+" if step["status"] == "completed" else "x" if step["status"] == "error" else "..."
        duration = f"({step.get('duration_seconds', 0):.1f}s)" if step.get("duration_seconds") else ""
        print(f"  [{status_icon}] {step['name']} {duration}")

    print(f"\nInput files: {len(manifest.get('input_files', {}))}")
    print(f"Output files: {len(manifest.get('output_files', {}))}")
    print(f"Files cleaned up: {len(manifest.get('cleanup_performed', []))}")
    print(f"{'='*60}")
