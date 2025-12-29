import json
import os
import subprocess
import time
from typing import Dict


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def create_run_dir(base_dir: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config_snapshot(config: Dict, run_dir: str) -> str:
    path = os.path.join(run_dir, "config_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path


def write_run_metadata(run_dir: str) -> None:
    path = os.path.join(run_dir, "run_meta.json")
    payload = {"git_commit": _git_commit_hash()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
