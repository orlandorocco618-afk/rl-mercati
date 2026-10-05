#!/usr/bin/env python3
"""
Safe cleanup utility for generated artifacts.
It only removes files that are reproducible (cache/log/generated outputs).
"""

from pathlib import Path
import argparse
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.manage_plots import (
    prune_chkpt_snapshots,
    prune_generated_action_csv,
    quarantine_corrupted_action_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["backtest", "train", "utils", "env", "tools", "data", "models"]
LOG_FILES = [
    PROJECT_ROOT / "test_output.log",
    PROJECT_ROOT / "test_chunked.log",
]


def _remove_path(path: Path, apply: bool):
    if not path.exists():
        return False
    if apply:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return True


def clean_pycache(apply: bool):
    removed = 0
    for rel_dir in SCAN_DIRS:
        base = PROJECT_ROOT / rel_dir
        if not base.exists():
            continue
        for cache_dir in base.rglob("__pycache__"):
            if _remove_path(cache_dir, apply):
                removed += 1
                print(f"{'REMOVED' if apply else 'WOULD REMOVE'} {cache_dir}")
    return removed


def clean_logs(apply: bool):
    removed = 0
    for fpath in LOG_FILES:
        if _remove_path(fpath, apply):
            removed += 1
            print(f"{'REMOVED' if apply else 'WOULD REMOVE'} {fpath}")
    return removed


def main():
    parser = argparse.ArgumentParser(description="Safe cleanup for generated artifacts.")
    parser.add_argument("--apply", action="store_true", help="Actually remove files")
    args = parser.parse_args()

    print("Running retention cleanup for plots/checkpoints/actions...")
    prune_chkpt_snapshots()
    quarantine_corrupted_action_csv()
    prune_generated_action_csv()

    print("Cleaning __pycache__ directories...")
    pycache_count = clean_pycache(args.apply)

    print("Cleaning stale logs...")
    log_count = clean_logs(args.apply)

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] pycache_dirs={pycache_count}, log_files={log_count}")


if __name__ == "__main__":
    main()
