"""
Organized management of the analysis charts.

Structure:
  backtest/plots/
    ├── ltst/          (latest - most recently generated charts)
    ├── chkpt/         (checkpoint archive - dated history)
    ├── analysis_summary.json
    └── dashboard.html

Workflow:
  1. generate_actions() → produces charts with standard names (01_, 02_, ...)
  2. save_plots_to_ltst() → moves charts into ltst/ and archives the old ones
  3. (optional) list_checkpoints() to see the history
"""

import os
import shutil
import re
from datetime import datetime
from pathlib import Path
import pandas as pd


PLOTS_DIR = Path("backtest/plots")
LTST_DIR = PLOTS_DIR / "ltst"
CHKPT_DIR = PLOTS_DIR / "chkpt"

PLOT_NAMES = [
    "01_equity_curve_trades.png",
    "02_action_distribution.png",
    "03_pnl_per_type.png",
    "04_trade_pnl_histogram.png",
    "05_position_over_time.png",
    "06_activity_by_month.png",
    "07_action_by_regime.png",
]

DEFAULT_MAX_CHKPT_SNAPSHOTS = int(os.getenv("RL_MAX_CHKPT_SNAPSHOTS", "12"))
DEFAULT_MAX_ACTION_FILES = int(os.getenv("RL_MAX_ACTION_FILES", "12"))
DEFAULT_MIN_ACTION_ROWS = int(os.getenv("RL_MIN_ACTION_ROWS", "500"))
DEFAULT_MIN_CLOSE_VALID_RATIO = float(os.getenv("RL_MIN_CLOSE_VALID_RATIO", "0.95"))
DEFAULT_MIN_ACTION_VALID_RATIO = float(os.getenv("RL_MIN_ACTION_VALID_RATIO", "0.98"))


def ensure_dirs():
    """Creates the directories if they don't exist."""
    LTST_DIR.mkdir(parents=True, exist_ok=True)
    CHKPT_DIR.mkdir(parents=True, exist_ok=True)


def archive_ltst_to_chkpt():
    """
    Archives the previous charts from ltst/ into chkpt/ with a timestamp.
    Es: 20260209_141530_01_equity_curve_trades.png
    """
    if not LTST_DIR.exists():
        return
    
    ltst_files = list(LTST_DIR.glob("*.png")) + list(LTST_DIR.glob("*.html"))
    if not ltst_files:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[ARCHIVE] Archiving previous charts with timestamp {timestamp}")
    
    for fpath in ltst_files:
        if fpath.is_file():
            new_name = f"{timestamp}_{fpath.name}"
            dest = CHKPT_DIR / new_name
            shutil.copy2(fpath, dest)
            print(f"  -> {new_name}")


def _extract_snapshot_id(filename: str) -> str:
    """
    Extract snapshot id from filename like:
    20260210_172048_01_equity_curve_trades.png -> 20260210_172048
    """
    match = re.match(r"^(\d{8}_\d{6})_", filename)
    return match.group(1) if match else ""


def prune_chkpt_snapshots(max_snapshots: int = DEFAULT_MAX_CHKPT_SNAPSHOTS):
    """
    Keep only the newest N checkpoint snapshots.
    A snapshot is grouped by timestamp prefix YYYYMMDD_HHMMSS.
    """
    if not CHKPT_DIR.exists():
        return

    files = [f for f in CHKPT_DIR.glob("*") if f.is_file()]
    snapshot_map = {}
    for fpath in files:
        snapshot_id = _extract_snapshot_id(fpath.name)
        if not snapshot_id:
            continue
        snapshot_map.setdefault(snapshot_id, []).append(fpath)

    snapshot_ids = sorted(snapshot_map.keys(), reverse=True)
    keep_ids = set(snapshot_ids[:max_snapshots])
    remove_ids = [sid for sid in snapshot_ids if sid not in keep_ids]

    removed = 0
    for sid in remove_ids:
        for fpath in snapshot_map.get(sid, []):
            try:
                fpath.unlink()
                removed += 1
            except Exception:
                pass

    # Remove legacy non-timestamped files in chkpt (reproducible, stale by definition).
    legacy_names = set(PLOT_NAMES + ["dashboard.html", "analysis_summary.json"])
    for fpath in CHKPT_DIR.glob("*"):
        if not fpath.is_file():
            continue
        if _extract_snapshot_id(fpath.name):
            continue
        if fpath.name in legacy_names:
            try:
                fpath.unlink()
                removed += 1
            except Exception:
                pass

    if removed > 0:
        print(f"[PRUNE] Removed {removed} chkpt files beyond retention ({max_snapshots} snapshots)")


def prune_generated_action_csv(max_files: int = DEFAULT_MAX_ACTION_FILES):
    """
    Keep core CSV and only newest generated extra CSV files.
    """
    data_dir = Path("backtest/data")
    if not data_dir.exists():
        return

    quarantine_corrupted_action_csv()

    keep_always = {
        "xauusd_actions.csv",
        "xauusd_actions_phase2.csv",
    }

    candidates = [f for f in data_dir.glob("xauusd_actions*.csv") if f.name not in keep_always]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    to_remove = candidates[max_files:]

    removed = 0
    for fpath in to_remove:
        try:
            fpath.unlink()
            removed += 1
        except Exception:
            pass

    if removed > 0:
        print(f"[PRUNE] Removed {removed} generated action CSVs beyond retention ({max_files})")


def _validate_action_csv(
    csv_path: Path,
    min_rows: int = DEFAULT_MIN_ACTION_ROWS,
    min_close_valid_ratio: float = DEFAULT_MIN_CLOSE_VALID_RATIO,
    min_action_valid_ratio: float = DEFAULT_MIN_ACTION_VALID_RATIO,
):
    """
    Return (is_valid, reason) for generated action CSV.
    """
    if not csv_path.exists() or not csv_path.is_file():
        return False, "missing_file"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"read_error:{type(e).__name__}"

    if df.empty:
        return False, "empty_file"
    if len(df) < int(max(1, min_rows)):
        return False, f"too_few_rows:{len(df)}"

    required_cols = {"Action", "Close"}
    missing = sorted(required_cols.difference(set(df.columns)))
    if missing:
        return False, f"missing_cols:{','.join(missing)}"

    close_valid_ratio = float(pd.to_numeric(df["Close"], errors="coerce").notna().mean())
    if close_valid_ratio < float(min_close_valid_ratio):
        return False, f"close_nan_ratio_too_high:{1.0 - close_valid_ratio:.3f}"

    action_series = pd.to_numeric(df["Action"], errors="coerce")
    action_valid_ratio = float(action_series.isin([0, 1, 2, 3]).mean())
    if action_valid_ratio < float(min_action_valid_ratio):
        return False, f"invalid_action_ratio:{1.0 - action_valid_ratio:.3f}"

    return True, "ok"


def validate_action_csv(
    csv_path,
    min_rows: int = DEFAULT_MIN_ACTION_ROWS,
    min_close_valid_ratio: float = DEFAULT_MIN_CLOSE_VALID_RATIO,
    min_action_valid_ratio: float = DEFAULT_MIN_ACTION_VALID_RATIO,
):
    """
    Public wrapper around action CSV validation.
    """
    return _validate_action_csv(
        Path(csv_path),
        min_rows=min_rows,
        min_close_valid_ratio=min_close_valid_ratio,
        min_action_valid_ratio=min_action_valid_ratio,
    )


def quarantine_corrupted_action_csv(
    min_rows: int = DEFAULT_MIN_ACTION_ROWS,
    min_close_valid_ratio: float = DEFAULT_MIN_CLOSE_VALID_RATIO,
    min_action_valid_ratio: float = DEFAULT_MIN_ACTION_VALID_RATIO,
):
    """
    Move invalid/legacy action CSV files out of auto-comparison paths.
    """
    data_dir = Path("backtest/data")
    if not data_dir.exists():
        return 0

    quarantine_dir = data_dir / "legacy_corrupted"
    moved = 0

    for csv_path in sorted(data_dir.glob("xauusd_actions*.csv")):
        valid, reason = _validate_action_csv(
            csv_path,
            min_rows=min_rows,
            min_close_valid_ratio=min_close_valid_ratio,
            min_action_valid_ratio=min_action_valid_ratio,
        )
        if valid:
            continue

        quarantine_dir.mkdir(parents=True, exist_ok=True)
        dst = quarantine_dir / csv_path.name
        if dst.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = quarantine_dir / f"{stamp}_{csv_path.name}"

        try:
            shutil.move(str(csv_path), str(dst))
            moved += 1
            print(f"[QUARANTINE] {csv_path.name} -> {dst.name} ({reason})")
        except Exception:
            pass

    if moved > 0:
        print(f"[QUARANTINE] Moved {moved} invalid/legacy action CSVs to {quarantine_dir}")

    return moved


def move_plots_to_ltst(source_dir=PLOTS_DIR):
    """
    Moves the charts from source_dir to ltst/ with standardized names.
    Assumes the charts have names like:
      - equity_curve_trades.png -> 01_equity_curve_trades.png
      - action_distribution.png -> 02_action_distribution.png
      - etc.
    
    Also cleans the old (unnumbered) files out of source_dir.
    """
    ensure_dirs()
    
    # Map from original name to standardized name
    mapping = {
        "equity_curve_trades.png": "01_equity_curve_trades.png",
        "01_equity_curve_trades.png": "01_equity_curve_trades.png",
        "action_distribution.png": "02_action_distribution.png",
        "02_action_distribution.png": "02_action_distribution.png",
        "pnl_per_type.png": "03_pnl_per_type.png",
        "03_pnl_per_type.png": "03_pnl_per_type.png",
        "trade_pnl_hist.png": "04_trade_pnl_histogram.png",
        "04_trade_pnl_histogram.png": "04_trade_pnl_histogram.png",
        "position_over_time.png": "05_position_over_time.png",
        "05_position_over_time.png": "05_position_over_time.png",
        "06_activity_by_month.png": "06_activity_by_month.png",
        "07_action_by_regime.png": "07_action_by_regime.png",
        "dashboard.html": "dashboard.html",
    }
    
    print("[MOVE] Moving previous charts to chkpt/")
    archive_ltst_to_chkpt()
    
    print("[COPY] Copying new charts into ltst/")
    for src_name, dst_name in mapping.items():
        src = source_dir / src_name
        if src.exists() and src.is_file():
            dst = LTST_DIR / dst_name
            shutil.copy2(src, dst)
            print(f"  {src_name} -> {dst_name}")
    
    # Copy analysis_summary.json
    summary_src = source_dir / "analysis_summary.json"
    if summary_src.exists():
        shutil.copy2(summary_src, LTST_DIR / "analysis_summary.json")
        print(f"  analysis_summary.json copied")
    
    # Cleanup: delete the old unnumbered files from source_dir
    print("[CLEANUP] Removing unnumbered files from source_dir...")
    old_files = [
        "equity_curve_trades.png",
        "action_distribution.png",
        "pnl_per_type.png",
        "trade_pnl_hist.png",
        "position_over_time.png",
        "equity_curve.png",
    ]
    for fname in old_files:
        fpath = source_dir / fname
        if fpath.exists():
            fpath.unlink()
            print(f"  Rimosso: {fname}")

    # Safe retention cleanup to avoid storage overload.
    prune_chkpt_snapshots()
    prune_generated_action_csv()


def list_checkpoints():
    """Lists the archived checkpoints."""
    if not CHKPT_DIR.exists():
        print("No checkpoints found")
        return
    
    chkpt_files = sorted(CHKPT_DIR.glob("*.png"))
    print(f"\n[CHECKPOINTS] {len(chkpt_files)} archived charts:")
    for f in chkpt_files[-10:]:  # Last 10
        print(f"  {f.name}")


def get_latest_summary() -> dict:
    """Loads the summary JSON from the latest checkpoint."""
    import json
    summary_path = LTST_DIR / "analysis_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    ensure_dirs()
    move_plots_to_ltst()
    list_checkpoints()
