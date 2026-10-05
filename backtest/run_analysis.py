"""
Orchestrator script: runs the analysis and organizes the charts in ltst/ with chkpt/ management
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add backtest to the path
sys.path.insert(0, str(Path(__file__).parent))

from analyze_actions_v2 import analyze_actions_v2
from manage_plots import move_plots_to_ltst, get_latest_summary


def append_run_history(summary: dict, history_path: str = "backtest/plots/run_history.jsonl", max_entries: int = 300):
    """
    Persist latest summaries in a compact JSONL history with retention.
    """
    if not summary:
        return

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "final_capital": summary.get("final_capital"),
        "total_return_pct": summary.get("total_return_pct"),
        "total_trades": summary.get("total_trades"),
        "pct_profitable": summary.get("pct_profitable"),
        "max_drawdown_pct": summary.get("max_drawdown_pct"),
        "active_steps_pct": summary.get("active_steps_pct"),
        "max_hold_streak": summary.get("max_hold_streak"),
        "bias_flags": summary.get("bias_flags", []),
    }

    existing = []
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                existing = [line.rstrip("\n") for line in f if line.strip()]
        except Exception:
            existing = []

    existing.append(json.dumps(record, ensure_ascii=False))
    existing = existing[-max_entries:]

    with path.open("w", encoding="utf-8") as f:
        for line in existing:
            f.write(line + "\n")


def run_analysis_and_organize(
    actions_csv: str = "backtest/data/xauusd_actions.csv",
    initial_capital: float = 10000.0,
    save_dir: str = "backtest/plots"
):
    """
    1. Runs the analysis (generates numbered charts)
    2. Organizes the charts: ltst/ for the new ones, archives the old ones in chkpt/
    3. Prints the summary and checkpoint info
    """
    print("\n" + "="*70)
    print("ANALYSIS AND CHART ORGANIZATION")
    print("="*70)
    
    try:
        # Step 1: Generate charts
        print("\n[1/3] Generating charts...")
        stats = analyze_actions_v2(
            actions_csv=actions_csv,
            initial_capital=initial_capital,
            save_dir=save_dir
        )
        
        # Step 2: Organize files
        print("\n[2/3] Organizing chart files...")
        move_plots_to_ltst(source_dir=Path(save_dir))
        
        # Step 3: Print summary
        print("\n[3/3] Results summary...")
        summary = get_latest_summary()
        
        print("\n" + "="*70)
        print("RESULTS AVAILABLE IN:")
        print("="*70)
        print(f"  Latest:      backtest/plots/ltst/")
        print(f"  Archive:     backtest/plots/chkpt/")
        print(f"  Dashboard:   backtest/plots/ltst/dashboard.html")
        
        if summary:
            print("\n" + "="*70)
            print("LATEST METRICS:")
            print("="*70)
            print(f"  Capital:     ${summary.get('final_capital', 0):,.2f}")
            print(f"  Return:      {summary.get('total_return_pct', 0):+.2f}%")
            print(f"  Trades:      {summary.get('total_trades', 0)}")
            print(f"  Win Rate:    {summary.get('pct_profitable', 0):.1f}%")
            print(f"  Drawdown:    {summary.get('max_drawdown_pct', 0):.2f}%")
            print(f"  Active:      {summary.get('active_steps_pct', 0):.1f}%")
            print(f"  Max HOLD st: {summary.get('max_hold_streak', 0)}")

            append_run_history(summary)
            print("  History log: backtest/plots/run_history.jsonl")

        print("\n" + "="*70 + "\n")
        
        return stats
        
    except Exception as e:
        print(f"\n[ERR] ERROR during the analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_analysis_and_organize()
