#!/usr/bin/env python3
"""Re-analyze valid timechunk CSV with final-close logic."""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

from backtest.analyze_actions_v2 import analyze_actions_v2
from backtest.manage_plots import validate_action_csv


print("\n" + "=" * 80)
print("QUICK CHECK: Re-analyzing valid timechunk CSV")
print("=" * 80)

data_dir = Path("backtest/data")
runs = sorted(data_dir.glob("xauusd_actions_timechunk*.csv"))

results = []
for csv_path in runs:
    ok, reason = validate_action_csv(csv_path)
    if not ok:
        print(f"Skipping {csv_path.name}: {reason}")
        continue

    label = csv_path.stem
    print(f"\n{'=' * 80}")
    print(f"RUN: {label}")
    print(f"{'=' * 80}")

    try:
        save_dir = Path("backtest/plots_time_chunks") / f"{label}_verified"
        os.makedirs(save_dir, exist_ok=True)

        stats = analyze_actions_v2(actions_csv=str(csv_path), initial_capital=10000.0, save_dir=str(save_dir))

        print("\n[SUMMARY]")
        print(f"  Return:      {stats.get('total_return_pct', 0):+.2f}%")
        print(f"  Trades:      {stats.get('total_trades', 0)}")
        print(f"  Win Rate:    {stats.get('pct_profitable', 0):.1f}%")
        print(f"  Drawdown:    {stats.get('max_drawdown_pct', 0):.2f}%")

        results.append({
            "label": label,
            "return": stats.get("total_return_pct", 0),
            "trades": stats.get("total_trades", 0),
            "winrate": stats.get("pct_profitable", 0),
            "drawdown": stats.get("max_drawdown_pct", 0),
        })
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'=' * 80}")
print("COMPARISON SUMMARY")
print(f"{'=' * 80}")
for r in results:
    print(f"\n{r['label']}")
    print(f"  Return: {r['return']:+7.2f}% | Trades: {r['trades']:3d} | WinRate: {r['winrate']:5.1f}% | DD: {r['drawdown']:6.2f}%")

print(f"\n{'=' * 80}\n")
