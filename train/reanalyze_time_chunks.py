from pathlib import Path
import sys

sys.path.insert(0, ".")

from backtest.manage_plots import validate_action_csv
from backtest.run_analysis import run_analysis_and_organize


base = Path("backtest/data")
paths = sorted(base.glob("xauusd_actions_timechunk*.csv"))

if not paths:
    print("No timechunk CSV found in backtest/data")
else:
    for p in paths:
        ok, reason = validate_action_csv(p)
        if not ok:
            print(f"Skipping {p.name}: {reason}")
            continue

        save_dir = Path("backtest/plots_time_chunks/reanalyzed") / p.stem
        print(f"Analysing {p}")
        run_analysis_and_organize(
            actions_csv=str(p),
            initial_capital=10000.0,
            save_dir=str(save_dir),
        )

print("Done")
