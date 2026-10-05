#!/usr/bin/env python3
"""Replay the SR best actions with the weighted equity curve to validate the scaled exposures."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.analyze_actions_v2 import analyze_actions_v2


BASE_DIR = Path("backtest/plots_oos/br_sr_comparison/sr_final_analysis")
SR_ACTIONS = Path("backtest/plots_oos/br_sr_comparison/sr_best_actions.csv")
WEIGHTED_EQUITY = BASE_DIR / "sr_weighted_equity.csv"
WEIGHTED_ACTIONS = BASE_DIR / "sr_weighted_actions.csv"
BACKTEST_DIR = BASE_DIR / "weighted_backtest"


def main() -> None:
    weighted_dir = BACKTEST_DIR
    weighted_dir.mkdir(parents=True, exist_ok=True)

    actions = pd.read_csv(SR_ACTIONS)
    weighted = pd.read_csv(WEIGHTED_EQUITY)
    frame_length = min(len(actions), len(weighted))
    if len(actions) != len(weighted):
        print(f"Warning: action/weighted lengths differ ({len(actions)} vs {len(weighted)}); truncating to {frame_length}.")
        actions = actions.iloc[:frame_length].copy()
        weighted = weighted.iloc[:frame_length].copy()

    actions["Equity"] = weighted["weighted_equity"]
    if "monthly_weight" in weighted.columns:
        actions["monthly_weight"] = weighted["monthly_weight"]

    actions.to_csv(WEIGHTED_ACTIONS, index=False)
    analyze_actions_v2(
        actions_csv=str(WEIGHTED_ACTIONS),
        save_dir=str(weighted_dir),
    )

    print("Weighted backtest complete.")
    print(f"  Weighted actions: {WEIGHTED_ACTIONS}")
    print(f"  Dashboard: {weighted_dir / 'dashboard.html'}")


if __name__ == "__main__":
    main()
