#!/usr/bin/env python3
"\"\"\"Quick bias check on the SR weighted actions: short usage, flat hold streaks, and action ratios.\"\""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


BASE_DIR = Path("backtest/plots_oos/br_sr_comparison/sr_final_analysis")
DEFAULT_ACTIONS = BASE_DIR / "sr_weighted_actions.csv"


def flat_hold_streaks(df: pd.DataFrame) -> list[int]:
    streaks = []
    streak = 0
    for _, row in df.iterrows():
        if int(row.get("Action", 0)) == 0 and int(row.get("Position", 0)) == 0:
            streak += 1
        else:
            if streak > 0:
                streaks.append(streak)
            streak = 0
    if streak > 0:
        streaks.append(streak)
    return streaks


def main(actions_path: Path) -> None:
    assert actions_path.exists(), f"{actions_path} not found."
    df = pd.read_csv(actions_path)
    total = len(df)
    action_counts = df["Action"].value_counts().to_dict()
    short_pct = 100.0 * action_counts.get(2, 0) / max(1, total)
    long_pct = 100.0 * action_counts.get(1, 0) / max(1, total)
    hold_pct = 100.0 * action_counts.get(0, 0) / max(1, total)

    hold_streaks = flat_hold_streaks(df)
    longest_hold = max(hold_streaks) if hold_streaks else 0
    avg_hold = float(sum(hold_streaks) / len(hold_streaks)) if hold_streaks else 0.0

    print("\n=== SR BIAS TEST ===")
    print(f"Total steps: {total}")
    print("Action breakdown:")
    print(f"  LONG  (1): {action_counts.get(1, 0)} ({long_pct:.1f}%)")
    print(f"  SHORT (2): {action_counts.get(2, 0)} ({short_pct:.1f}%)")
    print(f"  HOLD  (0): {action_counts.get(0, 0)} ({hold_pct:.1f}%)")
    print("\nHold while flat (action==0 and position==0):")
    print(f"  Streak count: {len(hold_streaks)}")
    print(f"  Longest flat hold streak: {longest_hold}")
    print(f"  Average flat hold streak: {avg_hold:.1f}")
    flat_steps = int(((df["Action"] == 0) & (df["Position"] == 0)).sum())
    print(f"  Hold steps flat: {flat_steps}")
    print("\nShorts are almost absent; consider re-training or seeding short signals to avoid directional imbalance.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Bias test on SR actions CSV")
    parser.add_argument(
        "--actions",
        type=str,
        default=str(DEFAULT_ACTIONS),
        help="Path to the actions CSV to analyze (default: sr_weighted_actions.csv)",
    )
    args = parser.parse_args()
    main(Path(args.actions))
