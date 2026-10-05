# TODO Regime + Sizing Roadmap

## Goal (current phase)
- Eliminate silent overfitting and HOLD-collapse.
- Keep regime detection always active.
- Use regime mainly for dynamic sizing (not binary trade/no-trade by default).
- Ensure the agent trades across the full CSV, except optional extreme hard-stop windows.

## Phase A - Data and Diagnostics Baseline
1. Run baseline training without hard-stop gating, with regime-aware sizing enabled.
2. Export enriched action logs (`DangerLevel`, `PositionSize`, `ForcedAction`, `HardStop`, `Equity`, `Reward`).
3. Analyze:
- action balance (HOLD/LONG/SHORT/CLOSE)
- max HOLD streak
- active steps %
- regime/danger-level action distribution
- month-by-month trading activity
4. Flag bias patterns automatically (`bias_flags` in summary JSON).

Done when:
- no NaN crash
- active steps are non-trivial
- both LONG and SHORT appear in non-extreme periods

## Phase B - Anti-Overfitting Validation
1. Use strict time splits (train years vs different test years).
2. Keep no-look-forward policy in every script.
3. Compare train vs OOS with fixed thresholds:
- return delta
- win rate delta
- trade count collapse
- HOLD streak explosion in OOS
4. Store validation report per run.

Done when:
- OOS behavior is coherent (even if low-profit)
- strategy logic remains stable across time chunks

## Phase C - Regime Logic Improvement
1. Quantify where strategy fails most:
- losses by regime
- losses by danger level
- losses by month/crisis clusters
2. Decide regime signal design:
- Option 1: feature-engineered rules (ATR/range/gap/volume)
- Option 2: learned detector (RL/supervised) for danger score
3. Keep detector output continuous/semi-discrete for sizing:
- danger 0/1/2 -> geometric position shrink
- hard-stop optional only at top danger
4. Backtest with and without hard-stop for comparison.

Done when:
- sizing responds to unstable periods without freezing most trading
- unstable windows reduce risk but do not dominate policy behavior

## Phase D - Stabilization for "Official" Pipeline
1. Freeze one canonical training pipeline.
2. Freeze one canonical time-split validation pipeline.
3. Add regression checks:
- NaN guard
- action-distribution guard
- OOS drift guard
4. Keep artifact retention policy active (no data overload).

Done when:
- repeated runs are analyzable and comparable
- artifacts remain compact and useful
