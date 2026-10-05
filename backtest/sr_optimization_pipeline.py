#!/usr/bin/env python3
"""Generate reliability weights, risk sizing hints, and a weighted equity curve for the SR best strategy."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path("backtest/plots_oos/br_sr_comparison")
SR_ACTIONS = BASE_DIR / "sr_best_actions.csv"
ANALYSIS_SUMMARY = BASE_DIR / "sr_final_analysis/analysis_summary.json"
OUTPUT_DIR = BASE_DIR / "sr_final_analysis"
SUMMARY_JSON = OUTPUT_DIR / "sr_optimization_summary.json"
WEIGHTED_CSV = OUTPUT_DIR / "sr_weighted_equity.csv"

TARGET_MONTHLY_RETURN = 0.15  # 15% monthly goal


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def load_actions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Time" not in df.columns or "Equity" not in df.columns:
        raise RuntimeError("Expected 'Time' and 'Equity' columns in the actions CSV.")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Equity"] = pd.to_numeric(df["Equity"], errors="coerce")
    df["month"] = df["Time"].dt.to_period("M").astype(str)
    df["Action"] = pd.to_numeric(df["Action"], errors="coerce").fillna(0).astype(int)
    return df


def compute_monthly_stats(df: pd.DataFrame) -> tuple[list[dict], dict]:
    monthly_stats = []
    weights = {}
    for period, group in df.groupby("month", sort=True):
        equities = group["Equity"].dropna()
        if equities.empty:
            continue
        start = equities.iloc[0]
        end = equities.iloc[-1]
        ret = (end - start) / start if start != 0 else 0.0
        high = equities.max()
        low = equities.min()
        drawdown = (low - high) / high if high > 0 else 0.0
        drawdown_pct = abs(drawdown)
        penalty = min(drawdown_pct, 0.2)
        base_weight = 1.0 + np.clip(ret / TARGET_MONTHLY_RETURN, -1.0, 1.5)
        monthly_weight = clamp(base_weight * (1 - penalty), 0.4, 1.7)

        active_steps = int((group["Action"] != 0).sum())
        trades_closed = int((group["Action"] == 3).sum())

        month_info = {
            "period": period,
            "start_equity": float(start),
            "end_equity": float(end),
            "total_return_pct": round(ret * 100, 4),
            "drawdown_pct": round(-drawdown_pct * 100, 4),
            "active_steps": active_steps,
            "closed_trades": trades_closed,
            "weight": round(monthly_weight, 4),
        }
        monthly_stats.append(month_info)
        weights[period] = monthly_weight
    return monthly_stats, weights


def simulate_trades(df: pd.DataFrame) -> pd.DataFrame:
    trades = []
    position = 0
    entry_price = None
    entry_idx = None
    entry_regime = None
    entry_danger = None

    def close_trade(idx: int, close_price: float, trade_type: str) -> None:
        nonlocal position, entry_price, entry_idx, entry_regime, entry_danger
        if entry_price is None or entry_idx is None:
            return
        if position == 1:
            pnl = (close_price - entry_price) / entry_price
        else:
            pnl = (entry_price - close_price) / entry_price
        trades.append(
            {
                "close_idx": idx,
                "pnl": pnl,
                "type": trade_type,
                "regime": str(entry_regime) if entry_regime is not None else "unknown",
                "danger": str(entry_danger) if entry_danger is not None else "unknown",
                "duration": idx - entry_idx,
            }
        )
        entry_price = None
        entry_idx = None
        entry_regime = None
        entry_danger = None

    for idx, row in df.iterrows():
        action = int(row.get("Action", 0))
        price = row.get("Close")
        if price is None or pd.isna(price):
            continue

        current_regime = row.get("Regime")
        current_danger = row.get("DangerLevel")

        if action == 1:  # LONG
            if position == -1:
                close_trade(idx, price, "cover_short")
            position = 1
            entry_price = price
            entry_idx = idx
            entry_regime = current_regime
            entry_danger = current_danger

        elif action == 2:  # SHORT
            if position == 1:
                close_trade(idx, price, "close_long")
            position = -1
            entry_price = price
            entry_idx = idx
            entry_regime = current_regime
            entry_danger = current_danger

        elif action == 3:  # CLOSE
            if position == 1:
                close_trade(idx, price, "close_long")
            elif position == -1:
                close_trade(idx, price, "cover_short")
            position = 0

    if position != 0 and entry_price is not None:
        final_price = float(df.iloc[-1].get("Close", entry_price))
        close_trade(len(df) - 1, final_price, "final_close")
    return pd.DataFrame(trades)


def summarize_groups(trades: pd.DataFrame, field: str) -> dict[str, dict]:
    if trades.empty:
        return {}

    summary = {}
    for key, group in trades.groupby(field):
        total = len(group)
        profitable = int((group["pnl"] > 0).sum())
        win_rate = profitable / total if total > 0 else 0.0
        avg_pnl = float(group["pnl"].mean())
        weight = clamp(0.6 + win_rate * 0.6, 0.5, 1.5)
        if avg_pnl < 0:
            weight *= 0.9
        summary[str(key)] = {
            "total_trades": total,
            "profitable_trades": profitable,
            "win_rate": round(win_rate, 4),
            "avg_pnl": round(avg_pnl, 6),
            "weight": round(weight, 4),
        }
    return summary


def apply_weighted_equity(df: pd.DataFrame, weights: dict[str, float], initial_capital: float) -> pd.DataFrame:
    df = df.copy()
    df["monthly_weight"] = df["month"].map(weights).fillna(1.0)
    df["equity_delta"] = df["Equity"].diff().fillna(0.0)
    df["weighted_delta"] = df["equity_delta"] * df["monthly_weight"]
    df["weighted_equity"] = initial_capital + df["weighted_delta"].cumsum()
    return df


def compute_drawdown(series: pd.Series) -> float:
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return float(drawdown.min() * 100) if not series.empty else 0.0


def load_analysis_summary() -> dict:
    if not ANALYSIS_SUMMARY.exists():
        return {}
    with ANALYSIS_SUMMARY.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_actions(SR_ACTIONS)
    summary_stats = load_analysis_summary()
    initial_capital = summary_stats.get("initial_capital", 10000.0)

    monthly_stats, monthly_weights = compute_monthly_stats(df)
    trades = simulate_trades(df)
    regime_summary = summarize_groups(trades, "regime")
    danger_summary = summarize_groups(trades, "danger")

    weighted_df = apply_weighted_equity(df, monthly_weights, initial_capital)
    weighted_return_pct = round(
        (weighted_df["weighted_equity"].iloc[-1] / initial_capital - 1.0) * 100, 4
    )
    weighted_drawdown = compute_drawdown(weighted_df["weighted_equity"])

    result = {
        "analysis_source": str(SR_ACTIONS),
        "target_monthly_return": TARGET_MONTHLY_RETURN,
        "monthly_stats": monthly_stats,
        "regime_summary": regime_summary,
        "danger_summary": danger_summary,
        "weighted_backtest": {
            "final_equity": round(float(weighted_df["weighted_equity"].iloc[-1]), 4),
            "total_return_pct": weighted_return_pct,
            "max_drawdown_pct": round(weighted_drawdown, 4),
        },
        "risk_plan": {
            "position_scale_by_month": {
                stat["period"]: stat["weight"] for stat in monthly_stats
            },
            "regime_weighting": {
                name: data["weight"] for name, data in regime_summary.items()
            },
            "danger_weighting": {
                name: data["weight"] for name, data in danger_summary.items()
            },
            "bias_flags": summary_stats.get("bias_flags", []),
            "notes": [
                "Weight > 1.0 encourages more exposure when monthly return nears the +15% goal.",
                "Weight < 1.0 throttles exposure in challenging months/regimes or when drawdowns exceed ~5%.",
            ],
        },
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    weighted_df.to_csv(WEIGHTED_CSV, index=False)

    print("SR optimization summary written:")
    print(f"  Summary JSON: {SUMMARY_JSON}")
    print(f"  Weighted equity CSV: {WEIGHTED_CSV}")


if __name__ == "__main__":
    main()
