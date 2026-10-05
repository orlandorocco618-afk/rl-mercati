#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, ".")

from backtest.create_br_sr_comparison_html import prepare_test_data, resolve_existing_path
from backtest.out_of_sample_test import _run_slice


ROBUST_SELECTION = Path("best_models/robust_selection.json")
OUT_DIR = Path("backtest/plots_oos/sr_dynamic_overlays")
SUMMARY_CSV = OUT_DIR / "overlay_results.csv"
MONTHLY_CSV = OUT_DIR / "overlay_monthly_returns.csv"
SUMMARY_JSON = OUT_DIR / "overlay_summary.json"
HTML_DASHBOARD = OUT_DIR / "dashboard.html"

SCENARIOS = [
    {"name": "base_2_1", "spread_bps": 2.0, "slippage_bps": 1.0},
    {"name": "medium_6_3", "spread_bps": 6.0, "slippage_bps": 3.0},
]

OVERLAYS = [
    {
        "name": "baseline",
        "label": "Baseline",
        "overrides": {},
    },
    {
        "name": "dynamic_size_v1",
        "label": "Dynamic Size",
        "overrides": {
            "enable_dynamic_sizing": True,
            "dynamic_reference_window": 96,
            "dynamic_size_floor": 0.70,
            "dynamic_size_ceiling": 1.35,
            "dynamic_confidence_power": 1.10,
            "dynamic_volatility_weight": 0.40,
            "dynamic_signal_weight": 0.15,
            "dynamic_zone_weight": 0.45,
            "dynamic_breakout_weight": 0.0,
            "enable_confidence_gate": False,
        },
    },
    {
        "name": "dynamic_gate_058",
        "label": "Dynamic + Gate 0.58",
        "overrides": {
            "enable_dynamic_sizing": True,
            "dynamic_reference_window": 96,
            "dynamic_size_floor": 0.70,
            "dynamic_size_ceiling": 1.35,
            "dynamic_confidence_power": 1.10,
            "dynamic_volatility_weight": 0.40,
            "dynamic_signal_weight": 0.15,
            "dynamic_zone_weight": 0.45,
            "dynamic_breakout_weight": 0.0,
            "enable_confidence_gate": True,
            "confidence_gate_threshold": 0.58,
            "confidence_gate_flatten_on_reverse": True,
        },
    },
    {
        "name": "dynamic_gate_062",
        "label": "Dynamic + Gate 0.62",
        "overrides": {
            "enable_dynamic_sizing": True,
            "dynamic_reference_window": 96,
            "dynamic_size_floor": 0.70,
            "dynamic_size_ceiling": 1.35,
            "dynamic_confidence_power": 1.10,
            "dynamic_volatility_weight": 0.40,
            "dynamic_signal_weight": 0.15,
            "dynamic_zone_weight": 0.45,
            "dynamic_breakout_weight": 0.0,
            "enable_confidence_gate": True,
            "confidence_gate_threshold": 0.62,
            "confidence_gate_flatten_on_reverse": True,
        },
    },
]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_env_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def monthly_stats(actions_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    frame = actions_df[["Time", "Equity"]].copy()
    frame["Time"] = pd.to_datetime(frame["Time"], errors="coerce")
    frame["Equity"] = pd.to_numeric(frame["Equity"], errors="coerce")
    frame = frame.dropna(subset=["Time"]).sort_values("Time")

    monthly = frame.set_index("Time")["Equity"].resample("ME").last().dropna()
    monthly_returns = monthly.pct_change().dropna() * 100.0
    monthly_df = monthly_returns.reset_index()
    monthly_df.columns = ["Time", "MonthlyReturnPct"]
    monthly_df["Month"] = monthly_df["Time"].dt.strftime("%Y-%m")

    if monthly_returns.empty:
        stats = {
            "months": 0,
            "avg_monthly_pct": 0.0,
            "median_monthly_pct": 0.0,
            "worst_month_pct": 0.0,
            "best_month_pct": 0.0,
            "positive_month_pct": 0.0,
        }
    else:
        stats = {
            "months": int(len(monthly_returns)),
            "avg_monthly_pct": round(float(monthly_returns.mean()), 6),
            "median_monthly_pct": round(float(monthly_returns.median()), 6),
            "worst_month_pct": round(float(monthly_returns.min()), 6),
            "best_month_pct": round(float(monthly_returns.max()), 6),
            "positive_month_pct": round(float((monthly_returns > 0).mean() * 100.0), 6),
        }
    return stats, monthly_df


def overlay_diagnostics(actions_df: pd.DataFrame) -> dict:
    frame = actions_df.copy()
    diagnostics = {
        "avg_position_size": 0.0,
        "avg_dynamic_multiplier": 1.0,
        "avg_setup_confidence": 0.0,
        "confidence_gate_blocks": 0,
        "confidence_gate_block_pct": 0.0,
    }
    if frame.empty:
        return diagnostics

    if "PositionSize" in frame:
        diagnostics["avg_position_size"] = round(float(pd.to_numeric(frame["PositionSize"], errors="coerce").mean()), 6)
    if "DynamicSizeMultiplier" in frame:
        diagnostics["avg_dynamic_multiplier"] = round(float(pd.to_numeric(frame["DynamicSizeMultiplier"], errors="coerce").mean()), 6)
    if "SetupConfidence" in frame:
        diagnostics["avg_setup_confidence"] = round(float(pd.to_numeric(frame["SetupConfidence"], errors="coerce").mean()), 6)
    if "ConfidenceGateBlocked" in frame:
        blocked = frame["ConfidenceGateBlocked"].fillna(False).astype(bool)
        diagnostics["confidence_gate_blocks"] = int(blocked.sum())
        diagnostics["confidence_gate_block_pct"] = round(float(blocked.mean() * 100.0), 6)
    return diagnostics


def evaluate_overlay(candidate: dict, overlay: dict, scenario: dict, df_test: pd.DataFrame):
    model_path = str(resolve_existing_path(candidate["model_path"]))
    env_path = str(resolve_existing_path(candidate["env_config_path"]))
    env_cfg = load_env_config(env_path)
    env_cfg.setdefault("enable_zones", True)
    env_cfg.setdefault("zone_tol", 0.003)
    env_cfg.setdefault("zone_lookback", 5)
    env_cfg["spread_bps"] = float(scenario["spread_bps"])
    env_cfg["slippage_bps"] = float(scenario["slippage_bps"])
    env_cfg["slippage_atr_coef"] = float(env_cfg.get("slippage_atr_coef", 0.0))
    env_cfg.update(overlay["overrides"])

    model = PPO.load(model_path)
    label = f"{overlay['name']}_{scenario['name']}"
    actions_df, metrics = _run_slice(
        model=model,
        df_slice=df_test,
        env_cfg=env_cfg,
        allow_hold_when_flat=True,
        label=label,
    )
    monthly_summary, monthly_df = monthly_stats(actions_df)
    diagnostics = overlay_diagnostics(actions_df)
    return metrics, monthly_summary, diagnostics, monthly_df


def render_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data.</p>"
    return df.to_html(index=False, classes="table", border=0)


def build_dashboard(summary_df: pd.DataFrame, summary: dict) -> str:
    base = summary_df.loc[summary_df["overlay_name"] == "baseline", ["scenario_name", "total_return_pct", "max_drawdown_pct"]].rename(
        columns={"total_return_pct": "baseline_return_pct", "max_drawdown_pct": "baseline_dd_pct"}
    )
    merged = summary_df.merge(base, on="scenario_name", how="left")
    merged["return_delta_vs_baseline"] = merged["total_return_pct"] - merged["baseline_return_pct"]
    merged["dd_delta_vs_baseline"] = merged["max_drawdown_pct"] - merged["baseline_dd_pct"]

    compact = merged[
        [
            "scenario_name",
            "overlay_label",
            "total_return_pct",
            "return_delta_vs_baseline",
            "max_drawdown_pct",
            "dd_delta_vs_baseline",
            "total_trades",
            "median_monthly_pct",
            "worst_month_pct",
            "avg_position_size",
            "avg_setup_confidence",
            "confidence_gate_blocks",
        ]
    ].copy()
    compact = compact.sort_values(["scenario_name", "total_return_pct"], ascending=[True, False])
    pivot = summary_df.pivot(index="overlay_label", columns="scenario_name", values="total_return_pct").reset_index()
    summary_block = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <title>SR Dynamic Overlays</title>
  <style>
    :root {{
      --bg: #08111f;
      --panel: rgba(16, 28, 47, 0.92);
      --line: rgba(149, 171, 194, 0.18);
      --text: #edf2fa;
      --muted: #9fb4ca;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(242, 184, 64, 0.12), transparent 28%),
        linear-gradient(180deg, #07101b 0%, #0b1827 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-top: 18px;
      box-shadow: 0 14px 30px rgba(0, 0, 0, 0.18);
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .table th, .table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    .table th {{
      color: var(--muted);
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      padding: 12px;
      border-radius: 12px;
      background: rgba(7, 17, 28, 0.9);
      border: 1px solid var(--line);
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1>SR Dynamic Overlays</h1>
      <p>
        Comparison between the SR_Robust baseline, dynamic sizing and a confidence gate.
        The gate should only be promoted if it really improves the OOS profile.
      </p>
    </div>
    <div class="card">
      <h2>Overlay matrix</h2>
      {render_table(compact)}
    </div>
    <div class="card">
      <h2>Returns by scenario</h2>
      {render_table(pivot)}
    </div>
    <div class="card">
      <h2>Summary JSON</h2>
      <pre>{summary_block}</pre>
    </div>
  </div>
</body>
</html>
"""


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def main():
    robust = load_json(ROBUST_SELECTION)
    sr = robust.get("sr")
    if not sr:
        raise RuntimeError("SR robust selection not found")

    candidate = {
        "label": "SR_Robust",
        "model_path": sr["model_path"],
        "env_config_path": sr["env_config_path"],
    }
    df_test = prepare_test_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    monthly_rows: list[dict] = []

    for scenario in SCENARIOS:
        for overlay in OVERLAYS:
            metrics, monthly_summary, diagnostics, monthly_df = evaluate_overlay(candidate, overlay, scenario, df_test)
            row = {
                "candidate_label": candidate["label"],
                "scenario_name": scenario["name"],
                "spread_bps": float(scenario["spread_bps"]),
                "slippage_bps": float(scenario["slippage_bps"]),
                "overlay_name": overlay["name"],
                "overlay_label": overlay["label"],
                **metrics,
                **monthly_summary,
                **diagnostics,
            }
            summary_rows.append(row)

            if not monthly_df.empty:
                monthly_df = monthly_df.copy()
                monthly_df["Scenario"] = scenario["name"]
                monthly_df["Overlay"] = overlay["name"]
                monthly_rows.extend(monthly_df.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows).sort_values(["scenario_name", "total_return_pct"], ascending=[True, False])
    monthly_df = pd.DataFrame(monthly_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    monthly_df.to_csv(MONTHLY_CSV, index=False)

    summary = {
        "candidate": candidate,
        "scenarios": SCENARIOS,
        "overlays": OVERLAYS,
        "results": summary_df.to_dict(orient="records"),
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(make_json_safe(summary), handle, indent=2, ensure_ascii=False)

    HTML_DASHBOARD.write_text(build_dashboard(summary_df, summary), encoding="utf-8")

    print(f"[OK] overlay results -> {SUMMARY_CSV}")
    print(f"[OK] monthly returns -> {MONTHLY_CSV}")
    print(f"[OK] summary -> {SUMMARY_JSON}")
    print(f"[OK] dashboard -> {HTML_DASHBOARD}")


if __name__ == "__main__":
    main()
