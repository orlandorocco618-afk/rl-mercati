#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, ".")

from create_br_sr_comparison_html import prepare_test_data, resolve_existing_path
from out_of_sample_test import _run_slice


ROBUST_SELECTION = Path("best_models/robust_selection.json")
SR_SCORED = Path("backtest/plots_oos/robust_objective/sr_robust_scored.csv")
BR_SCORED = Path("backtest/plots_oos/robust_objective/br_robust_scored.csv")

OUT_DIR = Path("backtest/plots_oos/cost_stress")
SUMMARY_CSV = OUT_DIR / "scenario_results.csv"
MONTHLY_CSV = OUT_DIR / "scenario_monthly_returns.csv"
SUMMARY_JSON = OUT_DIR / "stress_test_summary.json"
HTML_DASHBOARD = OUT_DIR / "dashboard.html"

SCENARIOS = [
    {"name": "base_2_1", "spread_bps": 2.0, "slippage_bps": 1.0},
    {"name": "mild_4_2", "spread_bps": 4.0, "slippage_bps": 2.0},
    {"name": "medium_6_3", "spread_bps": 6.0, "slippage_bps": 3.0},
    {"name": "harsh_10_5", "spread_bps": 10.0, "slippage_bps": 5.0},
]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def load_env_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def evaluate_candidate(candidate: dict, scenario: dict, df_test: pd.DataFrame):
    model_path = str(resolve_existing_path(candidate["model_path"]))
    env_path = str(resolve_existing_path(candidate["env_config_path"]))
    env_cfg = load_env_config(env_path)
    env_cfg["spread_bps"] = float(scenario["spread_bps"])
    env_cfg["slippage_bps"] = float(scenario["slippage_bps"])
    env_cfg.setdefault("slippage_atr_coef", 0.0)
    env_cfg.setdefault("enable_zones", False)
    env_cfg.setdefault("zone_tol", 0.002)
    env_cfg.setdefault("zone_lookback", 5)

    model = PPO.load(model_path)
    label = f"{candidate['label']}_{scenario['name']}"
    actions_df, metrics = _run_slice(
        model=model,
        df_slice=df_test,
        env_cfg=env_cfg,
        allow_hold_when_flat=True,
        label=label,
    )
    monthly_summary, monthly_df = monthly_stats(actions_df)
    return metrics, monthly_summary, monthly_df


def select_candidates() -> list[dict]:
    robust = load_json(ROBUST_SELECTION)
    sr_ranked = pd.read_csv(SR_SCORED)
    br_ranked = pd.read_csv(BR_SCORED)

    candidates = []
    if robust.get("sr"):
        candidates.append(
            {
                "label": "SR_Robust",
                "family": "SR",
                "model_path": robust["sr"]["model_path"],
                "env_config_path": robust["sr"]["env_config_path"],
                "reference_return_pct": float(robust["sr"]["total_return_pct"]),
            }
        )
    if len(sr_ranked) > 1:
        alt = sr_ranked.iloc[1]
        candidates.append(
            {
                "label": "SR_Alt_Stable",
                "family": "SR",
                "model_path": str(alt["model_path"]),
                "env_config_path": str(alt["env_config_path"]),
                "reference_return_pct": float(alt["total_return_pct"]),
            }
        )
    if robust.get("br"):
        candidates.append(
            {
                "label": "BR_Robust",
                "family": "BR",
                "model_path": robust["br"]["model_path"],
                "env_config_path": robust["br"]["env_config_path"],
                "reference_return_pct": float(robust["br"]["total_return_pct"]),
            }
        )
    if len(sr_ranked) > 5:
        alt = sr_ranked.iloc[5]
        candidates.append(
            {
                "label": "SR_Alt_HigherMedian",
                "family": "SR",
                "model_path": str(alt["model_path"]),
                "env_config_path": str(alt["env_config_path"]),
                "reference_return_pct": float(alt["total_return_pct"]),
            }
        )
    return candidates


def render_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data.</p>"
    return df.to_html(index=False, classes="table", border=0)


def build_dashboard(summary_df: pd.DataFrame, summary: dict) -> str:
    pivot = summary_df.pivot(index="candidate_label", columns="scenario_name", values="total_return_pct").reset_index()
    drawdown_pivot = summary_df.pivot(index="candidate_label", columns="scenario_name", values="max_drawdown_pct").reset_index()
    top_table = render_table(
        summary_df[
            [
                "candidate_label",
                "scenario_name",
                "spread_bps",
                "slippage_bps",
                "total_return_pct",
                "max_drawdown_pct",
                "median_monthly_pct",
                "worst_month_pct",
                "positive_month_pct",
                "return_decay_pct",
            ]
        ]
    )
    return_table = render_table(pivot)
    dd_table = render_table(drawdown_pivot)
    summary_block = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <title>Execution Cost Stress Test</title>
  <style>
    :root {{
      --bg: #08111f;
      --panel: #0f1c2e;
      --line: rgba(154, 176, 200, 0.18);
      --text: #e8eef7;
      --muted: #9ab0c8;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(90, 169, 255, 0.15), transparent 28%),
        linear-gradient(180deg, #07111c 0%, #0b1727 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px;
    }}
    .card {{
      background: rgba(15, 28, 46, 0.92);
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
      <h1>Execution Cost Stress Test</h1>
      <p>
        Stress test of the robust candidates under progressively worse spread and slippage scenarios.
      </p>
    </div>
    <div class="card">
      <h2>Scenario matrix</h2>
      {top_table}
    </div>
    <div class="card">
      <h2>Return by scenario</h2>
      {return_table}
    </div>
    <div class="card">
      <h2>Drawdown by scenario</h2>
      {dd_table}
    </div>
    <div class="card">
      <h2>Summary</h2>
      <pre>{summary_block}</pre>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_test = prepare_test_data()
    candidates = select_candidates()

    summary_rows: list[dict] = []
    monthly_rows: list[dict] = []

    for candidate in candidates:
        for scenario in SCENARIOS:
            print(
                f"[{candidate['label']}] scenario={scenario['name']} "
                f"spread={scenario['spread_bps']} slippage={scenario['slippage_bps']}"
            )
            metrics, monthly_summary, monthly_df = evaluate_candidate(candidate, scenario, df_test)
            row = {
                "candidate_label": candidate["label"],
                "family": candidate["family"],
                "scenario_name": scenario["name"],
                "spread_bps": float(scenario["spread_bps"]),
                "slippage_bps": float(scenario["slippage_bps"]),
                "reference_return_pct": float(candidate["reference_return_pct"]),
                "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 6),
                "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 6),
                "pct_profitable": round(float(metrics.get("pct_profitable", 0.0)), 6),
                "total_trades": int(metrics.get("total_trades", 0)),
                "final_capital": round(float(metrics.get("final_capital", 10000.0)), 6),
                "return_decay_pct": round(float(metrics.get("total_return_pct", 0.0)) - float(candidate["reference_return_pct"]), 6),
            }
            row.update(monthly_summary)
            summary_rows.append(row)

            monthly_df["candidate_label"] = candidate["label"]
            monthly_df["scenario_name"] = scenario["name"]
            monthly_rows.append(monthly_df[["candidate_label", "scenario_name", "Month", "MonthlyReturnPct"]])

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["candidate_label", "spread_bps", "slippage_bps"]
    ).reset_index(drop=True)
    monthly_all = pd.concat(monthly_rows, ignore_index=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    monthly_all.to_csv(MONTHLY_CSV, index=False)

    best_per_scenario = (
        summary_df.sort_values(["scenario_name", "total_return_pct"], ascending=[True, False])
        .groupby("scenario_name")
        .head(1)
    )
    summary = {
        "scenarios": SCENARIOS,
        "candidates": candidates,
        "best_per_scenario": best_per_scenario.to_dict(orient="records"),
        "baseline_recommendation": {
            "most_resilient_by_harsh_case": summary_df[summary_df["scenario_name"] == "harsh_10_5"]
            .sort_values(["total_return_pct", "max_drawdown_pct"], ascending=[False, False])
            .head(1)
            .to_dict(orient="records")
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    HTML_DASHBOARD.write_text(build_dashboard(summary_df, summary), encoding="utf-8")

    print("Execution cost stress test complete.")
    print(f"  Summary CSV: {SUMMARY_CSV}")
    print(f"  Monthly CSV: {MONTHLY_CSV}")
    print(f"  Summary JSON: {SUMMARY_JSON}")
    print(f"  Dashboard: {HTML_DASHBOARD}")


if __name__ == "__main__":
    main()
