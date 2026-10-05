#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from create_br_sr_comparison_html import prepare_test_data
from stress_test_execution_costs import (
    ROBUST_SELECTION,
    SR_SCORED,
    BR_SCORED,
    evaluate_candidate,
    load_json,
)


OUT_DIR = Path("backtest/plots_oos/walk_forward")
WINDOW_RESULTS_CSV = OUT_DIR / "window_results.csv"
WINDOW_SUMMARY_CSV = OUT_DIR / "window_summary.csv"
SUMMARY_JSON = OUT_DIR / "walk_forward_summary.json"
HTML_DASHBOARD = OUT_DIR / "dashboard.html"

WINDOW_FREQ = "QS"
SCENARIO = {"name": "base_2_1", "spread_bps": 2.0, "slippage_bps": 1.0}


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


def build_windows(df_test: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    frame = df_test.copy()
    frame["Time"] = pd.to_datetime(frame["Time"], errors="coerce")
    frame = frame.dropna(subset=["Time"]).sort_values("Time")
    start = frame["Time"].min().normalize()
    end = frame["Time"].max().normalize()
    quarter_starts = list(pd.date_range(start=start, end=end, freq=WINDOW_FREQ))
    starts = [start]
    for value in quarter_starts:
        if value > start:
            starts.append(value)
    windows = []
    for idx, win_start in enumerate(starts):
        if idx + 1 < len(starts):
            win_end = starts[idx + 1]
        else:
            win_end = end + pd.Timedelta(days=1)
        if win_start >= win_end:
            continue
        windows.append((win_start, win_end))
    return windows


def evaluate_windows(candidates: list[dict], df_test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    windows = build_windows(df_test)
    for candidate in candidates:
        for idx, (start, end) in enumerate(windows, start=1):
            window_df = df_test[(df_test["Time"] >= start) & (df_test["Time"] < end)].reset_index(drop=True)
            if len(window_df) < 100:
                continue
            print(f"[{candidate['label']}] window {idx}/{len(windows)} {start.date()} -> {(end - pd.Timedelta(seconds=1)).date()}")
            metrics, monthly_summary, _ = evaluate_candidate(candidate, SCENARIO, window_df)
            rows.append(
                {
                    "candidate_label": candidate["label"],
                    "family": candidate["family"],
                    "window_idx": idx,
                    "window_start": start.strftime("%Y-%m-%d"),
                    "window_end_exclusive": end.strftime("%Y-%m-%d"),
                    "bars": int(len(window_df)),
                    "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 6),
                    "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 6),
                    "pct_profitable": round(float(metrics.get("pct_profitable", 0.0)), 6),
                    "total_trades": int(metrics.get("total_trades", 0)),
                    "final_capital": round(float(metrics.get("final_capital", 10000.0)), 6),
                    "avg_monthly_pct": monthly_summary["avg_monthly_pct"],
                    "median_monthly_pct": monthly_summary["median_monthly_pct"],
                    "worst_month_pct": monthly_summary["worst_month_pct"],
                    "best_month_pct": monthly_summary["best_month_pct"],
                    "positive_month_pct": monthly_summary["positive_month_pct"],
                }
            )
    return pd.DataFrame(rows).sort_values(["candidate_label", "window_idx"]).reset_index(drop=True)


def summarize_windows(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["candidate_label", "family"], sort=False)
    rows = []
    for (label, family), chunk in grouped:
        rows.append(
            {
                "candidate_label": label,
                "family": family,
                "windows": int(len(chunk)),
                "avg_window_return_pct": round(float(chunk["total_return_pct"].mean()), 6),
                "median_window_return_pct": round(float(chunk["total_return_pct"].median()), 6),
                "worst_window_return_pct": round(float(chunk["total_return_pct"].min()), 6),
                "best_window_return_pct": round(float(chunk["total_return_pct"].max()), 6),
                "positive_window_pct": round(float((chunk["total_return_pct"] > 0).mean() * 100.0), 6),
                "avg_window_drawdown_pct": round(float(chunk["max_drawdown_pct"].mean()), 6),
                "worst_window_drawdown_pct": round(float(chunk["max_drawdown_pct"].min()), 6),
                "avg_window_trades": round(float(chunk["total_trades"].mean()), 6),
            }
        )
    summary = pd.DataFrame(rows)
    summary["walk_forward_score"] = (
        0.35 * summary["avg_window_return_pct"].rank(pct=True)
        + 0.20 * summary["median_window_return_pct"].rank(pct=True)
        + 0.20 * summary["worst_window_return_pct"].rank(pct=True)
        + 0.15 * summary["positive_window_pct"].rank(pct=True)
        + 0.10 * summary["worst_window_drawdown_pct"].rank(pct=True)
    ).round(6)
    return summary.sort_values(["walk_forward_score", "avg_window_return_pct"], ascending=[False, False]).reset_index(drop=True)


def render_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data.</p>"
    return df.to_html(index=False, classes="table", border=0)


def build_dashboard(window_summary: pd.DataFrame, window_results: pd.DataFrame, summary: dict) -> str:
    top_table = render_table(window_summary)
    detail_table = render_table(
        window_results[
            [
                "candidate_label",
                "window_idx",
                "window_start",
                "window_end_exclusive",
                "total_return_pct",
                "max_drawdown_pct",
                "total_trades",
                "positive_month_pct",
            ]
        ]
    )
    summary_block = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <title>Walk Forward Evaluation</title>
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
      <h1>Walk Forward Evaluation</h1>
      <p>Sequential quarterly windows on the current OOS period using the base execution-cost scenario.</p>
    </div>
    <div class="card">
      <h2>Candidate ranking</h2>
      {top_table}
    </div>
    <div class="card">
      <h2>Window details</h2>
      {detail_table}
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
    df_test["Time"] = pd.to_datetime(df_test["Time"], errors="coerce")
    candidates = select_candidates()

    window_results = evaluate_windows(candidates, df_test)
    window_summary = summarize_windows(window_results)

    window_results.to_csv(WINDOW_RESULTS_CSV, index=False)
    window_summary.to_csv(WINDOW_SUMMARY_CSV, index=False)

    summary = {
        "window_scheme": {
            "frequency": WINDOW_FREQ,
            "scenario": SCENARIO,
            "window_count": int(window_results["window_idx"].max()) if not window_results.empty else 0,
        },
        "ranking": window_summary.to_dict(orient="records"),
        "recommended_baseline": window_summary.head(1).to_dict(orient="records"),
        "candidates": candidates,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    HTML_DASHBOARD.write_text(build_dashboard(window_summary, window_results, summary), encoding="utf-8")

    print("Walk-forward evaluation complete.")
    print(f"  Window results: {WINDOW_RESULTS_CSV}")
    print(f"  Window summary: {WINDOW_SUMMARY_CSV}")
    print(f"  Summary JSON:   {SUMMARY_JSON}")
    print(f"  Dashboard:      {HTML_DASHBOARD}")
    if not window_summary.empty:
        top = window_summary.iloc[0]
        print()
        print(
            f"Top walk-forward candidate: {top['candidate_label']} | "
            f"avg_window_return={top['avg_window_return_pct']:+.3f}% | "
            f"worst_window={top['worst_window_return_pct']:+.3f}% | "
            f"score={top['walk_forward_score']:.3f}"
        )


if __name__ == "__main__":
    main()
