#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from create_br_sr_comparison_html import (
    BR_RESULTS,
    SR_RESULTS,
    evaluate_model,
    prepare_test_data,
    resolve_existing_path,
)


OUT_DIR = Path("backtest/plots_oos/robust_objective")
BR_SCORED_CSV = OUT_DIR / "br_robust_scored.csv"
SR_SCORED_CSV = OUT_DIR / "sr_robust_scored.csv"
ALL_SCORED_CSV = OUT_DIR / "all_candidates_scored.csv"
SUMMARY_JSON = OUT_DIR / "robust_selection_summary.json"
HTML_DASHBOARD = OUT_DIR / "dashboard.html"
ROBUST_SELECTION_JSON = Path("best_models/robust_selection.json")


def load_candidates(results_path: Path, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(results_path).copy()
    available_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            row_dict["resolved_model_path"] = str(resolve_existing_path(str(row["model_path"])))
            row_dict["resolved_env_config_path"] = str(resolve_existing_path(str(row["env_config_path"])))
            row_dict["family"] = family
            available_rows.append(row_dict)
        except Exception as exc:
            row_dict["family"] = family
            row_dict["skip_reason"] = str(exc)
            skipped_rows.append(row_dict)

    available_df = pd.DataFrame(available_rows)
    if not available_df.empty:
        available_df["source_duplicate_count"] = (
            available_df.groupby(["resolved_model_path", "resolved_env_config_path"])["model_path"].transform("count")
        )
        available_df = (
            available_df.sort_values(["total_return_pct", "model_path"], ascending=[False, True])
            .drop_duplicates(subset=["resolved_model_path", "resolved_env_config_path"], keep="first")
            .reset_index(drop=True)
        )

    return available_df, pd.DataFrame(skipped_rows)


def monthly_stats(actions_df: pd.DataFrame) -> dict:
    frame = actions_df[["Time", "Equity"]].copy()
    frame["Time"] = pd.to_datetime(frame["Time"], errors="coerce")
    frame["Equity"] = pd.to_numeric(frame["Equity"], errors="coerce")
    frame = frame.dropna(subset=["Time"]).sort_values("Time")
    monthly = frame.set_index("Time")["Equity"].resample("ME").last().dropna()
    monthly_returns = monthly.pct_change().dropna() * 100.0

    if monthly_returns.empty:
        return {
            "months": 0,
            "avg_monthly_pct": 0.0,
            "median_monthly_pct": 0.0,
            "best_month_pct": 0.0,
            "worst_month_pct": 0.0,
            "positive_month_pct": 0.0,
            "positive_months": 0,
            "negative_months": 0,
        }

    return {
        "months": int(len(monthly_returns)),
        "avg_monthly_pct": round(float(monthly_returns.mean()), 6),
        "median_monthly_pct": round(float(monthly_returns.median()), 6),
        "best_month_pct": round(float(monthly_returns.max()), 6),
        "worst_month_pct": round(float(monthly_returns.min()), 6),
        "positive_month_pct": round(float((monthly_returns > 0).mean() * 100.0), 6),
        "positive_months": int((monthly_returns > 0).sum()),
        "negative_months": int((monthly_returns < 0).sum()),
    }


def score_candidates(scored_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_df.copy()
    higher_better = [
        ("total_return_pct", 0.30),
        ("median_monthly_pct", 0.20),
        ("worst_month_pct", 0.15),
        ("max_drawdown_pct", 0.15),
        ("positive_month_pct", 0.10),
    ]
    lower_better = [
        ("total_trades", 0.10),
    ]

    df["robust_score"] = 0.0
    for column, weight in higher_better:
        rank_col = f"rank_{column}"
        df[rank_col] = df[column].rank(method="average", pct=True)
        df["robust_score"] += df[rank_col] * weight

    for column, weight in lower_better:
        rank_col = f"rank_low_{column}"
        df[rank_col] = 1.0 - df[column].rank(method="average", pct=True) + (1.0 / len(df))
        df["robust_score"] += df[rank_col] * weight

    df["robust_score"] = df["robust_score"].round(6)
    return df.sort_values(["robust_score", "total_return_pct"], ascending=[False, False]).reset_index(drop=True)


def evaluate_family(candidates: pd.DataFrame, df_test: pd.DataFrame, family: str) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(candidates)
    for idx, (_, row) in enumerate(candidates.iterrows(), start=1):
        label = f"{family.lower()}_robust_{idx}"
        print(f"[{family}] {idx}/{total} - scoring {row['model_path']}")
        actions_df, metrics, resolved_model_path, resolved_env_path = evaluate_model(
            model_path=str(row["model_path"]),
            env_config_path=str(row["env_config_path"]),
            df_test=df_test,
            label=label,
        )
        monthly = monthly_stats(actions_df)
        out = row.to_dict()
        out.update(metrics)
        out.update(monthly)
        out["resolved_model_path"] = resolved_model_path
        out["resolved_env_config_path"] = resolved_env_path
        rows.append(out)
    return pd.DataFrame(rows)


def render_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data.</p>"
    return df.to_html(index=False, classes="table", border=0)


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def build_dashboard(summary: dict, scored_df: pd.DataFrame, skipped_df: pd.DataFrame) -> str:
    top_table = render_table(
        scored_df[
            [
                "family",
                "seed",
                "total_return_pct",
                "median_monthly_pct",
                "worst_month_pct",
                "max_drawdown_pct",
                "positive_month_pct",
                "total_trades",
                "robust_score",
            ]
        ].head(12)
    )
    skipped_table = render_table(skipped_df[["family", "seed", "model_path", "skip_reason"]].head(20))
    summary_block = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <title>Robust Objective Scoring</title>
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
      <h1>Robust Objective Scoring</h1>
      <p>
        Candidate scoring based on monthly robustness instead of final return only.
        The composite score blends total return, median month, worst month, drawdown, monthly hit rate and turnover.
      </p>
    </div>
    <div class="card">
      <h2>Top ranked candidates</h2>
      {top_table}
    </div>
    <div class="card">
      <h2>Skipped candidates</h2>
      {skipped_table}
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
    ROBUST_SELECTION_JSON.parent.mkdir(parents=True, exist_ok=True)

    br_available, br_skipped = load_candidates(BR_RESULTS, "BR")
    sr_available, sr_skipped = load_candidates(SR_RESULTS, "SR")
    skipped_df = pd.concat([br_skipped, sr_skipped], ignore_index=True)

    print(f"Available candidates -> BR: {len(br_available)}, SR: {len(sr_available)}")
    print(f"Skipped candidates   -> BR: {len(br_skipped)}, SR: {len(sr_skipped)}")

    df_test = prepare_test_data()

    br_scored = evaluate_family(br_available, df_test, "BR")
    sr_scored = evaluate_family(sr_available, df_test, "SR")
    all_scored = pd.concat([br_scored, sr_scored], ignore_index=True)
    all_scored = score_candidates(all_scored)

    br_ranked = all_scored[all_scored["family"] == "BR"].reset_index(drop=True)
    sr_ranked = all_scored[all_scored["family"] == "SR"].reset_index(drop=True)

    br_ranked.to_csv(BR_SCORED_CSV, index=False)
    sr_ranked.to_csv(SR_SCORED_CSV, index=False)
    all_scored.to_csv(ALL_SCORED_CSV, index=False)

    summary = {
        "objective_weights": {
            "total_return_pct": 0.30,
            "median_monthly_pct": 0.20,
            "worst_month_pct": 0.15,
            "max_drawdown_pct": 0.15,
            "positive_month_pct": 0.10,
            "low_turnover": 0.10,
        },
        "available_candidates": {
            "br": int(len(br_available)),
            "sr": int(len(sr_available)),
        },
        "skipped_candidates": {
            "br": int(len(br_skipped)),
            "sr": int(len(sr_skipped)),
        },
        "top_overall": all_scored.head(10).to_dict(orient="records"),
        "selected_baseline_by_family": {
            "br": br_ranked.iloc[0].to_dict() if not br_ranked.empty else None,
            "sr": sr_ranked.iloc[0].to_dict() if not sr_ranked.empty else None,
        },
    }

    clean_summary = sanitize_for_json(summary)
    SUMMARY_JSON.write_text(json.dumps(clean_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    HTML_DASHBOARD.write_text(build_dashboard(summary, all_scored, skipped_df), encoding="utf-8")
    ROBUST_SELECTION_JSON.write_text(
        json.dumps(sanitize_for_json(summary["selected_baseline_by_family"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Robust scoring complete.")
    print(f"  BR scored: {BR_SCORED_CSV}")
    print(f"  SR scored: {SR_SCORED_CSV}")
    print(f"  All scored: {ALL_SCORED_CSV}")
    print(f"  Summary: {SUMMARY_JSON}")
    print(f"  Selection: {ROBUST_SELECTION_JSON}")
    print(f"  Dashboard: {HTML_DASHBOARD}")
    print()
    if not sr_ranked.empty:
        top_sr = sr_ranked.iloc[0]
        print(
            f"Top SR robust candidate: seed={int(top_sr['seed'])}, "
            f"return={top_sr['total_return_pct']:+.2f}%, "
            f"median_month={top_sr['median_monthly_pct']:+.3f}%, "
            f"worst_month={top_sr['worst_month_pct']:+.3f}%, "
            f"score={top_sr['robust_score']:.3f}"
        )
    if not br_ranked.empty:
        top_br = br_ranked.iloc[0]
        print(
            f"Top BR robust candidate: seed={int(top_br['seed'])}, "
            f"return={top_br['total_return_pct']:+.2f}%, "
            f"median_month={top_br['median_monthly_pct']:+.3f}%, "
            f"worst_month={top_br['worst_month_pct']:+.3f}%, "
            f"score={top_br['robust_score']:.3f}"
        )


if __name__ == "__main__":
    main()
