#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


BR_ACTIONS = Path("backtest/plots_oos/br_sr_comparison/br_best_actions.csv")
SR_ACTIONS = Path("backtest/plots_oos/br_sr_comparison/sr_best_actions.csv")

OUT_DIR = Path("backtest/plots_oos/regime_selector")
DIAGNOSTICS_CSV = OUT_DIR / "monthly_regime_diagnostics.csv"
REGIME_SUMMARY_CSV = OUT_DIR / "regime_summary.csv"
DECISIONS_CSV = OUT_DIR / "selector_decisions.csv"
SELECTOR_CURVE_CSV = OUT_DIR / "selector_equity_curve.csv"
MONTHLY_COMPARISON_CSV = OUT_DIR / "monthly_return_comparison.csv"
DECISION_QUALITY_CSV = OUT_DIR / "decision_quality.csv"
SUMMARY_JSON = OUT_DIR / "selector_summary.json"
HTML_DASHBOARD = OUT_DIR / "dashboard.html"

LOOKBACK_MONTHS = 6
DEFAULT_STRATEGY = "SR"
MIN_EDGE_PCT = 0.15
INITIAL_CAPITAL = 10000.0
SWITCH_COST_BPS = 6.0


def load_actions(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    df["Regime"] = pd.to_numeric(df["Regime"], errors="coerce").fillna(-1).astype(int)
    df["Action"] = pd.to_numeric(df["Action"], errors="coerce").fillna(0).astype(int)
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce").fillna(0).astype(int)
    df["Equity"] = pd.to_numeric(df["Equity"], errors="coerce").ffill().bfill()
    df["BarReturn"] = df["Equity"].pct_change().fillna(0.0)
    df["Month"] = df["Time"].dt.to_period("M").astype(str)
    df["Strategy"] = label
    return df


def align_actions(br_df: pd.DataFrame, sr_df: pd.DataFrame) -> pd.DataFrame:
    merged = br_df.merge(
        sr_df,
        on="Time",
        how="inner",
        suffixes=("_BR", "_SR"),
    )
    merged["Regime"] = merged["Regime_SR"]
    regime_match = merged["Regime_BR"] == merged["Regime_SR"]
    if not bool(regime_match.all()):
        merged.loc[~regime_match, "Regime"] = merged.loc[~regime_match, "Regime_SR"]

    merged["Month"] = merged["Time"].dt.to_period("M").astype(str)
    return merged[
        [
            "Time",
            "Month",
            "Regime",
            "Equity_BR",
            "Equity_SR",
            "BarReturn_BR",
            "BarReturn_SR",
            "Action_BR",
            "Action_SR",
            "Position_BR",
            "Position_SR",
        ]
    ].copy()


def summarize_return(series: pd.Series) -> tuple[float, float]:
    returns = pd.to_numeric(series, errors="coerce").fillna(0.0)
    gross = float((1.0 + returns).prod())
    total_return_pct = (gross - 1.0) * 100.0
    return gross, total_return_pct


def compute_monthly_regime_diagnostics(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy in ["BR", "SR"]:
        return_col = f"BarReturn_{strategy}"
        action_col = f"Action_{strategy}"
        grouped = aligned.groupby(["Month", "Regime"], sort=True)
        for (month, regime), chunk in grouped:
            gross_mult, total_return_pct = summarize_return(chunk[return_col])
            returns = pd.to_numeric(chunk[return_col], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "Month": month,
                    "Regime": int(regime),
                    "Strategy": strategy,
                    "Bars": int(len(chunk)),
                    "GrossMultiplier": round(gross_mult, 10),
                    "TotalReturnPct": round(total_return_pct, 6),
                    "AvgBarReturnBps": round(float(returns.mean() * 10000.0), 6),
                    "PositiveBarPct": round(float((returns > 0).mean() * 100.0), 4),
                    "ActiveStepPct": round(float((chunk[action_col] != 0).mean() * 100.0), 4),
                }
            )
    return pd.DataFrame(rows).sort_values(["Month", "Regime", "Strategy"]).reset_index(drop=True)


def compute_regime_summary(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy in ["BR", "SR"]:
        return_col = f"BarReturn_{strategy}"
        action_col = f"Action_{strategy}"
        grouped = aligned.groupby("Regime", sort=True)
        for regime, chunk in grouped:
            gross_mult, total_return_pct = summarize_return(chunk[return_col])
            returns = pd.to_numeric(chunk[return_col], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "Regime": int(regime),
                    "Strategy": strategy,
                    "Bars": int(len(chunk)),
                    "GrossMultiplier": round(gross_mult, 10),
                    "TotalReturnPct": round(total_return_pct, 6),
                    "AvgBarReturnBps": round(float(returns.mean() * 10000.0), 6),
                    "PositiveBarPct": round(float((returns > 0).mean() * 100.0), 4),
                    "ActiveStepPct": round(float((chunk[action_col] != 0).mean() * 100.0), 4),
                }
            )
    return pd.DataFrame(rows).sort_values(["Regime", "Strategy"]).reset_index(drop=True)


def sorted_months(month_values: pd.Series) -> list[str]:
    return [str(value) for value in sorted(pd.PeriodIndex(month_values.unique(), freq="M"))]


def choose_strategy(br_total_pct: float | None, sr_total_pct: float | None) -> str:
    if br_total_pct is None or sr_total_pct is None:
        return DEFAULT_STRATEGY
    # BR is allowed only when its trailing edge is both positive and materially better than SR.
    if br_total_pct > 0.0 and br_total_pct > sr_total_pct + MIN_EDGE_PCT:
        return "BR"
    return DEFAULT_STRATEGY


def build_decision_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    months = sorted_months(diagnostics["Month"])
    regimes = sorted(diagnostics["Regime"].unique())
    rows: list[dict] = []

    for month_idx, month in enumerate(months):
        history_months = months[max(0, month_idx - LOOKBACK_MONTHS):month_idx]
        for regime in regimes:
            history = diagnostics[
                diagnostics["Month"].isin(history_months) & (diagnostics["Regime"] == regime)
            ]
            br_history = history.loc[history["Strategy"] == "BR", "GrossMultiplier"]
            sr_history = history.loc[history["Strategy"] == "SR", "GrossMultiplier"]

            br_total_pct = None
            sr_total_pct = None
            if not br_history.empty:
                br_total_pct = (float(br_history.prod()) - 1.0) * 100.0
            if not sr_history.empty:
                sr_total_pct = (float(sr_history.prod()) - 1.0) * 100.0

            selected = choose_strategy(br_total_pct, sr_total_pct)
            rows.append(
                {
                    "Month": month,
                    "Regime": int(regime),
                    "LookbackMonths": len(history_months),
                    "HistoryWindow": ",".join(history_months),
                    "BRLookbackReturnPct": None if br_total_pct is None else round(br_total_pct, 6),
                    "SRLookbackReturnPct": None if sr_total_pct is None else round(sr_total_pct, 6),
                    "SelectedStrategy": selected,
                }
            )
    return pd.DataFrame(rows).sort_values(["Month", "Regime"]).reset_index(drop=True)


def curve_metrics(equity: pd.Series) -> dict:
    eq = pd.to_numeric(equity, errors="coerce").ffill().bfill().to_numpy(dtype=float)
    if eq.size == 0:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "final_capital": INITIAL_CAPITAL,
        }
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / (running_max + 1e-9) * 100.0
    final_capital = float(eq[-1])
    return {
        "total_return_pct": round((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0, 4),
        "max_drawdown_pct": round(float(drawdown.min()), 4),
        "final_capital": round(final_capital, 4),
    }


def build_selector_curve(aligned: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    decision_map = {
        (row["Month"], int(row["Regime"])): row["SelectedStrategy"] for _, row in decisions.iterrows()
    }
    selector_equity_gross = INITIAL_CAPITAL
    selector_equity_net = INITIAL_CAPITAL
    previous_strategy = None
    switch_count = 0
    switch_cost_total_pct = 0.0
    switch_cost_rate = SWITCH_COST_BPS / 10000.0
    rows: list[dict] = []

    for _, row in aligned.iterrows():
        regime = int(row["Regime"])
        month = row["Month"]
        selected = decision_map.get((month, regime), DEFAULT_STRATEGY)
        bar_return = float(row[f"BarReturn_{selected}"])

        did_switch = previous_strategy is not None and selected != previous_strategy
        if did_switch:
            switch_count += 1
            switch_cost_total_pct += SWITCH_COST_BPS / 100.0

        selector_equity_gross *= 1.0 + bar_return
        selector_equity_net *= (1.0 + bar_return) * (1.0 - (switch_cost_rate if did_switch else 0.0))
        previous_strategy = selected

        rows.append(
            {
                "Time": row["Time"],
                "Month": month,
                "Regime": regime,
                "SelectedStrategy": selected,
                "BR_Equity": float(row["Equity_BR"]),
                "SR_Equity": float(row["Equity_SR"]),
                "BR_Return": float(row["BarReturn_BR"]),
                "SR_Return": float(row["BarReturn_SR"]),
                "Selector_Return_Gross": bar_return,
                "Selector_Return_Net": ((1.0 + bar_return) * (1.0 - (switch_cost_rate if did_switch else 0.0))) - 1.0,
                "Selector_Equity_Gross": selector_equity_gross,
                "Selector_Equity_Net": selector_equity_net,
                "DidSwitch": did_switch,
                "SwitchCostBps": SWITCH_COST_BPS if did_switch else 0.0,
            }
        )

    selector_df = pd.DataFrame(rows)
    selector_df.attrs["switch_count"] = switch_count
    selector_df.attrs["switch_cost_total_pct"] = switch_cost_total_pct
    return selector_df


def compute_monthly_return_comparison(selector_curve: pd.DataFrame) -> pd.DataFrame:
    frame = selector_curve.copy()
    frame["Time"] = pd.to_datetime(frame["Time"], errors="coerce")
    frame = frame.dropna(subset=["Time"]).set_index("Time")

    rows = []
    for label, column in [
        ("BR", "BR_Equity"),
        ("SR", "SR_Equity"),
        ("Selector_Gross", "Selector_Equity_Gross"),
        ("Selector_Net", "Selector_Equity_Net"),
    ]:
        monthly = pd.to_numeric(frame[column], errors="coerce").resample("ME").last().dropna()
        monthly_returns = monthly.pct_change().dropna() * 100.0
        for timestamp, value in monthly_returns.items():
            rows.append(
                {
                    "Month": timestamp.strftime("%Y-%m"),
                    "Strategy": label,
                    "MonthlyReturnPct": round(float(value), 6),
                }
            )
    return pd.DataFrame(rows).sort_values(["Month", "Strategy"]).reset_index(drop=True)


def build_chart_payload(selector_curve: pd.DataFrame) -> dict:
    labels = pd.to_datetime(selector_curve["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    return {
        "labels": labels,
        "datasets": [
            {
                "label": "BR Best",
                "data": selector_curve["BR_Equity"].round(6).tolist(),
                "borderColor": "#5aa9ff",
                "backgroundColor": "#5aa9ff",
            },
            {
                "label": "SR Best",
                "data": selector_curve["SR_Equity"].round(6).tolist(),
                "borderColor": "#ffb347",
                "backgroundColor": "#ffb347",
            },
            {
                "label": "Adaptive Selector",
                "data": selector_curve["Selector_Equity_Gross"].round(6).tolist(),
                "borderColor": "#8de0b7",
                "backgroundColor": "#8de0b7",
                "borderDash": [6, 4],
            },
            {
                "label": "Adaptive Selector Net",
                "data": selector_curve["Selector_Equity_Net"].round(6).tolist(),
                "borderColor": "#3dd598",
                "backgroundColor": "#3dd598",
            },
        ],
    }


def html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data available.</p>"
    return df.to_html(index=False, classes="table", border=0)


def build_dashboard(
    summary: dict,
    selector_curve: pd.DataFrame,
    regime_summary: pd.DataFrame,
    decisions: pd.DataFrame,
    decision_quality: pd.DataFrame,
) -> str:
    chart_payload = json.dumps(build_chart_payload(selector_curve), ensure_ascii=False)

    overview_cards = f"""
    <div class="grid">
      <div class="card accent-blue">
        <h3>BR Best</h3>
        <div class="metric">{summary['br']['total_return_pct']:+.2f}%</div>
        <div>Max DD: {summary['br']['max_drawdown_pct']:.2f}%</div>
        <div>Final capital: {summary['br']['final_capital']:.2f}</div>
      </div>
      <div class="card accent-amber">
        <h3>SR Best</h3>
        <div class="metric">{summary['sr']['total_return_pct']:+.2f}%</div>
        <div>Max DD: {summary['sr']['max_drawdown_pct']:.2f}%</div>
        <div>Final capital: {summary['sr']['final_capital']:.2f}</div>
      </div>
      <div class="card accent-green">
        <h3>Selector Gross</h3>
        <div class="metric">{summary['selector_gross']['total_return_pct']:+.2f}%</div>
        <div>Max DD: {summary['selector_gross']['max_drawdown_pct']:.2f}%</div>
        <div>Final capital: {summary['selector_gross']['final_capital']:.2f}</div>
      </div>
      <div class="card accent-green">
        <h3>Selector Net</h3>
        <div class="metric">{summary['selector_net']['total_return_pct']:+.2f}%</div>
        <div>Max DD: {summary['selector_net']['max_drawdown_pct']:.2f}%</div>
        <div>Final capital: {summary['selector_net']['final_capital']:.2f}</div>
        <div>Switches: {summary['selector_net']['switch_count']}</div>
        <div>Switch cost total: {summary['selector_net']['switch_cost_total_pct']:.4f}%</div>
      </div>
    </div>
    """

    regime_table = html_table(
        regime_summary[
            ["Regime", "Strategy", "Bars", "TotalReturnPct", "AvgBarReturnBps", "PositiveBarPct", "ActiveStepPct"]
        ]
    )
    decisions_table = html_table(
        decisions[
            ["Month", "Regime", "LookbackMonths", "BRLookbackReturnPct", "SRLookbackReturnPct", "SelectedStrategy"]
        ]
    )
    quality_table = html_table(
        decision_quality[
            [
                "Month",
                "Regime",
                "SelectedStrategy",
                "OracleStrategy",
                "SelectedReturnPct",
                "OracleReturnPct",
                "LiftVsSRPct",
                "DecisionRegretPct",
                "IsCorrect",
            ]
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <title>Adaptive Regime Selector</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #08111f;
      --panel: #0f1c2e;
      --line: rgba(154, 176, 200, 0.18);
      --text: #e8eef7;
      --muted: #9ab0c8;
      --blue: #5aa9ff;
      --amber: #ffb347;
      --green: #3dd598;
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
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .card, .chart-wrap {{
      background: rgba(15, 28, 46, 0.9);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 14px 30px rgba(0, 0, 0, 0.18);
    }}
    .hero {{
      padding: 24px;
    }}
    .section {{
      margin-top: 20px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .card {{
      padding: 18px;
    }}
    .accent-blue {{ border-color: rgba(90, 169, 255, 0.36); }}
    .accent-amber {{ border-color: rgba(255, 179, 71, 0.36); }}
    .accent-green {{ border-color: rgba(61, 213, 152, 0.36); }}
    .metric {{
      font-size: 30px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .chart-wrap {{
      padding: 18px;
      margin-top: 16px;
    }}
    canvas {{
      width: 100%;
      height: 420px;
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
      font-weight: 600;
    }}
    .muted {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Adaptive SR/BR Selector</h1>
      <p>
        Monthly report by regime and a first adaptive selector. Every month, the selector picks BR or SR
        for each regime, using the previous {LOOKBACK_MONTHS} months as the comparison window.
      </p>
      <p class="muted">
        Note: this is a research proxy based on stitching per-bar equity returns.
        It does not yet include an explicit model of the cost of switching between strategies.
      </p>
    </section>

    <section class="section">
      {overview_cards}
      <div class="chart-wrap">
        <h2>Equity curve comparison</h2>
        <canvas id="equityChart"></canvas>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2>Regime summary</h2>
        {regime_table}
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2>Monthly decisions</h2>
        {decisions_table}
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2>Decision quality</h2>
        {quality_table}
      </div>
    </section>
  </div>

  <script>
    const payload = {chart_payload};
    const ctx = document.getElementById('equityChart').getContext('2d');
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: payload.labels,
        datasets: payload.datasets.map(ds => ({{
          ...ds,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.12
        }}))
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        interaction: {{
          mode: 'index',
          intersect: false
        }},
        plugins: {{
          legend: {{
            labels: {{
              color: '#e8eef7'
            }}
          }},
          tooltip: {{
            callbacks: {{
              label: function(context) {{
                return `${{context.dataset.label}}: $${{Number(context.parsed.y).toFixed(2)}}`;
              }}
            }}
          }}
        }},
        scales: {{
          x: {{
            ticks: {{
              color: '#9ab0c8',
              maxTicksLimit: 10
            }},
            grid: {{
              color: 'rgba(154, 176, 200, 0.12)'
            }}
          }},
          y: {{
            ticks: {{
              color: '#9ab0c8',
              callback: function(value) {{
                return '$' + Number(value).toFixed(0);
              }}
            }},
            grid: {{
              color: 'rgba(154, 176, 200, 0.12)'
            }}
          }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""


def compute_decision_quality(aligned: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, decision in decisions.iterrows():
        month = decision["Month"]
        regime = int(decision["Regime"])
        chunk = aligned[(aligned["Month"] == month) & (aligned["Regime"] == regime)]
        if chunk.empty:
            continue

        _, br_total_pct = summarize_return(chunk["BarReturn_BR"])
        _, sr_total_pct = summarize_return(chunk["BarReturn_SR"])
        selected = decision["SelectedStrategy"]
        selected_total_pct = br_total_pct if selected == "BR" else sr_total_pct

        if br_total_pct > sr_total_pct:
            oracle_strategy = "BR"
            oracle_total_pct = br_total_pct
        elif sr_total_pct > br_total_pct:
            oracle_strategy = "SR"
            oracle_total_pct = sr_total_pct
        else:
            oracle_strategy = DEFAULT_STRATEGY
            oracle_total_pct = sr_total_pct

        rows.append(
            {
                "Month": month,
                "Regime": regime,
                "Bars": int(len(chunk)),
                "SelectedStrategy": selected,
                "OracleStrategy": oracle_strategy,
                "BRRealizedReturnPct": round(br_total_pct, 6),
                "SRRealizedReturnPct": round(sr_total_pct, 6),
                "SelectedReturnPct": round(selected_total_pct, 6),
                "OracleReturnPct": round(oracle_total_pct, 6),
                "LiftVsSRPct": round(selected_total_pct - sr_total_pct, 6),
                "DecisionRegretPct": round(oracle_total_pct - selected_total_pct, 6),
                "WinnerMarginPct": round(abs(br_total_pct - sr_total_pct), 6),
                "IsCorrect": bool(selected == oracle_strategy),
            }
        )
    return pd.DataFrame(rows).sort_values(["Month", "Regime"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    br_df = load_actions(BR_ACTIONS, "BR")
    sr_df = load_actions(SR_ACTIONS, "SR")
    aligned = align_actions(br_df, sr_df)

    diagnostics = compute_monthly_regime_diagnostics(aligned)
    regime_summary = compute_regime_summary(aligned)
    decisions = build_decision_table(diagnostics)
    selector_curve = build_selector_curve(aligned, decisions)
    decision_quality = compute_decision_quality(aligned, decisions)
    monthly_comparison = compute_monthly_return_comparison(selector_curve)

    br_metrics = curve_metrics(selector_curve["BR_Equity"])
    sr_metrics = curve_metrics(selector_curve["SR_Equity"])
    selector_gross_metrics = curve_metrics(selector_curve["Selector_Equity_Gross"])
    selector_net_metrics = curve_metrics(selector_curve["Selector_Equity_Net"])
    selector_net_metrics["switch_count"] = int(selector_curve.attrs.get("switch_count", 0))
    selector_net_metrics["switch_cost_total_pct"] = round(
        float(selector_curve.attrs.get("switch_cost_total_pct", 0.0)),
        6,
    )

    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    regime_summary.to_csv(REGIME_SUMMARY_CSV, index=False)
    decisions.to_csv(DECISIONS_CSV, index=False)
    selector_curve.to_csv(SELECTOR_CURVE_CSV, index=False)
    monthly_comparison.to_csv(MONTHLY_COMPARISON_CSV, index=False)
    decision_quality.to_csv(DECISION_QUALITY_CSV, index=False)

    selection_counts = (
        decisions.groupby(["Regime", "SelectedStrategy"]).size().reset_index(name="MonthsSelected")
    )
    quality_summary = {
        "total_decisions": int(len(decision_quality)),
        "correct_decisions": int(decision_quality["IsCorrect"].sum()) if not decision_quality.empty else 0,
        "incorrect_decisions": int((~decision_quality["IsCorrect"]).sum()) if not decision_quality.empty else 0,
        "avg_lift_vs_sr_pct": round(
            float(decision_quality["LiftVsSRPct"].mean()) if not decision_quality.empty else 0.0,
            6,
        ),
        "avg_regret_pct": round(
            float(decision_quality["DecisionRegretPct"].mean()) if not decision_quality.empty else 0.0,
            6,
        ),
        "br_calls": int((decision_quality["SelectedStrategy"] == "BR").sum()) if not decision_quality.empty else 0,
        "sr_calls": int((decision_quality["SelectedStrategy"] == "SR").sum()) if not decision_quality.empty else 0,
    }

    summary = {
        "config": {
            "lookback_months": LOOKBACK_MONTHS,
            "default_strategy": DEFAULT_STRATEGY,
            "min_edge_pct": MIN_EDGE_PCT,
            "initial_capital": INITIAL_CAPITAL,
            "switch_cost_bps": SWITCH_COST_BPS,
        },
        "br": br_metrics,
        "sr": sr_metrics,
        "selector_gross": selector_gross_metrics,
        "selector_net": selector_net_metrics,
        "selection_counts": selection_counts.to_dict(orient="records"),
        "decision_quality_summary": quality_summary,
        "notes": [
            "The selector uses the previous months to pick BR or SR for each regime.",
            "This version applies an explicit switch cost when the selector changes strategy.",
            "The natural next step is to add confidence filters and a cost-sensitivity test.",
        ],
        "outputs": {
            "diagnostics_csv": str(DIAGNOSTICS_CSV),
            "regime_summary_csv": str(REGIME_SUMMARY_CSV),
            "decisions_csv": str(DECISIONS_CSV),
            "selector_curve_csv": str(SELECTOR_CURVE_CSV),
            "monthly_comparison_csv": str(MONTHLY_COMPARISON_CSV),
            "decision_quality_csv": str(DECISION_QUALITY_CSV),
            "dashboard_html": str(HTML_DASHBOARD),
        },
    }

    HTML_DASHBOARD.write_text(
        build_dashboard(summary, selector_curve, regime_summary, decisions, decision_quality),
        encoding="utf-8",
    )
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Adaptive regime selector built.")
    print(f"  Diagnostics: {DIAGNOSTICS_CSV}")
    print(f"  Decisions:   {DECISIONS_CSV}")
    print(f"  Curve:       {SELECTOR_CURVE_CSV}")
    print(f"  Dashboard:   {HTML_DASHBOARD}")
    print()
    print("Performance summary:")
    print(f"  BR best:      {br_metrics['total_return_pct']:+.2f}% | DD {br_metrics['max_drawdown_pct']:.2f}%")
    print(f"  SR best:      {sr_metrics['total_return_pct']:+.2f}% | DD {sr_metrics['max_drawdown_pct']:.2f}%")
    print(
        f"  Selector gross: {selector_gross_metrics['total_return_pct']:+.2f}% | "
        f"DD {selector_gross_metrics['max_drawdown_pct']:.2f}%"
    )
    print(
        f"  Selector net:   {selector_net_metrics['total_return_pct']:+.2f}% | "
        f"DD {selector_net_metrics['max_drawdown_pct']:.2f}% | "
        f"switches {selector_net_metrics['switch_count']}"
    )


if __name__ == "__main__":
    main()
