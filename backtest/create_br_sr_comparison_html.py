#!/usr/bin/env python3
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from backtest.out_of_sample_test import _run_slice, split_train_test_by_date
from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter


BR_RESULTS = Path("backtest/plots_oos/feature_grid_multi_seed/results.csv")
SR_RESULTS = Path("backtest/plots_oos/feature_grid_multi_seed_sr/results.csv")
COMBINED_CONFIG = Path("best_models/combined_config.json")
COPIED_MODELS_DIR = Path("best_models/copied")

OUT_DIR = Path("backtest/plots_oos/br_sr_comparison")
OUT_HTML = OUT_DIR / "br_vs_sr_best_comparison.html"
OUT_BR_CSV = OUT_DIR / "br_best_actions.csv"
OUT_SR_CSV = OUT_DIR / "sr_best_actions.csv"
OUT_CURVES_CSV = OUT_DIR / "comparison_equity_curves.csv"
OUT_META_JSON = OUT_DIR / "comparison_summary.json"


def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    fallback = COPIED_MODELS_DIR / path.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"File not found: {path_str}")


def normalize_weights(values: list[float]) -> list[float]:
    if not values:
        return []

    arr = np.asarray(values, dtype=float)
    total = float(arr.sum())
    if abs(total) < 1e-12:
        return [1.0 / len(arr)] * len(arr)
    return (arr / total).tolist()


def load_env_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        return {}
    return cfg


def prepare_test_data(
    train_end: str = "2022-12-31 23:00:00",
    val_end: str = "2023-12-31 23:00:00",
) -> pd.DataFrame:
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col="Close")
    df["Close"] = df["Close_KF"]
    _, df_test = split_train_test_by_date(df, train_end=train_end, val_end=val_end)
    return df_test.reset_index(drop=True)


def evaluate_model(model_path: str, env_config_path: str, df_test: pd.DataFrame, label: str):
    resolved_model_path = resolve_existing_path(model_path)
    resolved_env_path = resolve_existing_path(env_config_path)

    env_cfg = load_env_config(str(resolved_env_path))
    env_cfg.setdefault("enable_zones", False)
    env_cfg.setdefault("zone_tol", 0.002)
    env_cfg.setdefault("zone_lookback", 5)

    model = PPO.load(str(resolved_model_path))
    actions_df, metrics = _run_slice(
        model=model,
        df_slice=df_test,
        env_cfg=env_cfg,
        allow_hold_when_flat=True,
        label=label,
    )
    return actions_df, metrics, str(resolved_model_path), str(resolved_env_path)


def best_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values("total_return_pct", ascending=False).iloc[0]


def row_for_model_path(df: pd.DataFrame, model_path: str) -> pd.Series:
    match = df.loc[df["model_path"] == model_path]
    if match.empty:
        raise KeyError(f"Model path not present in results.csv: {model_path}")
    return match.iloc[0]


def actions_to_curve(df: pd.DataFrame) -> pd.DataFrame:
    curve = df[["Time", "Equity"]].copy()
    curve["Time"] = pd.to_datetime(curve["Time"], errors="coerce")
    curve["Equity"] = pd.to_numeric(curve["Equity"], errors="coerce")
    curve = curve.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return curve


def align_curves(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for name, curve in curves.items():
        current = curve.rename(columns={"Equity": name}).copy()
        current["Time"] = pd.to_datetime(current["Time"], errors="coerce")
        current = current.dropna(subset=["Time"]).sort_values("Time")
        if merged is None:
            merged = current
        else:
            merged = merged.merge(current, on="Time", how="outer")

    if merged is None:
        return pd.DataFrame(columns=["Time"])

    merged = merged.sort_values("Time").drop_duplicates(subset=["Time"]).reset_index(drop=True)
    value_cols = [col for col in merged.columns if col != "Time"]
    if value_cols:
        merged[value_cols] = merged[value_cols].apply(pd.to_numeric, errors="coerce")
        merged[value_cols] = merged[value_cols].ffill().bfill()
    return merged


def portfolio_curve(actions_frames: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    component_curves = {
        f"component_{idx + 1}": actions_to_curve(actions_df) for idx, actions_df in enumerate(actions_frames)
    }
    aligned = align_curves(component_curves)
    if aligned.empty:
        return pd.DataFrame(columns=["Time", "Equity"])

    aligned["Equity"] = 0.0
    for idx, weight in enumerate(weights, start=1):
        aligned["Equity"] += aligned[f"component_{idx}"] * float(weight)
    return aligned[["Time", "Equity"]].copy()


def curve_metrics(curve_df: pd.DataFrame) -> dict:
    equities = pd.to_numeric(curve_df["Equity"], errors="coerce").dropna().to_numpy(dtype=float)
    if equities.size == 0:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "final_capital": 10000.0,
        }

    running_max = np.maximum.accumulate(equities)
    drawdowns = (equities - running_max) / (running_max + 1e-9) * 100.0
    final_capital = float(equities[-1])
    return {
        "total_return_pct": round((final_capital - 10000.0) / 10000.0 * 100.0, 4),
        "max_drawdown_pct": round(float(drawdowns.min()), 4),
        "final_capital": round(final_capital, 4),
    }


def build_chart_payload(curves_df: pd.DataFrame, series: list[tuple[str, str, str]]) -> dict:
    labels = [str(ts) for ts in pd.to_datetime(curves_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")]
    datasets = []
    for column, label, color in series:
        values = []
        for value in pd.to_numeric(curves_df[column], errors="coerce").tolist():
            if pd.isna(value):
                values.append(None)
            else:
                values.append(round(float(value), 6))
        datasets.append(
            {
                "label": label,
                "data": values,
                "borderColor": color,
                "backgroundColor": color,
            }
        )
    return {"labels": labels, "datasets": datasets}


def metrics_block(name: str, row: pd.Series, metrics: dict, resolved_model_path: str, resolved_env_path: str) -> dict:
    result = {
        "name": name,
        "seed": int(row["seed"]) if "seed" in row else None,
        "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 4),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 4),
        "pct_profitable": round(float(metrics.get("pct_profitable", 0.0)), 4),
        "total_trades": int(metrics.get("total_trades", 0)),
        "final_capital": round(float(metrics.get("final_capital", 10000.0)), 4),
        "model_path": str(row.get("model_path", "")),
        "env_config_path": str(row.get("env_config_path", "")),
        "resolved_model_path": resolved_model_path,
        "resolved_env_config_path": resolved_env_path,
    }
    if "lookbacks" in row:
        result["lookbacks"] = str(row["lookbacks"])
    if "quantile" in row:
        result["quantile"] = float(row["quantile"])
    if "zone_tol" in row:
        result["zone_tol"] = float(row["zone_tol"])
    if "zone_lookback" in row:
        result["zone_lookback"] = int(row["zone_lookback"])
    return result


def component_summary(row: pd.Series, weight: float, metrics: dict, resolved_model_path: str, resolved_env_path: str) -> dict:
    info = metrics_block(
        name=f"Component seed {int(row['seed'])}",
        row=row,
        metrics=metrics,
        resolved_model_path=resolved_model_path,
        resolved_env_path=resolved_env_path,
    )
    info["weight"] = round(float(weight), 6)
    return info


def portfolio_summary(name: str, curve_df: pd.DataFrame, components: list[dict], weights: list[float]) -> dict:
    summary = curve_metrics(curve_df)
    summary.update(
        {
            "name": name,
            "component_count": len(components),
            "weights": [round(float(weight), 6) for weight in weights],
            "components": components,
            "component_trade_count_sum": int(sum(item.get("total_trades", 0) for item in components)),
            "weighted_win_rate_pct": round(
                float(
                    sum(
                        item.get("pct_profitable", 0.0) * float(weight)
                        for item, weight in zip(components, weights)
                    )
                ),
                4,
            ),
        }
    )
    return summary


def render_lines(items: list[tuple[str, str]]) -> str:
    rows = []
    for label, value in items:
        rows.append(f"<div><span class='label'>{html.escape(label)}:</span> {html.escape(str(value))}</div>")
    return "\n".join(rows)


def render_card(title: str, info: dict, fields: list[tuple[str, str]], accent_class: str) -> str:
    lines = []
    for label, key in fields:
        value = info.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            value_str = f"{value:.4f}".rstrip("0").rstrip(".")
        else:
            value_str = str(value)
        lines.append((label, value_str))

    metric_class = "pos" if float(info.get("total_return_pct", 0.0)) >= 0 else "neg"
    return f"""
    <div class="card {accent_class}">
      <h3>{html.escape(title)}</h3>
      <div class="metric {metric_class}">{float(info.get('total_return_pct', 0.0)):+.2f}%</div>
      {render_lines(lines)}
    </div>
    """


def generate_html(summary: dict, best_chart: dict, weighted_chart: dict, all_chart: dict) -> str:
    best_payload = json.dumps(best_chart, ensure_ascii=False)
    weighted_payload = json.dumps(weighted_chart, ensure_ascii=False)
    all_payload = json.dumps(all_chart, ensure_ascii=False)

    best_block = html.escape(json.dumps(summary["single_best"], ensure_ascii=False, indent=2))
    weighted_block = html.escape(json.dumps(summary["weighted_top3"], ensure_ascii=False, indent=2))
    all_block = html.escape(json.dumps(summary["all6_weighted"], ensure_ascii=False, indent=2))

    br_best = summary["single_best"]["br"]
    sr_best = summary["single_best"]["sr"]
    br_weighted = summary["weighted_top3"]["br"]
    sr_weighted = summary["weighted_top3"]["sr"]
    all_weighted = summary["all6_weighted"]

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <title>BR vs SR comparison - OOS Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #08111f;
      --panel: #0f1c2e;
      --panel-2: #13243a;
      --text: #e8eef7;
      --muted: #9ab0c8;
      --line: rgba(154, 176, 200, 0.18);
      --good: #3dd598;
      --bad: #ff7676;
      --blue: #5aa9ff;
      --amber: #ffb347;
      --sand: #f7d08a;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(90, 169, 255, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 179, 71, 0.16), transparent 30%),
        linear-gradient(180deg, #07111c 0%, #0b1727 100%);
      color: var(--text);
    }}

    .page {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px;
    }}

    h1, h2, h3 {{
      margin: 0 0 12px 0;
    }}

    p {{
      margin: 0;
      line-height: 1.6;
    }}

    .hero {{
      display: grid;
      gap: 14px;
      padding: 24px;
      border: 1px solid var(--line);
      background: rgba(15, 28, 46, 0.82);
      border-radius: 18px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.22);
    }}

    .hero .eyebrow {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 12px;
    }}

    .section {{
      margin-top: 22px;
      display: grid;
      gap: 14px;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}

    .card {{
      padding: 18px;
      border-radius: 16px;
      background: rgba(15, 28, 46, 0.9);
      border: 1px solid var(--line);
      box-shadow: 0 14px 30px rgba(0, 0, 0, 0.18);
    }}

    .accent-blue {{ border-color: rgba(90, 169, 255, 0.34); }}
    .accent-amber {{ border-color: rgba(255, 179, 71, 0.34); }}
    .accent-sand {{ border-color: rgba(247, 208, 138, 0.34); }}

    .metric {{
      font-size: 30px;
      font-weight: 700;
      margin-bottom: 10px;
    }}

    .pos {{ color: var(--good); }}
    .neg {{ color: var(--bad); }}

    .label {{
      color: var(--muted);
    }}

    .chart-wrap {{
      padding: 18px;
      border-radius: 18px;
      background: rgba(15, 28, 46, 0.9);
      border: 1px solid var(--line);
    }}

    canvas {{
      width: 100%;
      height: 420px;
    }}

    .details {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}

    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      padding: 14px;
      border-radius: 14px;
      background: rgba(7, 17, 28, 0.9);
      border: 1px solid var(--line);
      color: #d9e6f5;
      overflow-x: auto;
    }}

    .note {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Out-of-sample dashboard</div>
      <h1>BR vs SR comparison on the OOS equity curves</h1>
      <p>
        This dashboard continues the previous work: instead of comparing only the final returns in
        <code>results.csv</code>, it shows over time how the best single model and the
        top3 / all6 weighted portfolios behave on the same OOS test.
      </p>
      <p class="note">
        When a model path in the results was no longer available, it was loaded from
        the files copied into <code>best_models/copied</code>.
      </p>
    </section>

    <section class="section">
      <h2>Best singoli</h2>
      <div class="grid">
        {render_card("BR Best", br_best, [("Seed", "seed"), ("Lookbacks", "lookbacks"), ("Quantile", "quantile"), ("Max DD %", "max_drawdown_pct"), ("Trades", "total_trades"), ("Final capital", "final_capital")], "accent-blue")}
        {render_card("SR Best", sr_best, [("Seed", "seed"), ("Zone tol", "zone_tol"), ("Zone lookback", "zone_lookback"), ("Max DD %", "max_drawdown_pct"), ("Trades", "total_trades"), ("Final capital", "final_capital")], "accent-amber")}
      </div>
      <div class="chart-wrap">
        <h3>Equity curve - best vs best</h3>
        <canvas id="bestChart"></canvas>
      </div>
    </section>

    <section class="section">
      <h2>Top3 weighted</h2>
      <div class="grid">
        {render_card("BR Top3 Weighted", br_weighted, [("Components", "component_count"), ("Weights", "weights"), ("Max DD %", "max_drawdown_pct"), ("Weighted win rate %", "weighted_win_rate_pct"), ("Trade sum", "component_trade_count_sum"), ("Final capital", "final_capital")], "accent-blue")}
        {render_card("SR Top3 Weighted", sr_weighted, [("Components", "component_count"), ("Weights", "weights"), ("Max DD %", "max_drawdown_pct"), ("Weighted win rate %", "weighted_win_rate_pct"), ("Trade sum", "component_trade_count_sum"), ("Final capital", "final_capital")], "accent-amber")}
      </div>
      <div class="chart-wrap">
        <h3>Equity curve - BR top3 weighted vs SR top3 weighted</h3>
        <canvas id="weightedChart"></canvas>
      </div>
    </section>

    <section class="section">
      <h2>All 6 weighted</h2>
      <div class="grid">
        {render_card("All 6 Weighted", all_weighted, [("Components", "component_count"), ("Weights", "weights"), ("Max DD %", "max_drawdown_pct"), ("Weighted win rate %", "weighted_win_rate_pct"), ("Trade sum", "component_trade_count_sum"), ("Final capital", "final_capital")], "accent-sand")}
      </div>
      <div class="chart-wrap">
        <h3>Equity curve - weighted portfolios together</h3>
        <canvas id="allChart"></canvas>
      </div>
    </section>

    <section class="section">
      <h2>Methodology</h2>
      <div class="card">
        <p>
          The old <code>compare_strategies.py</code> only compared final aggregate metrics.
          This version re-evaluates the models on the same OOS window and builds aligned time curves.
          The weighted portfolios use the weights saved in <code>best_models/combined_config.json</code>,
          applied to the equity curves of the components.
        </p>
      </div>
      <div class="details">
        <div class="card">
          <h3>Best singoli</h3>
          <pre>{best_block}</pre>
        </div>
        <div class="card">
          <h3>Top3 weighted</h3>
          <pre>{weighted_block}</pre>
        </div>
        <div class="card">
          <h3>All 6 weighted</h3>
          <pre>{all_block}</pre>
        </div>
      </div>
    </section>
  </div>

  <script>
    const charts = {{
      best: {best_payload},
      weighted: {weighted_payload},
      all: {all_payload}
    }};

    function buildChart(canvasId, payload) {{
      const ctx = document.getElementById(canvasId).getContext('2d');
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
    }}

    buildChart('bestChart', charts.best);
    buildChart('weightedChart', charts.weighted);
    buildChart('allChart', charts.all);
  </script>
</body>
</html>
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    br_df = pd.read_csv(BR_RESULTS)
    sr_df = pd.read_csv(SR_RESULTS)
    with open(COMBINED_CONFIG, "r", encoding="utf-8") as handle:
        combined_cfg = json.load(handle)

    br_best_row = best_row(br_df)
    sr_best_row = best_row(sr_df)
    df_test = prepare_test_data()

    br_best_actions, br_best_metrics, br_best_model_resolved, br_best_env_resolved = evaluate_model(
        model_path=str(br_best_row["model_path"]),
        env_config_path=str(br_best_row["env_config_path"]),
        df_test=df_test,
        label="br_best_comparison",
    )
    sr_best_actions, sr_best_metrics, sr_best_model_resolved, sr_best_env_resolved = evaluate_model(
        model_path=str(sr_best_row["model_path"]),
        env_config_path=str(sr_best_row["env_config_path"]),
        df_test=df_test,
        label="sr_best_comparison",
    )

    br_best_actions.to_csv(OUT_BR_CSV, index=False)
    sr_best_actions.to_csv(OUT_SR_CSV, index=False)

    br_best_info = metrics_block(
        "BR Best", br_best_row, br_best_metrics, br_best_model_resolved, br_best_env_resolved
    )
    sr_best_info = metrics_block(
        "SR Best", sr_best_row, sr_best_metrics, sr_best_model_resolved, sr_best_env_resolved
    )

    br_weighted_actions = []
    br_weighted_components = []
    br_weights = normalize_weights([float(weight) for weight in combined_cfg["br_weights"]])
    for idx, (model_cfg, weight) in enumerate(zip(combined_cfg["br_models"], br_weights), start=1):
        row = row_for_model_path(br_df, model_cfg["model_path"])
        actions_df, metrics, resolved_model, resolved_env = evaluate_model(
            model_path=model_cfg["model_path"],
            env_config_path=model_cfg["env_config_path"],
            df_test=df_test,
            label=f"br_weighted_{idx}",
        )
        br_weighted_actions.append(actions_df)
        br_weighted_components.append(component_summary(row, weight, metrics, resolved_model, resolved_env))

    sr_weighted_actions = []
    sr_weighted_components = []
    sr_weights = normalize_weights([float(weight) for weight in combined_cfg["sr_weights"]])
    for idx, (model_cfg, weight) in enumerate(zip(combined_cfg["sr_models"], sr_weights), start=1):
        row = row_for_model_path(sr_df, model_cfg["model_path"])
        actions_df, metrics, resolved_model, resolved_env = evaluate_model(
            model_path=model_cfg["model_path"],
            env_config_path=model_cfg["env_config_path"],
            df_test=df_test,
            label=f"sr_weighted_{idx}",
        )
        sr_weighted_actions.append(actions_df)
        sr_weighted_components.append(component_summary(row, weight, metrics, resolved_model, resolved_env))

    br_weighted_curve = portfolio_curve(br_weighted_actions, br_weights)
    sr_weighted_curve = portfolio_curve(sr_weighted_actions, sr_weights)

    all_weights = normalize_weights(br_weights + sr_weights)
    all_components = [
        dict(component, weight=round(float(weight), 6))
        for component, weight in zip(br_weighted_components + sr_weighted_components, all_weights)
    ]
    all_weighted_curve = portfolio_curve(br_weighted_actions + sr_weighted_actions, all_weights)

    comparison_curves = align_curves(
        {
            "BR_Best": actions_to_curve(br_best_actions),
            "SR_Best": actions_to_curve(sr_best_actions),
            "BR_Top3_Weighted": br_weighted_curve,
            "SR_Top3_Weighted": sr_weighted_curve,
            "All6_Weighted": all_weighted_curve,
        }
    )
    comparison_curves["Time"] = pd.to_datetime(comparison_curves["Time"], errors="coerce")
    comparison_curves.to_csv(OUT_CURVES_CSV, index=False)

    br_weighted_info = portfolio_summary(
        "BR Top3 Weighted", br_weighted_curve, br_weighted_components, br_weights
    )
    sr_weighted_info = portfolio_summary(
        "SR Top3 Weighted", sr_weighted_curve, sr_weighted_components, sr_weights
    )
    all_weighted_info = portfolio_summary(
        "All 6 Weighted", all_weighted_curve, all_components, all_weights
    )

    summary = {
        "comparison_type": "oos_equity_comparison_dashboard",
        "output_html": str(OUT_HTML),
        "output_equity_csv": str(OUT_CURVES_CSV),
        "single_best": {
            "br": br_best_info,
            "sr": sr_best_info,
        },
        "weighted_top3": {
            "br": br_weighted_info,
            "sr": sr_weighted_info,
        },
        "all6_weighted": all_weighted_info,
        "notes": [
            "The old compare_strategies.py only compared final aggregate returns.",
            "This dashboard adds time curves for the single best models, top3 weighted and all6 weighted.",
            "When an original model_path no longer existed, loading used best_models/copied.",
        ],
    }

    best_chart = build_chart_payload(
        comparison_curves,
        [
            ("BR_Best", "BR Best Equity", "#5aa9ff"),
            ("SR_Best", "SR Best Equity", "#ffb347"),
        ],
    )
    weighted_chart = build_chart_payload(
        comparison_curves,
        [
            ("BR_Top3_Weighted", "BR Top3 Weighted", "#5aa9ff"),
            ("SR_Top3_Weighted", "SR Top3 Weighted", "#ffb347"),
        ],
    )
    all_chart = build_chart_payload(
        comparison_curves,
        [
            ("BR_Top3_Weighted", "BR Top3 Weighted", "#5aa9ff"),
            ("SR_Top3_Weighted", "SR Top3 Weighted", "#ffb347"),
            ("All6_Weighted", "All 6 Weighted", "#f7d08a"),
        ],
    )

    html_text = generate_html(summary, best_chart, weighted_chart, all_chart)
    OUT_HTML.write_text(html_text, encoding="utf-8")
    OUT_META_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Created the BR/SR comparison dashboard:")
    print(f"  HTML: {OUT_HTML}")
    print(f"  Curves CSV: {OUT_CURVES_CSV}")
    print(f"  BR actions: {OUT_BR_CSV}")
    print(f"  SR actions: {OUT_SR_CSV}")
    print(f"  Summary: {OUT_META_JSON}")
    print()
    print("Main results:")
    print(f"  BR best: {br_best_info['total_return_pct']:+.2f}%")
    print(f"  SR best: {sr_best_info['total_return_pct']:+.2f}%")
    print(f"  BR top3 weighted: {br_weighted_info['total_return_pct']:+.2f}%")
    print(f"  SR top3 weighted: {sr_weighted_info['total_return_pct']:+.2f}%")
    print(f"  All 6 weighted: {all_weighted_info['total_return_pct']:+.2f}%")


if __name__ == "__main__":
    main()
