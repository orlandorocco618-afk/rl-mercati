#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

import json
import os
from pathlib import Path
import itertools
import pandas as pd

from stable_baselines3 import PPO
from backtest.out_of_sample_test import _run_slice, split_train_test_by_date
from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter

ROOT = Path('.')
OUT_DIR = ROOT / 'backtest' / 'plots_oos'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path('models/ppo_trading.zip')
ENV_CFG_PATH = Path('models/multi_seed_runs/seed_33/stage2_env_config.json')


def load_data(train_end, val_end):
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col='Close')
    df['Close'] = df['Close_KF']
    _, df_test = split_train_test_by_date(df, train_end=train_end, val_end=val_end)
    return df_test


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not ENV_CFG_PATH.exists():
        raise FileNotFoundError(f"Env config not found: {ENV_CFG_PATH}")

    with open(ENV_CFG_PATH, 'r', encoding='utf-8') as f:
        base_cfg = json.load(f)

    train_end = os.getenv('RL_TRAIN_END', '2022-12-31 23:00:00')
    val_end = os.getenv('RL_VAL_END', '2023-12-31 23:00:00')

    df_test = load_data(train_end, val_end)

    model = PPO.load(str(MODEL_PATH))

    grid = {
        'turnover_penalty_coef': [0.0001, 0.0002, 0.00035, 0.0005],
        'breakout_entry_strength_quantile': [0.4, 0.45, 0.5, 0.55],
    }
    keys = list(grid.keys())
    rows = []

    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(base_cfg)
        # disable breakout to match old model obs space
        cfg['breakout_lookbacks'] = []
        for k, v in zip(keys, vals):
            cfg[k] = v
        label = 'grid_' + '_'.join([f"{k.split('_')[-1]}{str(v).replace('.', '')}" for k, v in zip(keys, vals)])
        print(f"Running {label} -> {dict(zip(keys, vals))}")
        actions_df, metrics = _run_slice(model=model, df_slice=df_test, env_cfg=cfg, allow_hold_when_flat=True, label=label)
        metrics_out = dict(cfg_overrides=dict(zip(keys, vals)), **metrics)
        rows.append(metrics_out)

    csv_path = OUT_DIR / 'turnover_strength_grid_results.csv'
    json_path = OUT_DIR / 'turnover_strength_grid_results.json'
    df_out = pd.DataFrame(rows)
    df_out.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    html_path = OUT_DIR / 'turnover_strength_grid_report.html'
    html_table = df_out.to_html(index=False, classes='table table-striped')
    html_full = f"""<html>
    <head>
      <meta charset='utf-8'>
      <title>Turnover/Strength Grid</title>
      <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/mini.css/3.0.1/mini-default.min.css'>
    </head>
    <body>
      <h1>Turnover Penalty & Breakout Strength Grid</h1>
      {html_table}
      <p>CSV: <a href='{csv_path.name}'>{csv_path.name}</a></p>
    </body>
    </html>"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_full)

    print(f"Saved grid results: {csv_path}, {json_path}")
    print(f"Saved HTML report: {html_path}")


if __name__ == '__main__':
    main()
