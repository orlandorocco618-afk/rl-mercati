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
ENV_CFG_PATH = Path('models/env_config.json')


def load_data(train_end, val_end):
    # for fast inference we don't need expensive regime calculation; just load
    # the H1 file and set a flat 'REGIME' column (the env will handle missing
    # values gracefully).
    from utils.data_utils import load_h1_simple

    df = load_h1_simple()
    # ensure some of the derived columns exist for env
    df = apply_kalman_filter(df, price_col='Close')
    df['Close'] = df['Close_KF']
    # enforce REGIME column so env doesn't recompute anything
    if 'REGIME' not in df.columns:
        df['REGIME'] = 2
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

    # grid over "safe" knobs that don't alter observation space.
    # We are using the stable model which was trained without any breakout
    # features (obs dim 15), hence adding lookbacks would change the
    # observation dimension and cause mismatches.  For the fast scan we
    # therefore clear any breakouts and only vary parameters that
    # influence the environment at inference time but keep the input size
    # constant.  This gives a very limited view – for anything involving
    # breakouts the model must be retrained (thorough stage).
    grid = {
        'atr_stop_multiple': [1.5, 2.0, 2.5, 3.0],
        'reward_shaping_weight': [0.0, 0.06, 0.12, 0.24],
    }
    keys = list(grid.keys())
    rows = []

    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(base_cfg)
        # disable all breakouts so that obs space stays at 15 dimensions
        cfg['breakout_lookbacks'] = []
        # apply the current parameter values
        a, r = vals
        cfg['atr_stop_multiple'] = a
        cfg['reward_shaping_weight'] = r
        label = f"atr{a}_rw{r}".replace(' ', '')
        print(f"Running {label} -> atr_stop={a}, reward_shaping={r}")
        actions_df, metrics = _run_slice(model=model, df_slice=df_test, env_cfg=cfg, allow_hold_when_flat=True, label=label)
        metrics_out = dict(atr_stop_multiple=a, reward_shaping_weight=r, **metrics)
        rows.append(metrics_out)

    csv_path = OUT_DIR / 'feature_grid_results.csv'
    json_path = OUT_DIR / 'feature_grid_results.json'
    df_out = pd.DataFrame(rows)
    df_out.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    html_path = OUT_DIR / 'feature_grid_report.html'
    html_table = df_out.to_html(index=False, classes='table table-striped')
    html_full = f"""<html>
    <head>
      <meta charset='utf-8'>
      <title>Feature Grid Report</title>
      <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/mini.css/3.0.1/mini-default.min.css'>
    </head>
    <body>
      <h1>Feature Generation Grid</h1>
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
