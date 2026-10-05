#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import json
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

MS_DIR = ROOT / 'models' / 'multi_seed_runs'


def load_data(train_end, val_end):
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col='Close')
    df['Close'] = df['Close_KF']
    _, df_test = split_train_test_by_date(df, train_end=train_end, val_end=val_end)
    return df_test


def find_seed_dirs():
    if not MS_DIR.exists():
        return []
    return sorted([p for p in MS_DIR.iterdir() if p.is_dir()])


def main():
    seed_dirs = find_seed_dirs()
    if not seed_dirs:
        print('No seed dirs found in', MS_DIR)
        return

    # Grid (same as duration grid)
    grid = {
        'min_trade_duration_hours': [0, 6],
        'preferred_max_trade_duration_hours': [24, 72],
        'trade_duration_bonus_coef': [0.0, 0.001],
    }
    keys = list(grid.keys())

    train_end = '2022-12-31 23:00:00'
    val_end = '2023-12-31 23:00:00'
    df_test = load_data(train_end, val_end)

    rows = []
    for sd in seed_dirs:
        # prefer stage2.zip then stage1.zip
        model_path = sd / 'stage2.zip'
        if not model_path.exists():
            model_path = sd / 'stage1.zip'
        if not model_path.exists():
            print('  skipping seed (no model):', sd)
            continue
        print('Running seed:', sd.name, 'model:', model_path.name)
        model = PPO.load(str(model_path))

        # base env cfg: try seed-specific config then global
        env_cfg_path = sd / 'stage2_env_config.json'
        if not env_cfg_path.exists():
            env_cfg_path = sd / 'stage1_env_config.json'
        if env_cfg_path.exists():
            with open(env_cfg_path, 'r', encoding='utf-8') as f:
                base_cfg = json.load(f)
        else:
            global_cfg = ROOT / 'models' / 'env_config.json'
            base_cfg = json.load(open(global_cfg)) if global_cfg.exists() else {}

        for vals in itertools.product(*[grid[k] for k in keys]):
            cfg = dict(base_cfg)
            for k, v in zip(keys, vals):
                cfg[k] = v
            label = f"{sd.name}_" + '_'.join([f"{k.split('_')[-1]}{str(v).replace('.', '')}" for k, v in zip(keys, vals)])
            print(' ', label, '->', dict(zip(keys, vals)))
            actions_df, metrics = _run_slice(model=model, df_slice=df_test, env_cfg=cfg, allow_hold_when_flat=True, label=label)
            metrics_out = dict(seed=sd.name, cfg_overrides=dict(zip(keys, vals)), **metrics)
            rows.append(metrics_out)

    # save
    csv_path = OUT_DIR / 'duration_grid_multi_seed_results.csv'
    json_path = OUT_DIR / 'duration_grid_multi_seed_results.json'
    df_out = pd.DataFrame(rows)
    df_out.to_csv(csv_path, index=False)
    json.dump(rows, open(json_path, 'w', encoding='utf-8'), indent=2)

    # HTML
    html_path = OUT_DIR / 'duration_grid_multi_seed_report.html'
    html_table = df_out.to_html(index=False, classes='table table-striped')
    html_full = f"""<html><head><meta charset='utf-8'><title>Duration Grid Multi-seed</title>
    <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/mini.css/3.0.1/mini-default.min.css'>
    </head><body><h1>Duration Grid Multi-seed</h1>{html_table}
    <p>CSV: <a href='{csv_path.name}'>{csv_path.name}</a></p></body></html>"""
    open(html_path, 'w', encoding='utf-8').write(html_full)
    print('Saved:', csv_path, json_path, html_path)


if __name__ == '__main__':
    main()
