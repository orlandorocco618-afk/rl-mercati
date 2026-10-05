#!/usr/bin/env python3
"""Multi-seed feature grid for Support/Resistance (zone-based) agent.

Analogous to tools/multi_seed_feature_grid.py but focused on tuning zone parameters
(tol, lookback) for the S/R agent. Breakout is disabled.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, '.')

from backtest.out_of_sample_test import _run_slice, split_train_test_by_date
from stable_baselines3 import PPO
from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter

ROOT = Path('.')
MODELS_DIR = ROOT / 'models'
SEED_RUN_DIR = MODELS_DIR / 'multi_seed_runs_sr'
OUT_BASE = ROOT / 'backtest' / 'plots_oos' / 'feature_grid_multi_seed_sr'


def _parse_int_list(raw: str, default: list[int]) -> list[int]:
    text = (raw or '').strip()
    if not text:
        return list(default)
    out: list[int] = []
    for token in text.replace(';', ',').split(','):
        t = token.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except Exception:
            continue
    return out or list(default)


def _run_train(
    *,
    seed: int,
    timesteps: int,
    train_end: str,
    val_end: str,
    resume_model: str,
    tag: str,
    zone_tol: float,
    zone_lookback: int,
):
    env = os.environ.copy()
    env.update(
        {
            'RL_SEED': str(seed),
            'RL_TRAIN_TIMESTEPS': str(timesteps),
            'RL_CHECKPOINT_FREQ': '50000',
            'RL_RUN_EDGE_SCAN': '0',
            'RL_SKIP_POST_ANALYSIS': '1',
            'RL_TRAIN_END': train_end,
            'RL_VAL_END': val_end,
            # No breakout for SR agent
            'RL_BREAKOUT_LOOKBACKS': '',
            # Zone tuning parameters
            'RL_ENABLE_ZONES': '1',
            'RL_ZONE_TOL': str(zone_tol),
            'RL_ZONE_LOOKBACK': str(zone_lookback),
        }
    )
    if resume_model:
        env['RL_RESUME_MODEL'] = resume_model
    elif 'RL_RESUME_MODEL' in env:
        del env['RL_RESUME_MODEL']

    cmd = [sys.executable, 'train/train_agent_sr.py']
    print(f"\n[GRID-SR] seed={seed} tag={tag} timesteps={timesteps} tol={zone_tol} lookback={zone_lookback}")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    seed_dir = SEED_RUN_DIR / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_src = MODELS_DIR / 'ppo_sr_zones.zip'
    cfg_src = MODELS_DIR / 'env_config_sr.json'
    model_dst = seed_dir / f'{tag}.zip'
    cfg_dst = seed_dir / f'{tag}_env_config.json'
    shutil.copy2(model_src, model_dst)
    shutil.copy2(cfg_src, cfg_dst)
    return model_dst, cfg_dst


def _evaluate(model_path: Path, env_cfg_path: Path, df_test: pd.DataFrame, label: str):
    with open(env_cfg_path,'r',encoding='utf-8') as f:
        env_cfg = json.load(f)
    model = PPO.load(str(model_path))
    actions_df, metrics = _run_slice(
        model=model,
        df_slice=df_test,
        env_cfg=env_cfg,
        allow_hold_when_flat=True,
        label=label,
    )
    active = int((actions_df['Action']!=0).sum()) if not actions_df.empty else 0
    metrics['active_steps_pct'] = 100.0 * active / max(1,len(actions_df))
    metrics['rows'] = int(len(actions_df))
    return actions_df, metrics


def main():
    train_end = os.getenv('RL_TRAIN_END','2022-12-31 23:00:00').strip()
    val_end = os.getenv('RL_VAL_END','2023-12-31 23:00:00').strip()
    seeds = _parse_int_list(os.getenv('RL_MULTI_SEEDS',''),[11,22,33,44,55])[:5]
    stage1_timesteps = int(os.getenv('RL_STAGE1_TIMESTEPS','50000'))
    stage2_timesteps = int(os.getenv('RL_STAGE2_TIMESTEPS','100000'))

    # grid: combos of (zone_tol, zone_lookback) to explore
    # lower tol = stricter zone clustering, higher lookback = longer windows for local extrema
    grid = [
        (0.001, 3),   # narrow tolerance, short lookback
        (0.002, 5),   # default
        (0.003, 5),   # relaxed tolerance
        (0.002, 7),   # longer lookback
        (0.002, 10),  # very long lookback
    ]

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    SEED_RUN_DIR.mkdir(parents=True, exist_ok=True)

    print('[GRID-SR] loading test data...')
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col='Close')
    df['Close'] = df['Close_KF']
    _, df_test = split_train_test_by_date(df, train_end=train_end, val_end=val_end)
    print(f'[GRID-SR] test rows={len(df_test)}')

    all_results = []
    for zone_tol, zone_lookback in grid:
        cfg_tag = f'tol{zone_tol:.3f}_lb{zone_lookback}'
        for seed in seeds:
            s1_model, _ = _run_train(
                seed=seed,
                timesteps=stage1_timesteps,
                train_end=train_end,
                val_end=val_end,
                resume_model='',
                tag=f'stage1_{cfg_tag}',
                zone_tol=zone_tol,
                zone_lookback=zone_lookback,
            )
            s2_model, s2_cfg = _run_train(
                seed=seed,
                timesteps=stage2_timesteps,
                train_end=train_end,
                val_end=val_end,
                resume_model=str(s1_model),
                tag=f'stage2_{cfg_tag}',
                zone_tol=zone_tol,
                zone_lookback=zone_lookback,
            )
            actions_df, metrics = _evaluate(s2_model, s2_cfg, df_test, label=f'{seed}_{cfg_tag}_test')
            row = {
                'seed': seed,
                'zone_tol': zone_tol,
                'zone_lookback': zone_lookback,
                **metrics,
                'model_path': str(s2_model),
                'env_config_path': str(s2_cfg),
            }
            all_results.append(row)
            print(f"[GRID-SR] completed seed={seed} cfg={cfg_tag} ret={metrics.get('total_return_pct'):+.2f}%")

    df_out = pd.DataFrame(all_results)
    csv_path = OUT_BASE / 'results.csv'
    json_path = OUT_BASE / 'results.json'
    df_out.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    html_path = OUT_BASE / 'report.html'
    html_table = df_out.to_html(index=False, classes='table table-striped')
    html_full = f"""<html>
    <head>
      <meta charset='utf-8'>
      <title>SR Agent Feature Grid</title>
      <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/mini.css/3.0.1/mini-default.min.css'>
    </head>
    <body>
      <h1>Support/Resistance Zone Tuning Grid (SR Agent)</h1>
      <p><strong>Edge:</strong> Support/Resistance zones, no breakout</p>
      <p><strong>Parameters tuned:</strong> zone_tol (cluster tolerance), zone_lookback (extrema window)</p>
      {html_table}
      <p>Files: <a href='{csv_path.name}'>{csv_path.name}</a>, <a href='{json_path.name}'>{json_path.name}</a></p>
    </body>
    </html>"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_full)

    print(f"\n[GRID-SR] Saved results:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")


if __name__ == '__main__':
    main()
