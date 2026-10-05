#!/usr/bin/env python3
"""Grid search over breakout feature parameters using multi-seed training.

This script is similar to ``train/multi_seed_oos.py`` but wraps an outer loop
that iterates over several combinations of ``RL_BREAKOUT_LOOKBACKS`` and
``RL_BREAKOUT_ENTRY_STRENGTH_QUANTILE``.  Each configuration is trained with
all seeds defined in ``RL_MULTI_SEEDS`` (default 11,22,33,44,55) for stage1
and stage2; the out-of-sample test slice is evaluated after stage2.  Results
are collected into ``backtest/plots_oos/feature_grid_multi_seed``.

Because training is heavy, this script does **not** parallelize seeds; it
runs them sequentially.  Time and compute cost are not a concern per user
request.

Usage is simply:

    python tools/multi_seed_feature_grid.py

Several environment variables can override the basic training parameters (see
``train/multi_seed_oos.py`` for details).  You may also adjust the ``grid``
list below.
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
SEED_RUN_DIR = MODELS_DIR / 'multi_seed_runs'
OUT_BASE = ROOT / 'backtest' / 'plots_oos' / 'feature_grid_multi_seed'


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
    prior_bonus: float,
    breakout_lookbacks: str,
    breakout_quantile: float,
    tag: str,
    enable_zones: bool = True,
    zone_tol: float = 0.002,
    zone_lookback: int = 5,
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
            'RL_BREAKOUT_LOOKBACKS': breakout_lookbacks,
            'RL_BREAKOUT_ENTRY_STRENGTH_QUANTILE': str(breakout_quantile),
            'RL_BREAKOUT_PRIOR_N': '96',
            'RL_BREAKOUT_PRIOR_BONUS_COEF': '0.00015',
            # zone options
            'RL_ENABLE_ZONES': '1' if enable_zones else '0',
            'RL_ZONE_TOL': str(zone_tol),
            'RL_ZONE_LOOKBACK': str(zone_lookback),
            # keep duration settings constant as before
            'RL_TRADE_DURATION_BONUS_COEF': os.getenv('RL_TRADE_DURATION_BONUS_COEF','0.001'),
            'RL_PREFERRED_MAX_TRADE_DURATION_HOURS': os.getenv('RL_PREFERRED_MAX_TRADE_DURATION_HOURS','72'),
            'RL_MIN_TRADE_DURATION_HOURS': os.getenv('RL_MIN_TRADE_DURATION_HOURS','0'),
        }
    )
    if resume_model:
        env['RL_RESUME_MODEL'] = resume_model
    elif 'RL_RESUME_MODEL' in env:
        del env['RL_RESUME_MODEL']

    cmd = [sys.executable, 'train/train_agent.py']
    print(f"\n[GRID] seed={seed} tag={tag} timesteps={timesteps} lookbacks={breakout_lookbacks} q={breakout_quantile}")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    seed_dir = SEED_RUN_DIR / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_src = MODELS_DIR / 'ppo_trading.zip'
    cfg_src = MODELS_DIR / 'env_config.json'
    model_dst = seed_dir / f'{tag}.zip'
    cfg_dst = seed_dir / f'{tag}_env_config.json'
    shutil.copy2(model_src, model_dst)
    shutil.copy2(cfg_src, cfg_dst)
    return model_dst, cfg_dst


def _evaluate(model_path: Path, env_cfg_path: Path, df_test: pd.DataFrame, label: str):
    with open(env_cfg_path,'r',encoding='utf-8') as f:
        env_cfg = json.load(f)
    # Ensure zone-related params have safe defaults if not in saved config
    env_cfg.setdefault('enable_zones', False)
    env_cfg.setdefault('zone_tol', 0.002)
    env_cfg.setdefault('zone_lookback', 5)
    
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
    stage1_prior_bonus = float(os.getenv('RL_STAGE1_PRIOR_BONUS','0.00045'))
    stage2_prior_bonus = float(os.getenv('RL_STAGE2_PRIOR_BONUS','0.00015'))

    # grid specification: combos of (lookbacks, quantile, enable_zones, zone_tol, zone_lookback)
    # Test BREAKOUT ONLY first (zones=False), then zones (zones=True) as expansion later
    grid = [
        ('72,96,144', 0.55, False, 0.002, 5),
        ('72,96,144', 0.65, False, 0.002, 5),
        ('72,96,144', 0.75, False, 0.002, 5),
        ('48,96', 0.65, False, 0.002, 5),
        ('96', 0.65, False, 0.002, 5),
    ]

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    SEED_RUN_DIR.mkdir(parents=True, exist_ok=True)

    print('[GRID] loading test data...')
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col='Close')
    df['Close'] = df['Close_KF']
    _, df_test = split_train_test_by_date(df, train_end=train_end, val_end=val_end)
    print(f'[GRID] test rows={len(df_test)}')

    all_results = []
    for lookbacks, quantile, enable_zones, zone_tol, zone_lookback in grid:
        cfg_tag = lookbacks.replace(',','_') + f'_q{int(quantile*100)}'
        if enable_zones:
            cfg_tag += f'_zones_t{zone_tol}_l{zone_lookback}'
        for seed in seeds:
            s1_model, _ = _run_train(
                seed=seed,
                timesteps=stage1_timesteps,
                train_end=train_end,
                val_end=val_end,
                resume_model='',
                prior_bonus=stage1_prior_bonus,
                breakout_lookbacks=lookbacks,
                breakout_quantile=quantile,
                tag=f'stage1_{cfg_tag}',
                enable_zones=enable_zones,
                zone_tol=zone_tol,
                zone_lookback=zone_lookback,
            )
            s2_model, s2_cfg = _run_train(
                seed=seed,
                timesteps=stage2_timesteps,
                train_end=train_end,
                val_end=val_end,
                resume_model=str(s1_model),
                prior_bonus=stage2_prior_bonus,
                breakout_lookbacks=lookbacks,
                breakout_quantile=quantile,
                tag=f'stage2_{cfg_tag}',
                enable_zones=enable_zones,
                zone_tol=zone_tol,
                zone_lookback=zone_lookback,
            )
            actions_df, metrics = _evaluate(s2_model, s2_cfg, df_test, label=f'{seed}_{cfg_tag}_test')
            row = {
                'seed': seed,
                'lookbacks': lookbacks,
                'quantile': quantile,
                'enable_zones': enable_zones,
                'zone_tol': zone_tol,
                'zone_lookback': zone_lookback,
                **metrics,
                'model_path': str(s2_model),
                'env_config_path': str(s2_cfg),
            }
            all_results.append(row)
            print(f"[GRID] completed seed={seed} cfg={cfg_tag} ret={metrics.get('total_return_pct'):+.2f}%")

    df_out = pd.DataFrame(all_results)
    csv_path = OUT_BASE / 'results.csv'
    json_path = OUT_BASE / 'results.json'
    df_out.to_csv(csv_path, index=False)
    with open(json_path,'w',encoding='utf-8') as f:
        json.dump(all_results,f,indent=2)
    print('Grid finished; results saved to', csv_path, json_path)


if __name__ == '__main__':
    main()
