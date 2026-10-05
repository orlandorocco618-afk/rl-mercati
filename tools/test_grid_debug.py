#!/usr/bin/env python3
"""
Test grid to check that the environment variables are set correctly.
Only 2 configurations and 2 seeds, for speed.
"""
import os
import sys
import json
import subprocess
from pathlib import Path
import shutil

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_BASE = ROOT / "models" / "test_grid_debug"
SEED_RUN_DIR = ROOT / "models" / "multi_seed_runs" / "test_debug"
MODELS_DIR = ROOT / "models"


def _run_train(
    *,
    seed: int,
    timesteps: int,
    train_end: str,
    val_end: str,
    breakout_lookbacks: str,
    breakout_quantile: float,
    tag: str,
):
    env = os.environ.copy()
    env.update(
        {
            'RL_SEED': str(seed),
            'RL_TRAIN_TIMESTEPS': str(timesteps),
            'RL_CHECKPOINT_FREQ': '50000',
            'RL_BREAKOUT_LOOKBACKS': breakout_lookbacks,
            'RL_BREAKOUT_ENTRY_STRENGTH_QUANTILE': str(breakout_quantile),
            'RL_TRAIN_END': train_end,
            'RL_VAL_END': val_end,
            'RL_SKIP_POST_ANALYSIS': '1',
        }
    )

    cmd = [sys.executable, 'train/train_agent.py']
    print(f"\n[TEST-GRID] Running: seed={seed} lookbacks={breakout_lookbacks} q={breakout_quantile}")
    print(f"[TEST-GRID] SUBPROCESS ENV VARS SET:")
    print(f"  RL_BREAKOUT_LOOKBACKS={env['RL_BREAKOUT_LOOKBACKS']}")
    print(f"  RL_BREAKOUT_ENTRY_STRENGTH_QUANTILE={env['RL_BREAKOUT_ENTRY_STRENGTH_QUANTILE']}")
    print(f"  RL_SEED={env['RL_SEED']}")
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


def main():
    train_end = "2022-12-31 23:00:00"
    val_end = "2023-12-31 23:00:00"
    seeds = [11, 22]  # Only 2 seeds for quick test
    stage1_timesteps = 5000  # SHORT for test
    
    # Grid with just 2 configurations
    grid = [
        ('72,96,144', 0.55),   # Config A
        ('96', 0.75),          # Config B - clearly different
    ]

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    SEED_RUN_DIR.mkdir(parents=True, exist_ok=True)

    for lookbacks, quantile in grid:
        cfg_tag = lookbacks.replace(',','_') + f'_q{int(quantile*100)}'
        for seed in seeds:
            _run_train(
                seed=seed,
                timesteps=stage1_timesteps,
                train_end=train_end,
                val_end=val_end,
                breakout_lookbacks=lookbacks,
                breakout_quantile=quantile,
                tag=f'test_{cfg_tag}',
            )
    
    print("\n[TEST-GRID] Completed. Check above for env var output from subprocesses.")


if __name__ == '__main__':
    main()
