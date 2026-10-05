"""
Time-chunked validation
- Periods: [2004-01-01 to 2008-01-01), [2008-01-01 to 2012-01-01), [2012-01-01 to 2024-12-31)
- For each step: train quickly on 'train' chunk (10k timesteps, using PPO baseline), then evaluate on test chunk
- Save analysis per run under backtest/plots_time_chunks/<train_start>_<test_start>
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

sys.path.insert(0, '.')

from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter
from env.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from backtest.generate_actions import generate_actions
from backtest.run_analysis import run_analysis_and_organize


CHUNK_TIMES = [
    ("2004-01-01", "2006-01-01"),   # First half for training
    ("2006-01-01", "2008-01-01"),   # Second half for testing
]

TRAIN_STEPS = 10000

OUT_DIR = Path('backtest/plots_time_chunks')
MODELS_DIR = Path('models/time_chunked')
ACTIONS_DIR = Path('backtest/data')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def make_env(df, allow_hold=True):
    def _init():
        rebalance = not allow_hold
        return TradingEnv(
            df=df,
            window_size=100,
            initial_capital=10000.0,
            max_drawdown=0.30,
            allow_hold_when_flat=allow_hold,
            regime_hard_stop=False,
            enforce_directional_rebalance=rebalance,
            rebalance_after_entries=30,
            rebalance_dominance_threshold=0.70,
            directional_balance_reward_coef=0.001 if rebalance else 0.0003
        )
    return _init


def slice_df_by_dates(df, start, end):
    # assume df has a datetime-like index or a column named 'Date' or 'datetime'
    if 'Time' in df.columns:
        df_loc = df.copy()
        df_loc['Time'] = pd.to_datetime(df_loc['Time'])
        mask = (df_loc['Time'] >= pd.to_datetime(start)) & (df_loc['Time'] < pd.to_datetime(end))
        return df_loc.loc[mask].reset_index(drop=True)
    if 'Date' in df.columns:
        df_loc = df.copy()
        df_loc['Date'] = pd.to_datetime(df_loc['Date'])
        mask = (df_loc['Date'] >= pd.to_datetime(start)) & (df_loc['Date'] < pd.to_datetime(end))
        return df_loc.loc[mask].reset_index(drop=True)
    if 'datetime' in df.columns:
        df_loc = df.copy()
        df_loc['datetime'] = pd.to_datetime(df_loc['datetime'])
        mask = (df_loc['datetime'] >= pd.to_datetime(start)) & (df_loc['datetime'] < pd.to_datetime(end))
        return df_loc.loc[mask].reset_index(drop=True)
    # fallback: try index
    df_idx = df.copy()
    try:
        df_idx.index = pd.to_datetime(df_idx.index)
        mask = (df_idx.index >= pd.to_datetime(start)) & (df_idx.index < pd.to_datetime(end))
        return df_idx.loc[mask].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def filter_shock_periods(df):
    """Remove REGIME=3 (Shock/Extreme) rows to avoid numerical instability during training"""
    if 'REGIME' not in df.columns:
        return df
    
    original_len = len(df)
    df_filtered = df[df['REGIME'] != 3].copy().reset_index(drop=True)
    removed = original_len - len(df_filtered)
    
    if removed > 0:
        pct = 100 * removed / original_len
        print(f"    Removed {removed} shock rows ({pct:.1f}%) from training")
    
    # Additional validation: check for NaNs in key columns
    key_cols = ['Close', 'High', 'Low', 'Open']
    for col in key_cols:
        if col in df_filtered.columns:
            nan_count = df_filtered[col].isna().sum()
            if nan_count > 0:
                print(f"    WARNING: {nan_count} NaN values in {col}, forward-filling...")
                df_filtered[col] = df_filtered[col].ffill().bfill()  # Use pandas methods
    
    return df_filtered


def run_once(train_slice, test_slice, run_name):
    print(f"\n=== RUN: {run_name} ===")
    # Strategy: train only on train_slice, evaluate strictly on test_slice.

    print(f"Phase 1: Training model on train chunk ({len(train_slice)} rows, 5000 steps)...")
    env_train = DummyVecEnv([make_env(train_slice, allow_hold=True)])
    model_train = PPO(
        'MlpPolicy',
        env_train,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.1,
        verbose=0,
        max_grad_norm=0.5,  # Gradient clipping to prevent NaN
        gae_lambda=0.95,
    )
    model_train.learn(total_timesteps=5000)

    model_path = MODELS_DIR / f"{run_name}.zip"
    model_train.save(str(model_path))
    print(f"Saved model {model_path}")

    # Generate actions on test set
    test_env = DummyVecEnv([make_env(test_slice, allow_hold=True)])
    obs = test_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    action_rows = []
    try:
        done = False
        while not done:
            act, _ = model_train.predict(obs, deterministic=True)
            obs, rew, done, info = test_env.step(act)
            if isinstance(done, (np.ndarray, list, tuple)):
                done = bool(done[0])
            else:
                done = bool(done)

            if isinstance(info, (list, tuple)) and len(info) > 0:
                info_dict = info[0] if isinstance(info[0], dict) else {}
            elif isinstance(info, dict):
                info_dict = info
            else:
                info_dict = {}

            internal_env = test_env.envs[0]
            executed_step = int(info_dict.get("executed_step", max(0, getattr(internal_env, "current_step", 0) - 1)))
            current_idx = min(max(executed_step, 0), len(test_slice) - 1)
            current_row = test_slice.iloc[current_idx]
            action_rows.append({
                "Time": current_row.get("Time"),
                "Open": current_row.get("Open"),
                "High": current_row.get("High"),
                "Low": current_row.get("Low"),
                "Close": current_row.get("Close"),
                "Regime": current_row.get("REGIME", 2),
                "Action": int(info_dict.get("executed_action", int(act[0]))),
                "ModelAction": int(act[0]),
                "Reward": float(rew[0]) if isinstance(rew, (np.ndarray, list, tuple)) else float(rew),
                "Position": int(info_dict.get("position", getattr(internal_env, "position", 0))),
                "Equity": float(info_dict.get("equity", getattr(internal_env, "equity", np.nan))),
                "DangerLevel": int(info_dict.get("danger_level", 0)),
                "PositionSize": float(info_dict.get("position_size", np.nan)),
                "ForcedAction": bool(info_dict.get("forced_action", False)),
                "HardStop": bool(info_dict.get("hard_stop_applied", False)),
            })
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup model/env
    del model_train, env_train
    # save actions csv
    out_csv = ACTIONS_DIR / f"xauusd_actions_timechunk_{run_name}.csv"
    df_out = pd.DataFrame(action_rows)
    # Ensure Close numeric: prefer Close_KF if present, then forward-fill
    if 'Close' in df_out.columns:
        df_out['Close'] = pd.to_numeric(df_out['Close'], errors='coerce')
    # forward-fill remaining NaNs to avoid all-NaN close (quick fix for analysis)
    if 'Close' in df_out.columns:
        df_out['Close'] = df_out['Close'].ffill().bfill()

    df_out.to_csv(out_csv, index=False)
    print(f"Saved actions to {out_csv}")

    # Quick internal simulation (lightweight) to avoid heavy plotting and to run train+test back-to-back
    try:
        tmp = df_out.copy()
        total_steps = len(tmp)
        action_counts = tmp['Action'].value_counts().to_dict()
        print(f"  Action distribution (quick): {action_counts}")

        # Simulate trades (same logic as analyze_actions_v2 but minimal)
        position = 0
        entry_price = None
        capital = 10000.0
        trades = []
        for idx, row in tmp.iterrows():
            action = int(row.get('Action', 0))
            price = float(row.get('Close', np.nan))
            if np.isnan(price):
                continue
            realized = 0.0
            if action == 1:
                if position <= 0:
                    if position == -1 and entry_price is not None:
                        realized = (entry_price - price) / entry_price
                        trades.append(realized)
                    position = 1
                    entry_price = price
            elif action == 2:
                if position >= 0:
                    if position == 1 and entry_price is not None:
                        realized = (price - entry_price) / entry_price
                        trades.append(realized)
                    position = -1
                    entry_price = price
            elif action == 3:
                if position == 1 and entry_price is not None:
                    realized = (price - entry_price) / entry_price
                    trades.append(realized)
                elif position == -1 and entry_price is not None:
                    realized = (entry_price - price) / entry_price
                    trades.append(realized)
                position = 0
                entry_price = None
            capital += realized * capital

        total_trades = len(trades)
        profitable = sum(1 for t in trades if t > 0)
        ret_pct = 100.0 * (capital - 10000.0) / 10000.0
        print(f"  Quick sim -> Trades: {total_trades}, Profitable: {profitable}, Return: {ret_pct:+.2f}%")
    except Exception as e:
        print(f"  Quick sim failed: {e}")

    # Full analysis (optional heavy) - keep for later, saved under OUT_DIR
    save_dir = OUT_DIR / run_name
    os.makedirs(save_dir, exist_ok=True)
    try:
        run_analysis_and_organize(str(out_csv), initial_capital=10000.0, save_dir=str(save_dir))
        print(f"Analysis saved to {save_dir}")
    except Exception as e:
        print(f"Full analysis failed: {e}")


if __name__ == '__main__':
    print("Loading full dataset...")
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col='Close')
    df['Close'] = df['Close_KF']
    print(f"Total rows: {len(df)}")

    # create slices
    slices = []
    for start, end in CHUNK_TIMES:
        s = slice_df_by_dates(df, start, end)
        print(f"Slice {start} -> {end}: {len(s)} rows")
        slices.append((start, end, s))

    # Run pairwise: train on chunk i, test on chunk i+1
    for i in range(len(slices)-1):
        tstart, tend, train_df = slices[i]
        _, _, test_df = slices[i+1]
        run_name = f"train_{tstart}_test_{slices[i+1][0]}"
        if train_df.empty or test_df.empty:
            print(f"Skipping run {run_name}: empty slice")
            continue
        run_once(train_df, test_df, run_name)
    print("All runs complete.")
