"""
Time-chunked validation V2 - WITH HYBRID REGIME FILTERING

Integrates the HYBRID regime detection system to automatically
exclude unstable periods during training
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
from utils.regime_hybrid import HybridRegimeClassifier
from env.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from backtest.generate_actions import generate_actions
from backtest.run_analysis import run_analysis_and_organize


CHUNK_TIMES = [
    ("2004-01-01", "2008-01-01"),
    ("2012-01-01", "2024-12-31"),  # Skip 2008-2012 crisis period entirely
]

TRAIN_STEPS = 10000
HYBRID_FILTER_LEVEL = "SEVERE"  # Options: "NONE", "SEVERE", "MILD_SEVERE"

OUT_DIR = Path('backtest/plots_time_chunks_v2')
MODELS_DIR = Path('models/time_chunked_v2')
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
    """Extract dataframe slice by date range"""
    if 'Time' in df.columns:
        df_loc = df.copy()
        df_loc['Time'] = pd.to_datetime(df_loc['Time'])
        mask = (df_loc['Time'] >= pd.to_datetime(start)) & (df_loc['Time'] < pd.to_datetime(end))
        return df_loc.loc[mask].reset_index(drop=True)
    return pd.DataFrame()


def filter_by_hybrid_regime(df, filter_level="SEVERE"):
    """
    Filter dataset using HYBRID regime classification
    
    Args:
        df: DataFrame with HYBRID_CLASS column
        filter_level: "NONE", "SEVERE" (exclude class 2), "MILD_SEVERE" (exclude 1,2)
    
    Returns:
        Filtered dataframe
    """
    if filter_level == "NONE":
        return df.copy()
    
    if 'HYBRID_CLASS' not in df.columns:
        print("    WARNING: HYBRID_CLASS not in dataframe, skipping filter")
        return df.copy()
    
    original_len = len(df)
    
    if filter_level == "SEVERE":
        df_filtered = df[df['HYBRID_CLASS'] != 2].copy().reset_index(drop=True)
        excluded = (df['HYBRID_CLASS'] == 2).sum()
    elif filter_level == "MILD_SEVERE":
        df_filtered = df[df['HYBRID_CLASS'] == 0].copy().reset_index(drop=True)
        excluded = (df['HYBRID_CLASS'] != 0).sum()
    else:
        df_filtered = df.copy()
        excluded = 0
    
    removed_pct = 100 * excluded / original_len if original_len > 0 else 0
    print(f"    Hybrid filter ({filter_level}): removed {excluded}/{original_len} candles ({removed_pct:.1f}%)")
    
    return df_filtered


def run_once(train_slice, test_slice, run_name, filter_level="SEVERE"):
    """
    Train on train period only, then evaluate out-of-sample on test period.

    Args:
        train_slice: DataFrame to train on
        test_slice: DataFrame to test on
        run_name: Name for this run
        filter_level: Hybrid regime filter level
    """
    print(f"\n=== RUN: {run_name} ===")
    print(f"Filter level: {filter_level}")

    # Phase 1: train ONLY on train chunk (filtered).
    print("Phase 1: Filtering and training on train chunk...")
    train_slice_clean = filter_by_hybrid_regime(train_slice, filter_level)
    if train_slice_clean.empty:
        raise ValueError("Filtered train slice is empty")
    print(f"  Training model ({len(train_slice_clean)} rows, 5000 steps)...")

    env_train = DummyVecEnv([make_env(train_slice_clean, allow_hold=True)])
    model_train = PPO(
        'MlpPolicy',
        env_train,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.1,
        verbose=0,
        max_grad_norm=0.5,
        gae_lambda=0.95,
    )
    model_train.learn(total_timesteps=5000)

    model_path = MODELS_DIR / f"{run_name}.zip"
    model_train.save(str(model_path))
    print(f"Saved model {model_path}")

    # Phase 2: strict out-of-sample inference on full test chunk (unfiltered coverage).
    print("Phase 2: Out-of-sample inference on test chunk...")
    test_env = DummyVecEnv([make_env(test_slice, allow_hold=True)])
    obs = test_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    action_rows = []
    try:
        done = False
        while not done:
            act, _ = model_train.predict(obs, deterministic=True)
            obs, rew, done_raw, info = test_env.step(act)

            if isinstance(done_raw, (np.ndarray, list, tuple)):
                done = bool(done_raw[0])
            else:
                done = bool(done_raw)

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

            regime_raw = current_row.get("REGIME", 2)
            try:
                regime_val = 2 if pd.isna(regime_raw) else int(regime_raw)
            except Exception:
                regime_val = 2

            reward_val = float(rew[0]) if isinstance(rew, (np.ndarray, list, tuple)) else float(rew)
            model_action_val = int(act[0]) if isinstance(act, (np.ndarray, list, tuple)) else int(act)
            action_val = int(info_dict.get("executed_action", model_action_val))

            action_rows.append({
                "Time": current_row.get("Time"),
                "Open": current_row.get("Open"),
                "High": current_row.get("High"),
                "Low": current_row.get("Low"),
                "Close": current_row.get("Close"),
                "Regime": regime_val,
                "Action": action_val,
                "ModelAction": model_action_val,
                "Reward": reward_val,
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
    finally:
        try:
            test_env.close()
        except Exception:
            pass

    # Save actions
    out_csv = ACTIONS_DIR / f"xauusd_actions_timechunk_v2_{run_name}.csv"
    df_out = pd.DataFrame(action_rows)
    if df_out.empty:
        raise RuntimeError("No actions generated during inference")

    if 'Close' in df_out.columns:
        df_out['Close'] = pd.to_numeric(df_out['Close'], errors='coerce').ffill().bfill()

    df_out.to_csv(out_csv, index=False)
    print(f"Saved actions to {out_csv}")

    # Quick internal simulation
    try:
        tmp = df_out.copy()
        capital = 10000.0
        position = 0
        entry_price = None
        trades = []

        for _, row in tmp.iterrows():
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

        if position == 1 and entry_price is not None:
            realized = (price - entry_price) / entry_price
            trades.append(realized)
            capital += realized * capital
        elif position == -1 and entry_price is not None:
            realized = (entry_price - price) / entry_price
            trades.append(realized)
            capital += realized * capital

        ret_pct = 100.0 * (capital - 10000.0) / 10000.0
        profitable = sum(1 for t in trades if t > 0)
        print(f"  Quick sim: {len(trades)} trades, {profitable} profitable, Return: {ret_pct:+.2f}%")
    except Exception as e:
        print(f"  Quick sim failed: {e}")

    # Full analysis
    save_dir = OUT_DIR / run_name
    os.makedirs(save_dir, exist_ok=True)
    try:
        run_analysis_and_organize(str(out_csv), initial_capital=10000.0, save_dir=str(save_dir))
        print(f"Analysis saved to {save_dir}")
    except Exception as e:
        print(f"Full analysis failed: {e}")

    del model_train, env_train

if __name__ == '__main__':
    print("\n" + "="*80)
    print("TIME-CHUNKED VALIDATION V2 - WITH HYBRID REGIME FILTERING")
    print("="*80)
    
    print("\n[1] Loading full dataset...")
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col='Close')
    df['Close'] = df['Close_KF']
    df['Time'] = pd.to_datetime(df['Time'])
    print(f"Total rows: {len(df)}")

    print("\n[2] Computing HYBRID regime classification...")
    classifier = HybridRegimeClassifier(df)
    df = classifier.add_hybrid_class()
    
    stats = classifier.get_stats()
    print(f"  STABLE: {stats['stable_pct']:.1f}%")
    print(f"  MILD: {stats['mild_pct']:.1f}%")
    print(f"  SEVERE: {stats['severe_pct']:.1f}%")

    print("\n[3] Creating time slices...")
    slices = []
    for start, end in CHUNK_TIMES:
        s = slice_df_by_dates(df, start, end)
        print(f"Slice {start} -> {end}: {len(s)} rows")
        slices.append((start, end, s))

    print(f"\n[4] Running pairwise training/testing (filter_level={HYBRID_FILTER_LEVEL})...")
    for i in range(len(slices)-1):
        tstart, tend, train_df = slices[i]
        _, _, test_df = slices[i+1]
        run_name = f"train_{tstart}_test_{slices[i+1][0]}"
        
        if train_df.empty or test_df.empty:
            print(f"Skipping run {run_name}: empty slice")
            continue
        
        try:
            run_once(train_df, test_df, run_name, filter_level=HYBRID_FILTER_LEVEL)
        except Exception as e:
            print(f"Run {run_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*80)
    print("All runs complete.")
    print("="*80)
