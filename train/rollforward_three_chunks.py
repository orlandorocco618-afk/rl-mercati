#!/usr/bin/env python3
"""
Roll-forward three-chunk test with regime smoothing
- Chunks: 1=(2004-06-11,2008-01-01), 2=(2008-01-01,2012-01-01), 3=(2012-01-01,2025-12-31)
- Train on chunk i, test on chunk (i+1 mod 3)
- Compute HYBRID_CLASS on D1, map to H1 and apply persistence smoothing
- Overwrite HYBRID_CLASS in H1 with smoothed values to avoid overreaction
- Train PPO briefly and evaluate to ensure numerical stability
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.regimes import add_regimes
from utils.regime_hybrid import HybridRegimeClassifier
from env.trading_env import TradingEnv


def load_and_prepare(smoothing_window_hours=72, smoothing_threshold=3):
    # Load
    df_h1 = pd.read_csv('data/xauusd_h1_clean.csv', parse_dates=['Date'])
    df_d1 = pd.read_csv('data/xauusd_d1_clean.csv', parse_dates=['Date'])
    df_d1.rename(columns={'Date': 'datetime'}, inplace=True)
    df_h1.rename(columns={'Date': 'datetime'}, inplace=True)

    # Ensure numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df_h1.columns:
            df_h1[col] = pd.to_numeric(df_h1[col], errors='coerce')
        if col in df_d1.columns:
            df_d1[col] = pd.to_numeric(df_d1[col], errors='coerce')

    # Compute D1 regimes (adds ATR etc.)
    df_d1 = add_regimes(df_d1)

    # Compute HYBRID_CLASS on D1
    classifier = HybridRegimeClassifier(df_d1)
    df_d1 = classifier.add_hybrid_class()

    # Map HYBRID_CLASS from D1 -> H1 by date
    df_d1_map = df_d1[['datetime', 'HYBRID_CLASS']].copy()
    df_d1_map['date_only'] = df_d1_map['datetime'].dt.date
    df_h1['date_only'] = df_h1['datetime'].dt.date
    df_h1 = df_h1.merge(df_d1_map[['date_only', 'HYBRID_CLASS']], on='date_only', how='left')

    # Fill and ensure integer
    df_h1['HYBRID_CLASS'] = df_h1['HYBRID_CLASS'].fillna(0).astype(int)

    # Smoothing: persistence of severe=2 over last smoothing_window_hours
    severe_flag = (df_h1['HYBRID_CLASS'] == 2).astype(int)
    # rolling sum over window (smoothing_window_hours) - requires contiguous index
    severe_roll = severe_flag.rolling(window=smoothing_window_hours, min_periods=1).sum()

    # Smoothed hybrid: if severe_roll >= threshold -> severe (2)
    # else if original ==1 -> mild, else 0
    smoothed = np.where(severe_roll >= smoothing_threshold, 2, df_h1['HYBRID_CLASS'])
    smoothed = np.where(smoothed == 2, 2, np.where(df_h1['HYBRID_CLASS'] == 1, 1, 0))

    df_h1['HYBRID_CLASS'] = smoothed.astype(int)

    # Drop helper
    df_h1.drop('date_only', axis=1, inplace=True)

    return df_h1


def slice_by_dates(df, start_str, end_str):
    return df[(df['datetime'] >= pd.Timestamp(start_str)) & (df['datetime'] < pd.Timestamp(end_str))].reset_index(drop=True)


def train_and_eval(df_train, df_test, steps=5000):
    env_train = TradingEnv(df=df_train, window_size=100, initial_capital=10000.0,
                           commission=0.0002, allow_hold_when_flat=True,
                           base_position_size=0.1, sizing_shrink=0.9,
                           enable_regime_sizing=True, regime_hard_stop=False,
                           enforce_directional_rebalance=True,
                           rebalance_after_entries=30,
                           rebalance_dominance_threshold=0.70,
                           directional_balance_reward_coef=0.001)
    vec = DummyVecEnv([lambda: env_train])
    model = PPO('MlpPolicy', vec, ent_coef=0.1, verbose=0)
    model.learn(total_timesteps=steps)

    env_test = TradingEnv(df=df_test, window_size=100, initial_capital=10000.0,
                          commission=0.0002, allow_hold_when_flat=True,
                          base_position_size=0.1, sizing_shrink=0.9,
                          enable_regime_sizing=True, regime_hard_stop=False,
                          enforce_directional_rebalance=False,
                          directional_balance_reward_coef=0.0003)
    obs = env_test.reset()[0]
    errors = []
    for _ in range(min(1000, len(df_test)-100)):
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_test.step(action)
            if done or truncated:
                obs = env_test.reset()[0]
        except Exception as e:
            errors.append(str(e))
            break
    final_equity = env_test.equity
    pnl = (final_equity - 10000.0) / 10000.0 * 100
    return pnl, final_equity, errors


if __name__ == '__main__':
    print('\n=== Roll-Forward Three-Chunks Test ===\n')
    df = load_and_prepare(smoothing_window_hours=72, smoothing_threshold=3)
    print('Data loaded and HYBRID_CLASS smoothed.')

    # Define chunks
    chunks = [
        ('2004-06-11', '2008-01-01'),  # 1
        ('2008-01-01', '2012-01-01'),  # 2
        ('2012-01-01', '2025-12-31'),  # 3
    ]

    # Circular roll-forward: train on i -> test on i+1
    results = []
    for i in range(len(chunks)):
        train_idx = i
        test_idx = (i + 1) % len(chunks)
        t0, t1 = chunks[train_idx]
        s0, s1 = chunks[test_idx]
        df_train = slice_by_dates(df, t0, t1)
        df_test = slice_by_dates(df, s0, s1)
        print(f'\n-- Run: train {train_idx+1} ({t0} to {t1}) -> test {test_idx+1} ({s0} to {s1})')
        print(f'   Train rows: {len(df_train)}, Test rows: {len(df_test)}')
        pnl, final_equity, errors = train_and_eval(df_train, df_test, steps=5000)
        print(f'   Result PnL: {pnl:.2f}% | Final equity: {final_equity:.2f} | Errors: {errors}')
        results.append({'train': train_idx+1, 'test': test_idx+1, 'pnl': pnl, 'final_equity': final_equity, 'errors': errors})

    print('\n=== Summary ===')
    for r in results:
        print(f"Train {r['train']} -> Test {r['test']}: PnL {r['pnl']:.2f}% | FinalEq {r['final_equity']:.2f} | Errors {len(r['errors'])}")

    print('\nDone.')
