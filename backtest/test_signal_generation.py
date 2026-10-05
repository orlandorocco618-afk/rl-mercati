#!/usr/bin/env python3
"""
Test: PPO training + signal generation with regime sizing
- Loads H1 with D1 regimes
- Creates a TradingEnv with sizing enabled
- Trains PPO for 10k steps (small, just to validate)
- Generates signals and checks there are no NaN
"""

import sys
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.regimes import add_regimes
from env.trading_env import TradingEnv

def load_h1_with_regimes():
    """Load H1 and D1 data, add regimes."""
    df_h1 = pd.read_csv('data/xauusd_h1_clean.csv', parse_dates=['Date'])
    df_d1 = pd.read_csv('data/xauusd_d1_clean.csv', parse_dates=['Date'])
    
    df_d1.rename(columns={'Date': 'datetime'}, inplace=True)
    df_h1.rename(columns={'Date': 'datetime'}, inplace=True)
    
    # Ensure numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df_h1.columns:
            df_h1[col] = pd.to_numeric(df_h1[col], errors='coerce')
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df_d1.columns:
            df_d1[col] = pd.to_numeric(df_d1[col], errors='coerce')
    
    # Add D1 regimes
    df_d1 = add_regimes(df_d1)
    
    # Merge regimes into H1
    df_d1_regimes = df_d1[['datetime', 'REGIME']].copy()
    df_d1_regimes['date_only'] = df_d1_regimes['datetime'].dt.date
    
    df_h1['date_only'] = df_h1['datetime'].dt.date
    df_h1 = df_h1.merge(
        df_d1_regimes[['date_only', 'REGIME']],
        on='date_only',
        how='left'
    )
    df_h1['REGIME'] = df_h1['REGIME'].fillna(2).astype(int)
    df_h1.drop('date_only', axis=1, inplace=True)
    
    return df_h1

def test_signal_generation():
    print("=" * 70)
    print("TEST: PPO SIGNAL GENERATION WITH REGIME SIZING")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    df_h1 = load_h1_with_regimes()
    print(f"    H1: {len(df_h1)} rows")
    print(f"    Date range: {df_h1['datetime'].min()} to {df_h1['datetime'].max()}")
    
    # Create env
    print("\n[2] Creating TradingEnv with regime sizing...")
    env = TradingEnv(
        df=df_h1,
        window_size=100,
        initial_capital=10000.0,
        commission=0.0002,
        allow_hold_when_flat=True,
        base_position_size=0.1,
        sizing_shrink=0.9,
        enable_regime_sizing=True
    )
    print(f"    Env created: window={env.window_size}, initial_capital={env.initial_capital}")
    
    # Create vectorized env for SB3
    print("\n[3] Creating vectorized environment for PPO...")
    vec_env = DummyVecEnv([lambda: env])
    
    # Train PPO for small steps
    print("\n[4] Training PPO for 10,000 steps...")
    try:
        model = PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,
            max_grad_norm=0.5,
            verbose=0
        )
        
        model.learn(total_timesteps=10000, log_interval=1)
        print(f"    ✓ Training completed successfully (10,000 steps)")
    except Exception as e:
        print(f"    ✗ Training failed: {str(e)}")
        return
    
    # Generate signals
    print("\n[5] Generating signals...")
    obs = env.reset()[0]  # Get observation only
    signals = []
    errors = []
    
    for step in range(500):
        try:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            
            position_size = info.get('position_size', 0.0)
            signal = {
                'step': step,
                'action': action,
                'position_size': position_size,
                'reward': reward,
                'done': done or truncated,
                'has_nan': np.isnan(position_size) or np.isnan(reward)
            }
            signals.append(signal)
            
            if signal['has_nan']:
                errors.append(f"Step {step}: NaN detected in position_size or reward")
            
            if done or truncated:
                obs = env.reset()[0]
        except Exception as e:
            errors.append(f"Step {step}: {str(e)}")
            if len(errors) > 5:
                break
    
    df_signals = pd.DataFrame(signals)
    
    if errors:
        print(f"    Errors encountered:")
        for err in errors[:10]:
            print(f"      {err}")
        return
    
    # Statistics
    print(f"\n[6] Signal generation statistics:")
    print(f"    Total signals: {len(df_signals)}")
    print(f"    NaN checks: {df_signals['has_nan'].sum()} NaN errors")
    print(f"    Action distribution:\n{df_signals['action'].value_counts().sort_index()}")
    print(f"    Position size range: [{df_signals['position_size'].min():.6f}, {df_signals['position_size'].max():.6f}]")
    print(f"    Mean reward: {df_signals['reward'].mean():.6f}")
    print(f"    Episodes ended: {df_signals['done'].sum()}")
    
    print(f"\n[7] Sample signals (first 20):")
    print(df_signals.head(20)[['step', 'action', 'position_size', 'reward', 'done']].to_string())
    
    print("\n" + "=" * 70)
    print("✓ TEST COMPLETED - SIGNAL GENERATION SUCCESSFUL")
    print("=" * 70)
    print(f"\nReady for production training run with time-chunked validation.")

if __name__ == '__main__':
    test_signal_generation()
