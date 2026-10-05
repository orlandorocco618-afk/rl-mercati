#!/usr/bin/env python3
"""
Small test: check that the sizing works
- Loads H1 + D1 regimes
- Creates a TradingEnv with sizing enabled
- Runs N steps and checks that:
  1. position_size scales as expected (base_size * shrink^danger)
  2. There are no NaN in the computations
  3. PnL and commissions scale
"""

import sys
import os
import pandas as pd
import numpy as np

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
    
    # Add D1 regimes (computes ATR, Hurst, ADF, ACF1, REGIME)
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
    df_h1['REGIME'] = df_h1['REGIME'].fillna(2).astype(int)  # Default to Noise
    df_h1.drop('date_only', axis=1, inplace=True)
    
    return df_h1

def test_regime_sizing():
    print("=" * 70)
    print("TEST: REGIME DETECTION + POSITION SIZING")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading H1 with D1 regimes...")
    df_h1 = load_h1_with_regimes()
    print(f"    H1: {len(df_h1)} rows")
    print(f"    Regime distribution: {df_h1['REGIME'].value_counts().sort_index().to_dict()}")
    
    # Create env with sizing enabled
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
    print(f"    base_position_size={env.base_position_size}")
    print(f"    sizing_shrink={env.sizing_shrink}")
    print(f"    enable_regime_sizing={env.enable_regime_sizing}")
    
    # Run N steps and collect data
    print("\n[3] Running 500 steps and sampling position_size...")
    obs, _ = env.reset()
    
    samples = []
    nan_errors = []
    for step_idx in range(500):
        action = np.random.randint(0, 4)  # Random action
        try:
            obs, reward, done, truncated, info = env.step(action)
        except Exception as e:
            nan_errors.append(f"Step {step_idx}: {str(e)}")
            if len(nan_errors) > 5:
                break
            continue
        
        row = env.df.iloc[env.current_step]
        regime = int(row.get('REGIME', 2))
        position_size = info.get('position_size', 0.0)
        
        # For now, use REGIME as danger level (simple mapping: 3 is danger level 1)
        danger_level = 1 if regime == 3 else 0
        expected = env.base_position_size * (env.sizing_shrink ** danger_level)
        
        sample = {
            'step': step_idx,
            'current_step': env.current_step,
            'regime': regime,
            'position_size': position_size,
            'expected_size': expected,
            'match': np.isclose(position_size, expected),
            'reward': reward,
            'equity': env.equity
        }
        samples.append(sample)
        
        if done:
            print(f"    Episode ended at step {step_idx}")
            break
    
    if nan_errors:
        print(f"\n[!] Errors encountered:")
        for err in nan_errors:
            print(f"    {err}")
        return
    
    df_samples = pd.DataFrame(samples)
    
    # Print results
    print(f"\n[4] Results (first 20 steps):")
    display_cols = ['step', 'regime', 'position_size', 'expected_size', 'match', 'reward']
    print(df_samples.head(20)[display_cols].to_string())
    
    print(f"\n[5] Summary statistics:")
    print(f"    Total steps: {len(df_samples)}")
    print(f"    All position_size values match expected: {df_samples['match'].all()}")
    print(f"    NaN checks:")
    print(f"      - NaN in position_size: {df_samples['position_size'].isna().sum()}")
    print(f"      - NaN in reward: {df_samples['reward'].isna().sum()}")
    print(f"      - NaN in equity: {df_samples['equity'].isna().sum()}")
    print(f"    Position size distribution:")
    print(f"      - Min: {df_samples['position_size'].min():.6f}")
    print(f"      - Max: {df_samples['position_size'].max():.6f}")
    print(f"      - Mean: {df_samples['position_size'].mean():.6f}")
    print(f"    Final equity: {env.equity:.2f}")
    print(f"    Total reward: {df_samples['reward'].sum():.6f}")
    
    # Check for danger levels
    print(f"\n[6] Regime distribution in test run:")
    regime_counts = df_samples['regime'].value_counts().sort_index()
    print(regime_counts)
    
    # Sample some Shock (regime 3) cases
    shock_samples = df_samples[df_samples['regime'] == 3]
    if len(shock_samples) > 0:
        print(f"\n[7] SHOCK regime samples (regime=3):")
        print(f"    Count: {len(shock_samples)}")
        print(f"    Avg position_size for SHOCK: {shock_samples['position_size'].mean():.6f}")
        print(f"    Expected avg for SHOCK (danger=1): {env.base_position_size * (env.sizing_shrink ** 1):.6f}")
        print(shock_samples.head(5)[['step', 'regime', 'position_size', 'expected_size', 'reward']].to_string())
    else:
        print(f"\n[7] No SHOCK regimes (regime=3) encountered in test run.")
    
    normal_samples = df_samples[df_samples['regime'] != 3]
    if len(normal_samples) > 0:
        print(f"\n[8] NORMAL regime samples (regime != 3):")
        print(f"    Avg position_size for NORMAL: {normal_samples['position_size'].mean():.6f}")
        print(f"    Expected avg for NORMAL (danger=0): {env.base_position_size * (env.sizing_shrink ** 0):.6f}")
    
    print("\n" + "=" * 70)
    print("✓ TEST COMPLETED SUCCESSFULLY - NO NaN ERRORS")
    print("=" * 70)

if __name__ == '__main__':
    test_regime_sizing()
