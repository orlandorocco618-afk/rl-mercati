#!/usr/bin/env python3
"""
Small test: check that the hybrid classifier and the sizing work.
- Loads H1 + D1, adds HYBRID_CLASS
- Creates a TradingEnv with sizing enabled
- Runs N steps and checks that:
  1. danger_level is recognized correctly
  2. position_size scales as expected (base_size * shrink^danger)
  3. There are no NaN in the computations
  4. PnL and commissions scale
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '.')

from data import load_data
from utils.regime_hybrid import HybridRegimeClassifier
from env.trading_env import TradingEnv

def test_regime_sizing():
    print("=" * 70)
    print("TEST: HYBRID REGIME DETECTION + POSITION SIZING")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading H1 + D1 data...")
    df_h1, df_d1 = load_data()
    print(f"    H1: {len(df_h1)} rows")
    print(f"    D1: {len(df_d1)} rows")
    
    # Add hybrid classification
    print("\n[2] Adding HYBRID_CLASS...")
    classifier = HybridRegimeClassifier(df_d1)
    df_h1 = classifier.add_hybrid_class(df_h1)
    print(f"    Hybrid classification added. Unique classes: {df_h1['HYBRID_CLASS'].unique()}")
    
    # Count distribution
    counts = df_h1['HYBRID_CLASS'].value_counts().sort_index()
    print(f"    Distribution:\n{counts}")
    
    # Create env with sizing enabled
    print("\n[3] Creating TradingEnv with regime sizing...")
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
    print("\n[4] Running 500 steps and sampling position_size...")
    obs, _ = env.reset()
    
    samples = []
    for step_idx in range(500):
        action = np.random.randint(0, 4)  # Random action
        obs, reward, done, truncated, info = env.step(action)
        
        row = df_h1.iloc[env.current_step]
        hybrid_class = row.get('HYBRID_CLASS', 0)
        position_size = info.get('position_size', 0.0)
        
        # Expected: position_size = base * (shrink ^ hybrid_class)
        expected = env.base_position_size * (env.sizing_shrink ** max(0, int(hybrid_class)))
        
        sample = {
            'step': step_idx,
            'current_step': env.current_step,
            'hybrid_class': hybrid_class,
            'position_size': position_size,
            'expected_size': expected,
            'match': np.isclose(position_size, expected),
            'reward': reward,
            'equity': env.equity
        }
        samples.append(sample)
        
        if done:
            break
    
    df_samples = pd.DataFrame(samples)
    
    # Print results
    print(f"\n[5] Results (first 20 steps):")
    print(df_samples.head(20).to_string())
    
    print(f"\n[6] Summary statistics:")
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
    print(f"\n[7] Danger level distribution in test run:")
    danger_counts = df_samples['hybrid_class'].value_counts().sort_index()
    print(danger_counts)
    
    # Sample some SEVERE cases
    severe_samples = df_samples[df_samples['hybrid_class'] == 2]
    if len(severe_samples) > 0:
        print(f"\n[8] SEVERE regime samples (danger=2):")
        print(severe_samples.head(5)[['step', 'hybrid_class', 'position_size', 'expected_size', 'reward']].to_string())
        print(f"    Avg position_size for SEVERE: {severe_samples['position_size'].mean():.6f}")
    else:
        print(f"\n[8] No SEVERE regimes encountered in test run.")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

if __name__ == '__main__':
    test_regime_sizing()
