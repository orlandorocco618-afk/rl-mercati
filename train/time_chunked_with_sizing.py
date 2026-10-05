#!/usr/bin/env python3
"""
Time-Chunked Validation with Regime Sizing
=========================================

Trains and validates on time chunks with:
- H1 + D1 regime data loading
- HYBRID_CLASS for danger scoring (optional)
- Adaptive position sizing per regime
- PPO training with allow_hold_when_flat + ent_coef
- PnL, win rate and drawdown metrics per chunk
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.regimes import add_regimes
from utils.regime_hybrid import HybridRegimeClassifier
from env.trading_env import TradingEnv

def load_h1_with_regimes(add_hybrid=False):
    """Load H1 + D1 regimes, optionally add HYBRID_CLASS."""
    df_h1 = pd.read_csv('data/xauusd_h1_clean.csv', parse_dates=['Date'])
    df_d1 = pd.read_csv('data/xauusd_d1_clean.csv', parse_dates=['Date'])
    
    df_d1.rename(columns={'Date': 'datetime'}, inplace=True)
    df_h1.rename(columns={'Date': 'datetime'}, inplace=True)
    
    # Numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df_h1.columns:
            df_h1[col] = pd.to_numeric(df_h1[col], errors='coerce')
        if col in df_d1.columns:
            df_d1[col] = pd.to_numeric(df_d1[col], errors='coerce')
    
    # Add D1 regimes
    df_d1 = add_regimes(df_d1)
    
    # Merge regimes into H1
    df_d1_regimes = df_d1[['datetime', 'REGIME']].copy()
    df_d1_regimes['date_only'] = df_d1_regimes['datetime'].dt.date
    
    df_h1['date_only'] = df_h1['datetime'].dt.date
    df_h1 = df_h1.merge(df_d1_regimes[['date_only', 'REGIME']], on='date_only', how='left')
    df_h1['REGIME'] = df_h1['REGIME'].fillna(2).astype(int)
    df_h1.drop('date_only', axis=1, inplace=True)
    
    # Optionally add HYBRID_CLASS
    if add_hybrid:
        try:
            classifier = HybridRegimeClassifier(df_h1)
            df_h1 = classifier.add_hybrid_class()
        except Exception as e:
            print(f"  Warning: Could not add HYBRID_CLASS: {e}")
    
    return df_h1

def slice_by_date(df, start_date, end_date):
    """Slice dataframe by date range."""
    mask = (df['datetime'] >= pd.Timestamp(start_date)) & (df['datetime'] < pd.Timestamp(end_date))
    return df[mask].reset_index(drop=True)

def run_chunk_training(df_train, df_test, chunk_name, model_path=None):
    """Train on df_train, evaluate on df_test."""
    print(f"\n  [{chunk_name}] Train: {len(df_train)} rows, Test: {len(df_test)} rows")
    
    if len(df_train) < 200 or len(df_test) < 100:
        print(f"    Skipping (insufficient data)")
        return None
    
    # Train env
    print(f"    Creating train env...")
    env_train = TradingEnv(
        df=df_train,
        window_size=100,
        initial_capital=10000.0,
        commission=0.0002,
        allow_hold_when_flat=True,
        base_position_size=0.1,
        sizing_shrink=0.9,
        enable_regime_sizing=True,
        regime_hard_stop=False,
        enforce_directional_rebalance=True,
        rebalance_after_entries=30,
        rebalance_dominance_threshold=0.70,
        directional_balance_reward_coef=0.001
    )
    
    vec_env = DummyVecEnv([lambda: env_train])
    
    print(f"    Training PPO for 5000 steps...")
    try:
        model = PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            ent_coef=0.1,
            max_grad_norm=0.5,
            verbose=0
        )
        model.learn(total_timesteps=5000, log_interval=1)
        print(f"    ✓ Training completed")
    except Exception as e:
        print(f"    ✗ Training failed: {str(e)}")
        return None
    
    # Test env
    print(f"    Creating test env...")
    env_test = TradingEnv(
        df=df_test,
        window_size=100,
        initial_capital=10000.0,
        commission=0.0002,
        allow_hold_when_flat=True,
        base_position_size=0.1,
        sizing_shrink=0.9,
        enable_regime_sizing=True,
        regime_hard_stop=False,
        enforce_directional_rebalance=False,
        directional_balance_reward_coef=0.0003
    )
    
    print(f"    Running test inference...")
    obs = env_test.reset()[0]
    trades = []
    errors = []
    
    for step in range(min(1000, len(df_test) - 100)):
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_test.step(action)
            
            trades.append({
                'action': action,
                'position': env_test.position,
                'reward': reward,
                'equity': env_test.equity,
                'done': done or truncated
            })
            
            if done or truncated:
                obs = env_test.reset()[0]
        except Exception as e:
            errors.append(f"Step {step}: {str(e)}")
            if len(errors) > 3:
                break
    
    if errors:
        print(f"    ✗ Errors during inference: {errors[0]}")
        return None
    
    # Calculate metrics
    df_trades = pd.DataFrame(trades)
    equity_seq = df_trades['equity'].values
    initial_equity = 10000.0
    final_equity = equity_seq[-1] if len(equity_seq) > 0 else initial_equity
    pnl = (final_equity - initial_equity) / initial_equity * 100
    
    # Drawdown
    cum_returns = equity_seq / initial_equity
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (running_max - cum_returns) / running_max
    max_dd = drawdown.max() if len(drawdown) > 0 else 0
    
    # Trade count
    position_changes = (df_trades['position'].diff() != 0).sum()
    
    print(f"    PnL: {pnl:.2f}% | Final Equity: {final_equity:.2f} | Max DD: {max_dd*100:.2f}% | Pos Changes: {position_changes}")
    
    return {
        'chunk': chunk_name,
        'pnl_pct': pnl,
        'final_equity': final_equity,
        'max_drawdown': max_dd,
        'trades': position_changes,
        'errors': len(errors)
    }

def main():
    print("=" * 70)
    print("TIME-CHUNKED VALIDATION WITH REGIME SIZING")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_h1_with_regimes(add_hybrid=False)  # Can set True for HYBRID_CLASS
    print(f"    Total rows: {len(df)}")
    print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"    REGIME distribution: {df['REGIME'].value_counts().sort_index().to_dict()}")
    
    # Define chunks
    print("\n[2] Defining time chunks...")
    chunks = [
        ('2004-06-01', '2006-01-01', '2006-03-01'),  # Pre-crisis
        ('2006-01-01', '2008-01-01', '2008-06-01'),  # Pre-crisis 2
    ]
    
    results = []
    for train_start, train_end, test_end in chunks:
        chunk_name = f"{train_start[:4]}-{train_end[:4]}"
        df_train = slice_by_date(df, train_start, train_end)
        df_test = slice_by_date(df, train_end, test_end)
        
        result = run_chunk_training(df_train, df_test, chunk_name)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n{df_results.to_string(index=False)}")
        print(f"\nAvg PnL: {df_results['pnl_pct'].mean():.2f}%")
        print(f"Total errors: {df_results['errors'].sum()}")
    else:
        print("No successful runs.")
    
    print("\n" + "=" * 70)
    print("✓ TIME-CHUNKED VALIDATION COMPLETED")
    print("=" * 70)

if __name__ == '__main__':
    main()
