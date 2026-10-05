"""
FLAT resolution: quick test with max entropy + free hold
We train a new small model on 2008-2012 with:
- allow_hold_when_flat = True
- ent_coef = 0.1 (max entropy for exploration)
- 5000 quick timesteps for debugging
"""
import os
import sys
sys.path.insert(0, '.')

from pathlib import Path
import pandas as pd
import numpy as np

from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter
from env.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from backtest.analyze_actions_v2 import analyze_actions_v2

def slice_df_by_dates(df, start, end):
    """Slice dataframe by date range using Time column"""
    if 'Time' in df.columns:
        df_loc = df.copy()
        df_loc['Time'] = pd.to_datetime(df_loc['Time'])
        mask = (df_loc['Time'] >= pd.to_datetime(start)) & (df_loc['Time'] < pd.to_datetime(end))
        return df_loc.loc[mask].reset_index(drop=True)
    return pd.DataFrame()

def make_env(df, allow_hold=True):
    """Create environment with custom parameters"""
    def _init():
        return TradingEnv(
            df=df, 
            window_size=100, 
            initial_capital=10000.0, 
            max_drawdown=0.30,
            allow_hold_when_flat=allow_hold
        )
    return _init

print("\n" + "="*80)
print("TEST FLAT RESOLUTION: Max Entropy + Hold Allowed")
print("="*80)

# Load data
print("\n[INIT] Loading data...")
df = merge_h1_with_d1_regimes()
df = apply_kalman_filter(df, price_col='Close')
df['Close'] = df['Close_KF']

# Get 2008-2012 chunk for testing
train_df = slice_df_by_dates(df, '2008-01-01', '2012-01-01')
print(f"  Train slice: {len(train_df)} rows")

if len(train_df) == 0:
    print("  ERROR: Empty slice")
    sys.exit(1)

# Create env with HOLD allowed
print("\n[CONFIG] Creating the environment with hold allowed (allow_hold_when_flat=True)")
env = DummyVecEnv([make_env(train_df, allow_hold=True)])

# Train with VERY HIGH entropy
print("\n[TRAIN] Quick training (5000 steps) with ent_coef=0.1")
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.1,  # VERY HIGH entropy - forces exploration
    verbose=0,
)
model.learn(total_timesteps=5000)

model_path = "models/debug_flat_fix.zip"
os.makedirs("models", exist_ok=True)
model.save(model_path)
print(f"  Model saved: {model_path}")

# Generate actions
print("\n[INFERENCE] Generating actions...")
test_env = DummyVecEnv([make_env(train_df, allow_hold=True)])
obs = test_env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

actions = []
for i in range(min(5000, len(train_df))):
    act, _ = model.predict(obs, deterministic=True)
    actions.append(int(act[0]))
    obs, _, _, _ = test_env.step(act)
    if i % 1000 == 0 and i > 0:
        print(f"  Progress: {i}")

print(f"  Generated {len(actions)} actions")

# Save and analyze
print("\n[SAVE] Saving actions...")
df_out = train_df.iloc[:len(actions)].copy()
if 'Close_KF' in df_out.columns:
    df_out['Close'] = df_out['Close_KF']
df_out['Close'] = df_out['Close'].fillna(method='ffill').fillna(method='bfill')
df_out['Action'] = actions

out_csv = "backtest/data/debug_flat_fix_actions.csv"
os.makedirs("backtest/data", exist_ok=True)
df_out.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}")

# Analyze
print("\n[ANALYSIS] Analisi rapida...")
out_dir = "backtest/plots_debug_flat"
os.makedirs(out_dir, exist_ok=True)

try:
    stats = analyze_actions_v2(actions_csv=out_csv, initial_capital=10000.0, save_dir=out_dir)
    
    # Action distribution
    action_dist = stats.get('action_distribution', {})
    print(f"\n[ACTIONS]")
    print(f"  HOLD:   {action_dist.get('HOLD', 0):6d} ({100*action_dist.get('HOLD', 0)/len(actions):5.1f}%)")
    print(f"  LONG:   {action_dist.get('LONG', 0):6d} ({100*action_dist.get('LONG', 0)/len(actions):5.1f}%)")
    print(f"  SHORT:  {action_dist.get('SHORT', 0):6d} ({100*action_dist.get('SHORT', 0)/len(actions):5.1f}%)")
    print(f"  CLOSE:  {action_dist.get('CLOSE', 0):6d} ({100*action_dist.get('CLOSE', 0)/len(actions):5.1f}%)")
    
    print(f"\n[RESULTS]")
    print(f"  Return:    {stats.get('total_return_pct', 0):+.2f}%")
    print(f"  Trades:    {stats.get('total_trades', 0)}")
    print(f"  Win Rate:  {stats.get('pct_profitable', 0):.1f}%")
    print(f"  Drawdown:  {stats.get('max_drawdown_pct', 0):.2f}%")
    
    if stats.get('total_trades', 0) > 0:
        print("\n✓ RESOLUTION SUCCESSFUL: we have trades!")
    else:
        print("\n✗ Still no trades - the problem remains")
        
except Exception as e:
    print(f"  Error during analysis: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80 + "\n")
