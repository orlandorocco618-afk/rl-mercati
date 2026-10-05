"""
Finds crisis periods in the D1 dataset and gives recommendations
for skipping them during training
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_utils import load_csv
from utils.regimes import compute_regime_features, classify_regime

print("\n" + "="*80)
print("CRISIS PERIOD DETECTION")
print("="*80)

# Load D1 data
print("\n[1] Loading D1...")
df_d1 = load_csv("data/xauusd_d1_clean.csv")
print(f"  D1: {len(df_d1)} rows, from {df_d1['Time'].min()} to {df_d1['Time'].max()}")

# Compute regime features
print("\n[2] Computing regime features...")
df_d1 = compute_regime_features(df_d1)

# Calculate ATR thresholds for classification
atr_high = df_d1["ATR"].quantile(0.75)
atr_extreme = df_d1["ATR"].quantile(0.95)

print(f"  ATR high (75%):     {atr_high:.2f}")
print(f"  ATR extreme (95%):  {atr_extreme:.2f}")

# Classify all rows
print("\n[3] Classifying regimes...")
df_d1["REGIME"] = df_d1.apply(
    lambda row: classify_regime(row, atr_high, atr_extreme),
    axis=1
)

regime_names = {0: "Mean Reversion", 1: "Trend", 2: "Noise", 3: "Shock/Extreme"}
regime_counts = df_d1["REGIME"].value_counts().sort_index()
print(f"  Regime distribution:")
for regime_id, count in regime_counts.items():
    pct = 100 * count / len(df_d1)
    print(f"    {regime_names.get(regime_id, '?')}: {count:4d} ({pct:5.1f}%)")

# Identify crisis periods (regime 3 or consecutive high volatility)
print("\n[4] Finding crisis periods...")
df_d1['Time'] = pd.to_datetime(df_d1['Time'])
df_d1 = df_d1.sort_values('Time')

# Mark crisis rows (regime 3)
crisis_rows = df_d1[df_d1['REGIME'] == 3]
print(f"  Rows in the Shock regime: {len(crisis_rows)} of {len(df_d1)} ({100*len(crisis_rows)/len(df_d1):.1f}%)")

# Find crisis periods (consecutive days with high volatility or regime 3)
crisis_periods = []
current_period_start = None

for i, row in df_d1.iterrows():
    is_crisis = (row['REGIME'] == 3) or (row['ROLL_STD_RET'] > df_d1['ROLL_STD_RET'].quantile(0.90))
    
    if is_crisis:
        if current_period_start is None:
            current_period_start = row['Time']
    else:
        if current_period_start is not None:
            crisis_periods.append((current_period_start, row['Time']))
            current_period_start = None

if current_period_start is not None:
    crisis_periods.append((current_period_start, df_d1['Time'].iloc[-1]))

# Merge overlapping/close periods
print(f"\n  Crisis periods (consecutive):  {len(crisis_periods)}")
for start, end in crisis_periods[:10]:  # Show first 10
    duration_days = (end - start).days
    print(f"    {start.date()} → {end.date()} ({duration_days} days)")

# Analyze our chunks
print("\n" + "="*80)
print("TIME CHUNK ANALYSIS")
print("="*80)

chunks = [
    ("2004-01-01", "2008-01-01", "Pre-crisis growth"),
    ("2008-01-01", "2012-01-01", "2008-2012 financial crisis"),
    ("2012-01-01", "2024-12-31", "Post-crisis recovery"),
]

for start_str, end_str, label in chunks:
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)
    
    chunk = df_d1[(df_d1['Time'] >= start_dt) & (df_d1['Time'] < end_dt)]
    
    if len(chunk) == 0:
        print(f"\n✗ {label} ({start_str} → {end_str}): NO DATA")
        continue
    
    shock_count = (chunk['REGIME'] == 3).sum()
    shock_pct = 100 * shock_count / len(chunk)
    
    mean_vol = chunk['ROLL_STD_RET'].mean()
    max_vol = chunk['ROLL_STD_RET'].max()
    mean_atr = chunk['ATR'].mean()
    
    status = "⚠ CRISIS PERIOD" if shock_pct > 15 else "✓ OK"
    
    print(f"\n{status} {label}")
    print(f"  Period:      {start_str} → {end_str}")
    print(f"  Days:        {len(chunk)}")
    print(f"  Shock days:  {shock_count} ({shock_pct:.1f}%)")
    print(f"  Volatility:  mean={mean_vol:.4f}, max={max_vol:.4f}")
    print(f"  Avg ATR:     {mean_atr:.2f}")
    
    if shock_pct > 15:
        print(f"  → RECOMMENDATION: skip this chunk during training")

print("\n" + "="*80 + "\n")
