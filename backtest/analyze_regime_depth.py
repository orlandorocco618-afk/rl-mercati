"""
Detailed analysis of the regime detection system
Checks whether the regimes are accurate or too conservative
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_utils import merge_h1_with_d1_regimes
from utils.regimes import compute_regime_features, classify_regime
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("IN-DEPTH REGIME DETECTION ANALYSIS")
print("="*80)

# Load full dataset
print("\n[1] Loading the dataset...")
df = merge_h1_with_d1_regimes()
df['Time'] = pd.to_datetime(df['Time'])
print(f"  Dataset: {len(df)} H1 candles")
print(f"  Range: {df['Time'].min()} to {df['Time'].max()}")

# Compute regime features
print("\n[2] Computing regime features...")
df = compute_regime_features(df)

# Dynamic thresholds
atr_series = df["ATR"].dropna()
atr_high = atr_series.quantile(0.70)
atr_extreme = atr_series.quantile(0.95)

print(f"  ATR thresholds:")
print(f"    High (70%): {atr_high:.2f}")
print(f"    Extreme (95%): {atr_extreme:.2f}")
print(f"    Mean: {atr_series.mean():.2f}")
print(f"    Std: {atr_series.std():.2f}")

# Classify all rows
print("\n[3] Classifying...")
df["REGIME"] = df.apply(
    lambda row: classify_regime(row, atr_high, atr_extreme),
    axis=1
)

regime_names = {0: "Mean Reversion", 1: "Trend", 2: "Noise", 3: "Shock/Extreme"}

# Overall distribution
print(f"\n[4] Overall distribution:")
for regime_id in range(4):
    count = (df['REGIME'] == regime_id).sum()
    pct = 100 * count / len(df)
    print(f"  {regime_names[regime_id]:20s}: {count:6d} ({pct:5.1f}%)")

# Period-by-period analysis
print("\n" + "="*80)
print("ANALYSIS BY PERIOD")
print("="*80)

periods = [
    ("2004-01-01", "2008-01-01", "Pre-Crisis Stable"),
    ("2008-01-01", "2008-09-01", "Crisis Begin (Bear Stearns → Lehman)"),
    ("2008-09-01", "2009-03-01", "Peak Crisis (Lehman → Bottom)"),
    ("2009-03-01", "2012-01-01", "Recovery Phase"),
    ("2012-01-01", "2024-12-31", "Post-Crisis Normal"),
]

crisis_analysis = []

for start_str, end_str, label in periods:
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)
    
    chunk = df[(df['Time'] >= start_dt) & (df['Time'] < end_dt)].copy()
    
    if len(chunk) == 0:
        continue
    
    print(f"\n📊 {label}")
    print(f"   Period: {start_str} → {end_str}")
    print(f"   Candles: {len(chunk)}")
    
    # Regime breakdown
    print(f"   Regimes:")
    for regime_id in range(4):
        count = (chunk['REGIME'] == regime_id).sum()
        pct = 100 * count / len(chunk)
        bar = "█" * int(pct / 2)
        print(f"     {regime_names[regime_id]:20s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # Volatility stats
    mean_atr = chunk['ATR'].mean()
    std_atr = chunk['ATR'].std()
    max_atr = chunk['ATR'].max()
    mean_vol = chunk['ROLL_STD_RET'].mean()
    
    print(f"   Volatility:")
    print(f"     ATR: mean={mean_atr:.2f}, std={std_atr:.2f}, max={max_atr:.2f}")
    print(f"     Ret volatility: {mean_vol:.4f}")
    
    # Feature stats
    hurst_mean = chunk['HURST'].mean()
    adf_mean = chunk['ADF_P'].mean()
    acf1_mean = chunk['ACF1'].mean()
    
    print(f"   Features:")
    print(f"     Hurst (mean={hurst_mean:.2f}, 0.5=random, <0.5=reversion, >0.5=trend)")
    print(f"     ADF p-val (mean={adf_mean:.2f}, <0.05=stationary)")
    print(f"     ACF1 (mean={acf1_mean:.2f}, <0=mean-reversion, >0=trending)")
    
    shock_pct = 100 * (chunk['REGIME'] == 3).sum() / len(chunk)
    crisis_analysis.append({
        'period': label,
        'shock_pct': shock_pct,
        'mean_atr': mean_atr,
        'hurst': hurst_mean,
        'adf_p': adf_mean
    })

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

crisis_df = pd.DataFrame(crisis_analysis)
print("\n" + crisis_df.to_string(index=False))

# Identify true instability
print("\n" + "="*80)
print("FINDING REAL INSTABILITY")
print("="*80)

# Method 1: High ATR periods
print("\n[Method 1] High-ATR periods (>99.5 percentile):")
extreme_atr = atr_series.quantile(0.995)
extreme_rows = df[df['ATR'] > extreme_atr]
if len(extreme_rows) > 0:
    print(f"  {len(extreme_rows)} candle ({100*len(extreme_rows)/len(df):.2f}%) with ATR > {extreme_atr:.2f}")
    print(f"  Date sample: {extreme_rows['Time'].iloc[0]} ... {extreme_rows['Time'].iloc[-1]}")

# Method 2: Large price moves (gaps or sudden volatility)
print("\n[Method 2] Large intraday moves:")
df['MOVE_PCT'] = abs(df['Close'] - df['Open']) / df['Open'] * 100
large_moves = df[df['MOVE_PCT'] > 2.0]  # >2% moves
print(f"  {len(large_moves)} candle ({100*len(large_moves)/len(df):.2f}%) with move > 2%")

# Method 3: Regime switches (instability signals)
print("\n[Method 3] Regime switches (fast changes):")
df['REGIME_CHANGE'] = df['REGIME'].diff() != 0
switches = df[df['REGIME_CHANGE']]
print(f"  {len(switches)} regime switches ({100*len(switches)/len(df):.2f}%)")

# Find clusters of instability
print("\n[Method 4] Instability clusters (5+ shock candles in 24h):")
instability_clusters = []
current_cluster = None

for i, row in df.iterrows():
    if row['REGIME'] == 3 or row['MOVE_PCT'] > 2.0:
        if current_cluster is None:
            current_cluster = {'start': row['Time'], 'candles': 1}
        else:
            # Check if within 24 hours
            hours_diff = (row['Time'] - current_cluster['start']).total_seconds() / 3600
            if hours_diff <= 24:
                current_cluster['candles'] += 1
            else:
                if current_cluster['candles'] >= 5:
                    instability_clusters.append(current_cluster)
                current_cluster = {'start': row['Time'], 'candles': 1}
    else:
        if current_cluster is not None and current_cluster['candles'] >= 5:
            instability_clusters.append(current_cluster)
        current_cluster = None

print(f"  Found {len(instability_clusters)} instability clusters (5+ events in 24h)")
for i, cluster in enumerate(instability_clusters[:10]):
    print(f"    {i+1}. {cluster['start'].date()}: {cluster['candles']} events")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("""
1. ⚠️ Regime 3 (Shock/Extreme) is CONSERVATIVE - it only removes extreme peaks
   - The 95th-percentile ATR threshold is reasonable for real shocks
   - But there may be other forms of instability it doesn't catch

2. 💡 Consider adding:
   - Intraday volatility (High-Low range)
   - Opening gaps (Open vs previous Close)
   - Clustering of regime changes
   - Abnormal volumes

3. 🤖 RL system for regime detection:
   - Trainer: a classifier that predicts the regime from features
   - Supervised: hand-label 100-200 candles as stable/unstable
   - RL reward: correct prediction = +1, error = -1
   - Converges to an optimal classification

4. 🔧 Gradual filtering options:
   - Level 1 (Conservative): only Regime 3 removed
   - Level 2 (Moderate): Regime 3 + large moves >2%
   - Level 3 (Aggressive): Regime 3 + moves >1.5% + high vol
""")
