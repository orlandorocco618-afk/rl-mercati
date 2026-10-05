"""
Debug: investigates the NaN problem in the 2008-2012 dataset
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_utils import merge_h1_with_d1_regimes

print("\nLoading the dataset...")
df = merge_h1_with_d1_regimes()
print(f"Full dataset: {len(df)} rows")

# Check for chunk 2008-2012
df['Time'] = pd.to_datetime(df['Time'])
chunk_2008 = df[(df['Time'] >= pd.to_datetime('2008-01-01')) & (df['Time'] < pd.to_datetime('2012-01-01'))].copy()

print(f"\nChunk 2008-2012: {len(chunk_2008)} rows")

# Count NaNs in key columns
key_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Time']
print("\nNaN counts per column:")
for col in key_cols:
    if col in chunk_2008.columns:
        nan_count = chunk_2008[col].isna().sum()
        pct = 100 * nan_count / len(chunk_2008)
        print(f"  {col:15s}: {nan_count:6d} ({pct:5.1f}%)")

# Check for non-numeric values in Close
print(f"\nClose column type: {chunk_2008['Close'].dtype}")
print(f"Sample Close values (first 10): {chunk_2008['Close'].head(10).tolist()}")

# Check for zeros or negative values
if 'Close' in chunk_2008.columns:
    close_numeric = pd.to_numeric(chunk_2008['Close'], errors='coerce')
    zeros = (close_numeric == 0).sum()
    negatives = (close_numeric < 0).sum()
    print(f"Zero Close values: {zeros}")
    print(f"Negative Close values: {negatives}")

# Check time gaps
print(f"\nTime range: {chunk_2008['Time'].min()} to {chunk_2008['Time'].max()}")
print(f"Time gaps analysis:")
time_diffs = chunk_2008['Time'].diff()
print(f"  Min time gap: {time_diffs.min()}")
print(f"  Max time gap: {time_diffs.max()}")
print(f"  Gaps > 2 hours: {(time_diffs > pd.Timedelta(hours=2)).sum()}")
