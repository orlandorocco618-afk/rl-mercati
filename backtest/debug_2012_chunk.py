"""
Checks the 2012-2024 dataset to understand why everything turns into NaN
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.data_utils import merge_h1_with_d1_regimes

print("\nLoading the dataset...")
df = merge_h1_with_d1_regimes()
df['Time'] = pd.to_datetime(df['Time'])

chunk_2012 = df[(df['Time'] >= pd.to_datetime('2012-01-01')) & (df['Time'] < pd.to_datetime('2024-12-31'))].copy()
chunk_2012 = chunk_2012.reset_index(drop=True)

print(f"\nChunk 2012-2024 BEFORE filtering:")
print(f"  Rows: {len(chunk_2012)}")
print(f"  Close: {chunk_2012['Close'].notna().sum()}/{len(chunk_2012)} non-NaN")
print(f"  Close sample: {chunk_2012['Close'].head()}")
print(f"  Regimes: {chunk_2012['REGIME'].value_counts().to_dict()}")

# Filter shock
if 'REGIME' in chunk_2012.columns:
    original_len = len(chunk_2012)
    chunk_filtered = chunk_2012[chunk_2012['REGIME'] != 3].copy().reset_index(drop=True)
    removed = original_len - len(chunk_filtered)
    
    print(f"\nAFTER shock filtering:")
    print(f"  Rows: {len(chunk_filtered)} (removed {removed})")
    print(f"  Close: {chunk_filtered['Close'].notna().sum()}/{len(chunk_filtered)} non-NaN")
    print(f"  Close sample: {chunk_filtered['Close'].head()}")
