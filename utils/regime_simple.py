"""
SIMPLIFIED AND ACCURATE regime detection system
Lighter on features, more accurate for real instability
"""
import pandas as pd
import numpy as np

class SimpleRegimeClassifier:
    """
    Regime classifier based on statistical logic,
    with no dependency on Hurst/ADF (too slow)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._compute_fast_features()
    
    def _compute_fast_features(self):
        """Compute fast features without ADF/Hurst"""
        df = self.df
        
        # ATR (compute it if missing)
        if 'ATR' not in df.columns:
            from utils.indicators import atr as calc_atr
            df['ATR'] = calc_atr(df, window=14)
        
        # Rolling volatility (already computed)
        if 'ROLL_STD_RET' not in df.columns:
            df['RET'] = np.log(df['Close'] / df['Close'].shift(1))
            df['ROLL_STD_RET'] = df['RET'].rolling(100).std()
        
        # 1. INTRADAY RANGE (High-Low as % of Close)
        df['INTRADAY_RANGE'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['INTRADAY_RANGE_MA'] = df['INTRADAY_RANGE'].rolling(50).mean()
        df['INTRADAY_RANGE_STD'] = df['INTRADAY_RANGE'].rolling(50).std()
        
        # 2. OVERNIGHT GAP (Open vs previous Close)
        df['OVERNIGHT_GAP'] = abs(df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
        df['OVERNIGHT_GAP_MA'] = df['OVERNIGHT_GAP'].rolling(50).mean()
        
        # 3. ATR ROLLING MEAN (per comparison)
        df['ATR_MA_100'] = df['ATR'].rolling(100).mean()
        
        # 4. MOMENTUM (RSI fast)
        df['MOMENTUM'] = df['Close'].pct_change(5).fillna(0) * 100
        df['MOMENTUM_MA'] = df['MOMENTUM'].rolling(20).mean()
        
        # 5. VOLUME ANOMALY
        if 'TickVolume' not in df.columns:
            df['TickVolume'] = 1
        df['VOL_MA'] = df['TickVolume'].rolling(100).mean()
        df['VOL_RATIO'] = df['TickVolume'] / (df['VOL_MA'] + 1e-9)
        
        # 6. CLOSE POSITION IN RANGE (0=low, 1=high)
        df['CLOSE_POSITION'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)
        
        # 7. TREND STRENGTH (using Close vs MA)
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['TREND_STRENGTH'] = abs(df['Close'] - df['MA_20']) / (df['MA_20'] + 1e-9) * 100
        
        self.df = df
    
    def classify_stability(self, row):
        """
        Classifies a candle as STABLE (0) or UNSTABLE (1)
        
        UNSTABLE indicators:
        - Intraday range 50%+ above average -> volatility spike
        - Overnight gap > 1% -> major event
        - Volume 2x+ the average -> volume stress
        - ATR 80%+ above average -> volatility peak
        """
        
        # Check NaNs
        intraday_range = row.get('INTRADAY_RANGE', 0)
        overnight_gap = row.get('OVERNIGHT_GAP', 0)
        vol_ratio = row.get('VOL_RATIO', 1.0)
        atr = row.get('ATR', 0)
        atr_ma = row.get('ATR_MA_100', 1)
        intraday_ma = row.get('INTRADAY_RANGE_MA', 1)
        intraday_std = row.get('INTRADAY_RANGE_STD', 1)
        
        if any(pd.isna([intraday_range, overnight_gap, vol_ratio, atr])):
            return 0  # Default stable if missing
        
        stability_score = 0
        
        # 1. ATR spike (80% above average)
        if atr > atr_ma * 1.8:
            stability_score += 1
        
        # 2. Intraday range spike (2+ std deviations)
        if intraday_std > 0 and intraday_range > intraday_ma + 2 * intraday_std:
            stability_score += 1
        
        # 3. Overnight gap
        if overnight_gap > 1.0:
            stability_score += 1
        
        # 4. Volume spike (2x+)
        if vol_ratio > 2.0:
            stability_score += 1
        
        # Unstable if 2+ indicators triggered
        return 1 if stability_score >= 2 else 0
    
    def add_stability_label(self):
        """Add STABILITY column to df"""
        self.df['STABILITY'] = self.df.apply(
            lambda row: self.classify_stability(row),
            axis=1
        )
        return self.df
    
    def get_stats(self):
        """Print statistics"""
        df = self.df
        total = len(df)
        stable = (df['STABILITY'] == 0).sum()
        unstable = (df['STABILITY'] == 1).sum()
        
        stats = {
            'total': total,
            'stable': stable,
            'stable_pct': 100 * stable / total,
            'unstable': unstable,
            'unstable_pct': 100 * unstable / total,
        }
        
        return stats


# TEST
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    
    from utils.data_utils import merge_h1_with_d1_regimes
    
    print("\n[1] Loading data...")
    df = merge_h1_with_d1_regimes()
    print(f"    {len(df)} candles")
    
    print("\n[2] Computing features...")
    classifier = SimpleRegimeClassifier(df)
    
    print("\n[3] Classifying stability...")
    df_classified = classifier.add_stability_label()
    
    print("\n[4] Statistics:")
    stats = classifier.get_stats()
    print(f"    Total: {stats['total']}")
    print(f"    Stable: {stats['stable']} ({stats['stable_pct']:.1f}%)")
    print(f"    Unstable: {stats['unstable']} ({stats['unstable_pct']:.1f}%)")
    
    # Compare with old Regime 3
    if 'REGIME' in df_classified.columns:
        shock_count = (df_classified['REGIME'] == 3).sum()
        print(f"\n    Regime 3 (old): {shock_count} ({100*shock_count/len(df):.1f}%)")
    
    print("\n[5] Sample unstable periods:")
    unstable = df_classified[df_classified['STABILITY'] == 1]
    if len(unstable) > 0:
        for idx in unstable.index[:5]:
            row = df_classified.iloc[idx]
            print(f"    {row['Time']}: Range={row['INTRADAY_RANGE']:.2f}%, Gap={row['OVERNIGHT_GAP']:.2f}%, Vol={row['VOL_RATIO']:.1f}x")
