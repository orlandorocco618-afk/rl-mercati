"""
HYBRID SYSTEM: Regime 3 + STABILITY

Combines the Regime 3 logic (for extreme ATR) and STABILITY
(for intraday spikes) for more accurate filtering

The logic:
- If ATR is in the top 5% (a real shock) → UNSTABLE
- OR if 3+ stress indicators fire → UNSTABLE
- Otherwise → STABLE
"""
import pandas as pd
import numpy as np

class HybridRegimeClassifier:
    """
    Hybrid classifier that combines:
    1. ATR extremeness (Regime 3 logic)
    2. Multiple stress indicators (STABILITY logic)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._compute_features()
    
    def _compute_features(self):
        """Compute all needed features"""
        df = self.df
        
        # ATR percentile
        atr_95 = df['ATR'].quantile(0.95)
        self.atr_95 = atr_95
        
        # Intraday range
        df['INTRADAY_RANGE'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['INTRADAY_RANGE_MA'] = df['INTRADAY_RANGE'].rolling(50).mean()
        df['INTRADAY_RANGE_STD'] = df['INTRADAY_RANGE'].rolling(50).std()
        
        # Overnight gap
        df['OVERNIGHT_GAP'] = abs(df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
        
        # Volume ratio
        if 'TickVolume' not in df.columns:
            df['TickVolume'] = 1
        df['VOL_MA'] = df['TickVolume'].rolling(100).mean()
        df['VOL_RATIO'] = df['TickVolume'] / (df['VOL_MA'] + 1e-9)
        
        # ATR rolling mean
        df['ATR_MA_100'] = df['ATR'].rolling(100).mean()
        
        self.df = df
    
    def classify(self, row):
        """
        Hybrid classification:
        
        TIER 1 (Certain Unstable):
          - ATR > 95° percentile → definitely unstable
        
        TIER 2 (Multiple Stress):
          - 3+ stress indicators → likely unstable
        
        TIER 3 (Single Stress):
          - 1-2 stress indicators → probably stable
        
        Returns:
          0 = STABLE (keep for training)
          1 = UNSTABLE_MILD (monitor)
          2 = UNSTABLE_SEVERE (exclude)
        """
        
        atr = row.get('ATR', 0)
        atr_ma = row.get('ATR_MA_100', 1)
        intraday = row.get('INTRADAY_RANGE', 0)
        intraday_ma = row.get('INTRADAY_RANGE_MA', 1)
        intraday_std = row.get('INTRADAY_RANGE_STD', 1)
        gap = row.get('OVERNIGHT_GAP', 0)
        vol_ratio = row.get('VOL_RATIO', 1)
        
        # Tier 1: Extreme ATR
        if atr > self.atr_95:
            return 2  # UNSTABLE_SEVERE
        
        # Count stress indicators
        stress_count = 0
        
        # Indicator 1: ATR spike (80% above mean)
        if atr > atr_ma * 1.8:
            stress_count += 1
        
        # Indicator 2: Intraday range spike
        if intraday_std > 0:
            z_score = (intraday - intraday_ma) / intraday_std
            if z_score > 1.5:  # 1.5 std above mean
                stress_count += 1
        
        # Indicator 3: Overnight gap
        if gap > 1.0:
            stress_count += 1
        
        # Indicator 4: Volume spike
        if vol_ratio > 2.0:
            stress_count += 1
        
        # Tier 2: Multiple stress
        if stress_count >= 3:
            return 2  # UNSTABLE_SEVERE
        
        # Tier 3: Some stress but not alarming
        if stress_count >= 2:
            return 1  # UNSTABLE_MILD
        
        return 0  # STABLE
    
    def add_hybrid_class(self):
        """Add HYBRID_CLASS column"""
        self.df['HYBRID_CLASS'] = self.df.apply(
            lambda row: self.classify(row),
            axis=1
        )
        return self.df
    
    def get_stats(self):
        """Statistics"""
        df = self.df
        stable = (df['HYBRID_CLASS'] == 0).sum()
        mild = (df['HYBRID_CLASS'] == 1).sum()
        severe = (df['HYBRID_CLASS'] == 2).sum()
        total = len(df)
        
        return {
            'total': total,
            'stable': stable,
            'stable_pct': 100 * stable / total,
            'mild': mild,
            'mild_pct': 100 * mild / total,
            'severe': severe,
            'severe_pct': 100 * severe / total,
            'excluded_if_severe': severe,
            'excluded_pct_if_severe': 100 * severe / total,
            'excluded_if_mild_severe': mild + severe,
            'excluded_pct_if_mild_severe': 100 * (mild + severe) / total,
        }


# TEST
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    
    from utils.data_utils import merge_h1_with_d1_regimes
    from utils.regimes import compute_regime_features, classify_regime
    from utils.regime_simple import SimpleRegimeClassifier
    
    print("\n" + "="*80)
    print("HYBRID REGIME CLASSIFICATION SYSTEM")
    print("="*80)
    
    print("\n[1] Loading data...")
    df = merge_h1_with_d1_regimes()
    df = compute_regime_features(df)
    df['Time'] = pd.to_datetime(df['Time'])
    
    print("\n[2] Adding old systems...")
    atr_series = df["ATR"].dropna()
    atr_extreme = atr_series.quantile(0.95)
    atr_high = atr_series.quantile(0.70)
    df["REGIME_3"] = df.apply(
        lambda row: classify_regime(row, atr_high, atr_extreme),
        axis=1
    ) == 3
    
    classifier_simple = SimpleRegimeClassifier(df)
    df = classifier_simple.add_stability_label()
    df['STABILITY_UNSTABLE'] = df['STABILITY'] == 1
    
    print("\n[3] Computing hybrid classification...")
    classifier_hybrid = HybridRegimeClassifier(df)
    df = classifier_hybrid.add_hybrid_class()
    
    print("\n[4] Statistics:")
    stats = classifier_hybrid.get_stats()
    print(f"   Total candles: {stats['total']}")
    print(f"   STABLE: {stats['stable']} ({stats['stable_pct']:.1f}%)")
    print(f"   UNSTABLE_MILD: {stats['mild']} ({stats['mild_pct']:.1f}%)")
    print(f"   UNSTABLE_SEVERE: {stats['severe']} ({stats['severe_pct']:.1f}%)")
    
    print(f"\n   Filtering scenarios:")
    print(f"     If exclude SEVERE only: {stats['severe']} excluded ({stats['excluded_pct_if_severe']:.1f}%)")
    print(f"     If exclude MILD+SEVERE: {stats['excluded_if_mild_severe']} excluded ({stats['excluded_pct_if_mild_severe']:.1f}%)")
    
    # Comparison
    print(f"\n[5] Comparison with old systems:")
    regime3_count = df['REGIME_3'].sum()
    stability_count = df['STABILITY_UNSTABLE'].sum()
    hybrid_severe = (df['HYBRID_CLASS'] == 2).sum()
    
    print(f"   Regime 3: {regime3_count} ({100*regime3_count/len(df):.1f}%)")
    print(f"   STABILITY: {stability_count} ({100*stability_count/len(df):.1f}%)")
    print(f"   HYBRID (SEVERE): {hybrid_severe} ({100*hybrid_severe/len(df):.1f}%)")
    
    # Sample unstable
    print(f"\n[6] Sample SEVERE unstable events:")
    severe_events = df[df['HYBRID_CLASS'] == 2].head(10)
    for idx, row in severe_events.iterrows():
        atr = row['ATR']
        intraday = row['INTRADAY_RANGE']
        gap = row['OVERNIGHT_GAP']
        vol = row['VOL_RATIO']
        print(f"   {row['Time']}: ATR={atr:.2f}, Intraday={intraday:.2f}%, Gap={gap:.2f}%, Vol={vol:.1f}x")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"""
Use HYBRID_CLASS for filtering:

OPTION 1 (Conservative):
  - Exclude SEVERE only ({stats['severe']} candles, {stats['excluded_pct_if_severe']:.1f}%)
  - Keeps 99.5% of the data
  - Best for maximum training data

OPTION 2 (Moderate):
  - Exclude MILD + SEVERE ({stats['excluded_if_mild_severe']} candles, {stats['excluded_pct_if_mild_severe']:.1f}%)
  - Reasonable compromise
  - Try this if Option 1 doesn't work

Advantages of the hybrid system:
  ✓ Keeps Regime 3 for extreme ATR (real shocks)
  ✓ Adds STABILITY for localized intraday spikes
  ✓ Multi-tier: classifies severity, not just binary
  ✓ Granular: filtering can be increased step by step
    """)
