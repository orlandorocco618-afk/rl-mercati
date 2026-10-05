import os
import numpy as np
import pandas as pd

from .indicators import (
    atr,
    log_returns,
    rolling_volatility,
    hurst_exponent,
    adf_pvalue,
    acf1
)


# ============================================================
#  ADVANCED FEATURES
# ============================================================

def rolling_slope(series, window=50):
    """
    Computes the trend slope with a linear regression.
    """
    if len(series) < window:
        return np.nan
    y = series[-window:]
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]
    return slope


# ============================================================
#  FEATURE COMPUTATION FOR REGIME DETECTION
# ============================================================

def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["RET"] = log_returns(df["Close"])

    # ATR
    df["ATR"] = atr(df, window=14)

    # Rolling volatility
    df["ROLL_STD_RET"] = rolling_volatility(df["RET"], window=100)

    # Trend slope
    df["SLOPE"] = df["Close"].rolling(50).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 50 else np.nan
    )

    # Bollinger Band width
    df["BB_WIDTH"] = (df["Close"].rolling(20).std() * 2) / df["Close"]

    # Range ratio
    df["RANGE_RATIO"] = (df["High"] - df["Low"]) / (df["ATR"] + 1e-9)

    # Volume percentile
    # Volume column compatibility: prefers 'TickVolume', then alternatives
    if "TickVolume" not in df.columns:
        if "Tickvolume" in df.columns:
            df["TickVolume"] = df["Tickvolume"]
        elif "Volume" in df.columns:
            df["TickVolume"] = df["Volume"]
        else:
            # If there is no volume column at all, create a dummy one
            df["TickVolume"] = 0

    df["VOL_PCT"] = df["TickVolume"].rolling(500).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) if len(x) > 10 else np.nan
    )

    # Advanced features computed on rolling windows
    hurst_vals = []
    adf_vals = []
    acf1_vals = []

    # If running a quick test, skip heavy ADF/ACF computations to speed up startup.
    quick_mode = os.getenv("RL_QUICK_TEST") == "1"
    if quick_mode:
        # Provide neutral placeholder values so classification can proceed.
        hurst_vals = [0.5] * len(df)
        adf_vals = [0.5] * len(df)
        acf1_vals = [0.0] * len(df)
    else:
        for i in range(len(df)):
            window = df["Close"].iloc[max(0, i - 300):i]

            if len(window) < 100:
                hurst_vals.append(np.nan)
                adf_vals.append(np.nan)
                acf1_vals.append(np.nan)
                continue

            hurst_vals.append(hurst_exponent(window))
            adf_vals.append(adf_pvalue(window))
            acf1_vals.append(acf1(window))

    df["HURST"] = hurst_vals
    df["ADF_P"] = adf_vals
    df["ACF1"] = acf1_vals

    # Rolling means used in classification (precompute to allow row-level comparison)
    df["ROLL_STD_RET_MEAN_500"] = df["ROLL_STD_RET"].rolling(500).mean()
    df["BB_WIDTH_MEAN_200"] = df["BB_WIDTH"].rolling(200).mean()

    return df


# ============================================================
#  REGIME CLASSIFICATION
# ============================================================

def classify_regime(row: pd.Series, atr_high: float, atr_extreme: float) -> int:
    """
    Classifies a single bar into a regime:
    0 = Mean Reversion
    1 = Trend
    2 = Sideways / Noise
    3 = Shock / Extreme (Safe Mode)
    """

    atr_val = row["ATR"]
    hurst = row["HURST"]
    adf_p = row["ADF_P"]
    acf1_val = row["ACF1"]

    # Not enough data → Noise
    if any(np.isnan([atr_val, hurst, adf_p, acf1_val])):
        return 2

    # Shock / Extreme volatility → Safe Mode
    if atr_val > atr_extreme:
        return 3

    # ============================
    # Mean Reversion regime
    # ============================
    if (
        adf_p < 0.05 and
        hurst < 0.45 and
        acf1_val < 0 and
        not np.isnan(row.get("BB_WIDTH_MEAN_200", np.nan)) and
        row["BB_WIDTH"] < row.get("BB_WIDTH_MEAN_200", np.nan)
    ):
        return 0

    # ============================
    # Trend regime
    # ============================
    if (
        adf_p > 0.1 and
        hurst > 0.55 and
        acf1_val > 0 and
        not np.isnan(row.get("ROLL_STD_RET_MEAN_500", np.nan)) and
        row["ROLL_STD_RET"] > row.get("ROLL_STD_RET_MEAN_500", np.nan) and
        not np.isnan(row.get("BB_WIDTH_MEAN_200", np.nan)) and
        row["BB_WIDTH"] > row.get("BB_WIDTH_MEAN_200", np.nan)
    ):
        return 1

    # Sideways / Noise
    return 2


# ============================================================
#  MAIN FUNCTION: ADDS THE "REGIME" COLUMN
# ============================================================

def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_regime_features(df)

    # Causal ATR thresholds: history up to t-1 only (no leakage from the future)
    atr_high_series = (
        pd.to_numeric(df["ATR"], errors="coerce")
        .expanding(min_periods=50)
        .quantile(0.70)
        .shift(1)
    )
    atr_extreme_series = (
        pd.to_numeric(df["ATR"], errors="coerce")
        .expanding(min_periods=50)
        .quantile(0.95)
        .shift(1)
    )

    regimes = []
    for i, row in df.iterrows():
        atr_high = float(atr_high_series.iloc[i]) if not np.isnan(atr_high_series.iloc[i]) else float("inf")
        atr_extreme = float(atr_extreme_series.iloc[i]) if not np.isnan(atr_extreme_series.iloc[i]) else float("inf")
        regimes.append(classify_regime(row, atr_high, atr_extreme))

    df["REGIME"] = regimes
    return df
