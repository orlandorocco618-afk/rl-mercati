import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf


# ============================================================
#  CLASSIC TECHNICAL INDICATORS
# ============================================================

def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Simple Moving Average.
    """
    return series.rolling(window).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range.
    df must contain the columns: High, Low, Close.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()

    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Z-score relative to the moving average.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-9)


# ============================================================
#  ADVANCED STATISTICAL FEATURES
# ============================================================

def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Computes the Hurst exponent.
    Values:
        < 0.45 → mean reversion
        ~ 0.5 → random walk
        > 0.55 → trend
    """
    series = series.dropna()
    if len(series) < max_lag + 5:
        return np.nan

    lags = np.arange(min_lag, max_lag)
    tau = []

    for lag in lags:
        diff = series.diff(lag).dropna()
        tau.append(np.sqrt(np.std(diff)))

    tau = np.array(tau)
    lags = np.array(lags)

    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = reg[0] * 2.0
        return hurst
    except Exception:
        return np.nan


def adf_pvalue(series: pd.Series) -> float:
    """
    Returns the p-value of the ADF test.
    p < 0.05 → stationary series (MR)
    """
    series = series.dropna()
    if len(series) < 30:
        return np.nan

    try:
        result = adfuller(series, autolag='AIC')
        return result[1]
    except Exception:
        return np.nan


def acf1(series: pd.Series) -> float:
    """
    Lag-1 autocorrelation.
    > 0 → trend
    < 0 → mean reversion
    """
    series = series.dropna()
    if len(series) < 10:
        return np.nan

    try:
        return acf(series, nlags=1, fft=False)[1]
    except Exception:
        return np.nan


# ============================================================
#  SUPPORTING FEATURES
# ============================================================

def log_returns(series: pd.Series) -> pd.Series:
    """
    Log returns: more stable than percentage returns.
    """
    return np.log(series / series.shift())


def rolling_volatility(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Rolling volatility of the returns.
    """
    return series.rolling(window).std()
