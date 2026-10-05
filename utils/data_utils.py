import os
import pandas as pd
import numpy as np

from .regimes import add_regimes
from .indicators import atr, log_returns, rolling_volatility


# ============================================================
#  CSV LOADING AND NORMALIZATION
# ============================================================

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads a CSV and normalizes the columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.capitalize() for c in df.columns]

    if "Tickvolume" in df.columns:
        df.rename(columns={"Tickvolume": "TickVolume"}, inplace=True)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Make sure Time is datetime. Some CSVs use 'Date' as the column name.
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
    elif "Date" in df.columns:
        # Normalize: create Time from Date when Time is not present
        df["Time"] = pd.to_datetime(df["Date"])

    # Coerce common numeric columns to numeric types (robust to strings/comma separators)
    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "TickVolume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    return df


# ============================================================
#  MERGE H1 + D1 WITH REGIMES
# ============================================================

def merge_h1_with_d1_regimes(
    h1_path: str = "data/xauusd_h1_clean.csv",
    d1_path: str = "data/xauusd_d1_clean.csv"
) -> pd.DataFrame:
    """
    Loads H1 and D1, computes the regimes on D1, and assigns them to every H1 bar.
    
    Procedure:
    1. Load H1 and D1
    2. Compute regimes on D1
    3. Extract the date (without the time) from every H1 bar
    4. Assign the D1 regime to all H1 bars of the same date
    
    Returns:
        H1 DataFrame with a REGIME column taken from D1
    """
    print("Loading H1...")
    df_h1 = load_csv(h1_path)
    
    print("Loading D1...")
    df_d1 = load_csv(d1_path)
    
    print("Computing regimes on D1...")
    df_d1 = add_regimes(df_d1)
    
    # Extract the date from Time (without hours/minutes)
    df_h1["Date"] = df_h1["Time"].dt.date
    df_d1["Date"] = df_d1["Time"].dt.date

    # Causal shift: every H1 bar of day D only uses D1 info available at the end of D-1.
    regime_cols = ["REGIME", "HURST", "ADF_P", "ACF1", "ROLL_STD_RET", "ATR"]
    for col in regime_cols:
        if col not in df_d1.columns:
            df_d1[col] = np.nan
    df_d1 = df_d1.sort_values("Date").reset_index(drop=True)
    df_d1[regime_cols] = df_d1[regime_cols].shift(1)
    
    # Merge on Date: every H1 bar gets the regime of its D1 date
    df_h1 = df_h1.merge(
        df_d1[["Date", "REGIME", "HURST", "ADF_P", "ACF1", "ROLL_STD_RET", "ATR"]],
        on="Date",
        how="left"
    )
    
    # Fill any NaN with regime 2 (Noise)
    df_h1["REGIME"] = df_h1["REGIME"].fillna(2).astype(int)

    # H1 trading features (causal) used by the env; they don't overwrite the D1 regime.
    if "TickVolume" not in df_h1.columns:
        if "Volume" in df_h1.columns:
            df_h1["TickVolume"] = pd.to_numeric(df_h1["Volume"], errors="coerce")
        else:
            df_h1["TickVolume"] = 0.0
    df_h1["RET"] = log_returns(pd.to_numeric(df_h1["Close"], errors="coerce"))
    df_h1["ATR_H1"] = atr(df_h1, window=14)
    if "ATR" not in df_h1.columns:
        df_h1["ATR"] = df_h1["ATR_H1"]
    else:
        df_h1["ATR"] = pd.to_numeric(df_h1["ATR"], errors="coerce").fillna(df_h1["ATR_H1"])
    df_h1["ROLL_STD_RET_H1"] = rolling_volatility(df_h1["RET"], window=100)
    if "ROLL_STD_RET" not in df_h1.columns:
        df_h1["ROLL_STD_RET"] = df_h1["ROLL_STD_RET_H1"]
    else:
        df_h1["ROLL_STD_RET"] = pd.to_numeric(df_h1["ROLL_STD_RET"], errors="coerce").fillna(df_h1["ROLL_STD_RET_H1"])
    
    # Drop the temporary Date column
    df_h1.drop(columns=["Date"], inplace=True)
    
    print(f"Dataset merged: {len(df_h1)} H1 rows with D1 regimes")
    
    return df_h1


# ============================================================
#  SIMPLE H1 LOADING (no regimes, normalization only)
# ============================================================

def load_h1_simple(h1_path: str = "data/xauusd_h1_clean.csv") -> pd.DataFrame:
    """
    Loads and normalizes H1 only (without computing regimes).
    Used where regime detection is not needed.
    """
    return load_csv(h1_path)
