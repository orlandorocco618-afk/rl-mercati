import numpy as np
import pandas as pd


class KalmanFilter1D:
    """
    One-dimensional Kalman filter to smooth price series.
    Model:
        x_k = x_{k-1} + w
        z_k = x_k + v
    Where:
        w ~ N(0, Q) = process noise
        v ~ N(0, R) = measurement noise
    """

    def __init__(self, process_var=1e-3, measurement_var=1e-2):
        self.Q = process_var
        self.R = measurement_var
        self.x = None
        self.P = None

    def filter(self, z):
        """
        Filters a single observed value z.
        """
        # Handle missing observation robustly: carry forward last valid state.
        if np.isnan(z):
            if self.x is None or np.isnan(self.x):
                return np.nan
            return float(self.x)

        # (Re)initialize if state is not valid.
        if self.x is None or self.P is None or np.isnan(self.x) or np.isnan(self.P):
            self.x = float(z)
            self.P = 1.0
            return float(z)

        # Prediction
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1 - K) * P_pred

        if np.isnan(self.x):
            # Fallback defensive: never propagate NaN state.
            self.x = float(z)
            self.P = 1.0
            return float(z)

        return float(self.x)


def apply_kalman_filter(df: pd.DataFrame, price_col="Close") -> pd.DataFrame:
    """
    Applies the Kalman filter to the price column.
    Adds a column: Close_KF
    """
    kf = KalmanFilter1D()

    filtered = []
    last_valid = np.nan
    for price in df[price_col]:
        try:
            z = float(price)
        except Exception:
            # If conversion fails, use NaN (kf.filter will initialize with NaN)
            try:
                z = float(str(price).replace(',', ''))
            except Exception:
                z = np.nan

        x = kf.filter(z)
        if np.isnan(x):
            # Keep continuity if filter still has no valid state.
            x = last_valid
        else:
            last_valid = x
        filtered.append(x)

    df = df.copy()
    df["Close_KF"] = filtered
    return df
