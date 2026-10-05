"""Utility routines to detect support/resistance levels.

The intention is to run this once on the historical price series used by the
environment and keep a small list of zones that have been ``touched`` multiple
times.

The detection algorithm is intentionally simple:

* scan the price series looking for local peaks (resistance) and local troughs
  (support) over a small sliding window;
* when a new extremum is found, cluster it with an existing zone if the price is
  within ``tol`` percent of an existing level; otherwise create a new zone
  entry;
* each zone records the ``price`` level, the number of ``touches`` and the
  last index where it was touched.  That allows the env to filter only zones
  that existed prior to the current step.

At step time the env will query the two lists and compute simple features such
as distance from the nearest zone and the zone's strength.

These routines are lightweight and suitable for running inside the RL data
pipeline.  More sophisticated detection (regression lines, volume clusters,
order-book levels) could replace this later.
"""

from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


Zone = Dict[str, float]  # {'price':.., 'touches':.., 'last_touch':..}


def _add_to_zones(zones: List[Zone], price: float, idx: int, tol: float) -> None:
    """Add a price observation to an existing zone or create a new one."""
    for z in zones:
        if abs(price - z['price']) / (z['price'] + 1e-9) < tol:
            z['touches'] += 1
            z['last_touch'] = idx
            # optionally update canonical price as running mean
            z['price'] = (z['price'] * (z['touches'] - 1) + price) / z['touches']
            return
    # no close zone found
    zones.append({'price': price, 'touches': 1, 'last_touch': idx})


def compute_support_resistance(
    df: pd.DataFrame,
    price_col: str = 'Close',
    lookback: int = 5,
    tol: float = 0.002,
) -> Tuple[List[Zone], List[Zone]]:
    """Scan the price column and return (support_zones, resistance_zones).

    ``lookback`` specifies how many bars on each side to examine for a local
    extremum; ``tol`` is the fractional tolerance when clustering levels.
    """
    prices = pd.to_numeric(df[price_col], errors='coerce').values
    n = len(prices)
    support: List[Zone] = []
    resistance: List[Zone] = []

    # simple local extrema detection
    for i in range(lookback, n - lookback):
        window = prices[i - lookback : i + lookback + 1]
        if np.isnan(prices[i]):
            continue
        if prices[i] == np.max(window):
            _add_to_zones(resistance, prices[i], i, tol)
        if prices[i] == np.min(window):
            _add_to_zones(support, prices[i], i, tol)

    return support, resistance


# convenience wrapper that accepts a series and returns sorted lists

def detect_zones_from_series(
    s: pd.Series, **kwargs
) -> Tuple[List[Zone], List[Zone]]:
    df = pd.DataFrame({s.name: s})
    return compute_support_resistance(df, price_col=s.name, **kwargs)
