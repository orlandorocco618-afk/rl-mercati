from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class SRExposureScaler:
    def __init__(self, summary_path: Path):
        self.summary_path = summary_path
        self.month_weights: dict[str, float] = {}
        self.regime_weights: dict[str, float] = {}
        self.danger_weights: dict[str, float] = {}
        self.bias_flags: list[str] = []
        self.loaded = False
        self._load_summary()

    def _load_summary(self) -> None:
        if not self.summary_path.exists():
            return
        try:
            with self.summary_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return
        risk_plan = data.get("risk_plan", {})
        self.month_weights = {
            str(k): float(v) for k, v in risk_plan.get("position_scale_by_month", {}).items()
        }
        self.regime_weights = {
            str(k): float(v) for k, v in risk_plan.get("regime_weighting", {}).items()
        }
        self.danger_weights = {
            str(k): float(v) for k, v in risk_plan.get("danger_weighting", {}).items()
        }
        self.bias_flags = list(risk_plan.get("bias_flags", []))
        self.loaded = True

    def get_scale(self, timestamp: Any, regime: Any, danger: Any) -> float:
        if not self.loaded:
            return 1.0
        month_key = self._month_key(timestamp)
        scale = 1.0
        scale *= self.month_weights.get(month_key, 1.0)
        scale *= self.regime_weights.get(str(regime), 1.0)
        scale *= self.danger_weights.get(str(danger), 1.0)
        if "directional_imbalance_long_or_short_missing" in self.bias_flags:
            scale *= 0.95
        if "very_long_hold_streak" in self.bias_flags:
            scale *= 0.9
        return clamp(scale, 0.35, 1.2)

    @staticmethod
    def _month_key(timestamp: Any) -> str:
        if timestamp is None:
            return ""
        try:
            dt = pd.to_datetime(timestamp)
        except Exception:
            return ""
        if np.isnan(dt.value):
            return ""
        return dt.strftime("%Y-%m")
