import gymnasium as gym
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Iterable

from gymnasium import spaces
from utils.indicators import zscore, rsi
from utils.regimes import add_regimes
from utils.sr_exposure_scaler import SRExposureScaler
from utils.support_zones import compute_support_resistance

SR_EXPOSURE_SUMMARY = Path("backtest/plots_oos/br_sr_comparison/sr_final_analysis/sr_optimization_summary.json")


class TradingEnv(gym.Env):
    """
    RL env for XAUUSD H1 with causal execution.
    Decision at t (closed candle), execution at t+1 open.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 100,
        initial_capital: float = 10000.0,
        max_drawdown: float = 0.30,
        commission: float = 0.0002,
        spread_bps: float = 2.0,
        slippage_bps: float = 1.0,
        slippage_atr_coef: float = 0.0,
        execution_delay_bars: int = 1,
        reverse_action_penalty_coef: float = 0.002,
        allow_hold_when_flat: bool = True,
        base_position_size: float = 0.1,
        sizing_shrink: float = 0.9,
        enable_regime_sizing: bool = True,
        enable_dynamic_sizing: bool = False,
        dynamic_reference_window: int = 96,
        dynamic_size_floor: float = 0.65,
        dynamic_size_ceiling: float = 1.35,
        dynamic_confidence_power: float = 1.15,
        dynamic_volatility_weight: float = 0.45,
        dynamic_signal_weight: float = 0.20,
        dynamic_zone_weight: float = 0.20,
        dynamic_breakout_weight: float = 0.15,
        enable_confidence_gate: bool = False,
        confidence_gate_threshold: float = 0.58,
        confidence_gate_flatten_on_reverse: bool = True,
        regime_hard_stop: bool = False,
        hard_stop_danger_level: int = 2,
        enforce_directional_rebalance: bool = False,
        rebalance_after_entries: int = 50,
        rebalance_dominance_threshold: float = 0.75,
        directional_balance_reward_coef: float = 0.0,
        min_trade_duration_hours: int = 4,
        preferred_max_trade_duration_hours: int = 24,
        hard_max_trade_duration_hours: int = 48,
        trade_duration_bonus_coef: float = 0.0015,
        overhold_penalty_coef: float = 0.0040,
        overhold_step_penalty_coef: float = 0.00035,
        hold_penalty_base: float = 0.000008,
        hold_penalty_growth: float = 0.045,
        hold_penalty_cap: float = 0.0015,
        hold_penalty_only_when_flat: bool = False,
        hold_in_position_grace_hours: int = 10,
        hold_in_position_penalty_coef: float = 0.00035,
        hold_in_position_penalty_growth: float = 0.045,
        hold_in_position_penalty_cap: float = 0.006,
        hold_missed_move_coef: float = 0.8,
        hold_missed_move_atr_floor: float = 0.5,
        target_active_steps_pct: float = 0.15,
        target_active_steps_tolerance: float = 0.03,
        active_target_reward_coef: float = 0.004,
        activity_hold_penalty_coef: float = 0.004,
        entry_quality_coef: float = 0.0,
        entry_quality_threshold: float = 0.15,
        entry_quality_noise_scale: float = 0.5,
        small_close_pnl_threshold: float = 0.00008,
        small_close_pnl_penalty_coef: float = 0.0,
        good_close_pnl_threshold: float = 0.00025,
        good_close_bonus_coef: float = 0.0,
        breakout_lookbacks: Iterable[int] | None = None,
        breakout_alignment_bonus_coef: float = 0.0008,
        breakout_misaligned_penalty_coef: float = 0.0009,
        breakout_strength_bonus_coef: float = 0.00035,
        breakout_selector_bonus_coef: float = 0.00025,
        breakout_entry_strength_quantile: float = 0.65,
        breakout_strength_quantile_window: int = 1500,
        breakout_block_misaligned_entries: bool = True,
        breakout_block_weak_entries: bool = True,
        atr_stop_multiple: float = 2.5,
        hard_time_stop_hours: int = 36,
        reward_shaping_weight: float = 0.12,
        turnover_penalty_coef: float = 0.00035,
        drawdown_penalty_coef: float = 0.0,
        drawdown_penalty_threshold: float = 0.03,
        low_confidence_open_penalty_coef: float = 0.0,
        high_confidence_open_bonus_coef: float = 0.0,
        confidence_reward_threshold: float = 0.58,
        short_gap_penalty_coef: float = 0.0,
        short_gap_threshold: int = 64,
        short_reward_bonus_coef: float = 0.0,
        long_reward_bonus_coef: float = 0.0,
        breakout_prior_n: int = 96,
        breakout_prior_bonus_coef: float = 0.00025,
        random_start_on_reset: bool = False,
        min_episode_steps: int = 1500,
        reward_clip_abs: float = 0.02,
        equity_from_reward: bool = False,
        recompute_regimes: bool = False,
        # support/resistance zone options (can also be set via RL_* env vars)
        enable_zones: bool | int | None = None,
        zone_tol: float | None = None,
        zone_lookback: int | None = None,
    ):
        super().__init__()

        self.df_raw = df.reset_index(drop=True).copy()
        if recompute_regimes or "REGIME" not in self.df_raw.columns:
            self.df = add_regimes(self.df_raw)
        else:
            self.df = self.df_raw.copy()

        self._ensure_required_columns()
        self.exposure_scaler = SRExposureScaler(SR_EXPOSURE_SUMMARY)

        self.window_size = int(window_size)
        self.initial_capital = float(initial_capital)
        self.max_drawdown = float(max_drawdown)
        self.commission = max(0.0, float(commission))
        self.spread_bps = max(0.0, float(spread_bps))
        self.slippage_bps = max(0.0, float(slippage_bps))
        self.slippage_atr_coef = max(0.0, float(slippage_atr_coef))
        self.execution_delay_bars = max(1, int(execution_delay_bars))
        self.reverse_action_penalty_coef = max(0.0, float(reverse_action_penalty_coef))

        self.position = 0
        self.entry_price = None
        self.position_size = 0.0
        self.position_size_at_entry = 0.0
        self.entry_atr_norm = 0.0
        self.equity = self.initial_capital
        self.steps_since_short = 0
        self.steps_since_short = 0
        self.allow_hold_when_flat = bool(allow_hold_when_flat)

        self.base_position_size = float(base_position_size)
        self.sizing_shrink = float(sizing_shrink)
        self.enable_regime_sizing = bool(enable_regime_sizing)
        self.enable_dynamic_sizing = bool(enable_dynamic_sizing)
        self.dynamic_reference_window = max(16, int(dynamic_reference_window))
        self.dynamic_size_floor = float(np.clip(float(dynamic_size_floor), 0.05, 2.0))
        self.dynamic_size_ceiling = max(self.dynamic_size_floor, float(np.clip(float(dynamic_size_ceiling), 0.05, 3.0)))
        self.dynamic_confidence_power = max(0.25, float(dynamic_confidence_power))
        self.dynamic_volatility_weight = max(0.0, float(dynamic_volatility_weight))
        self.dynamic_signal_weight = max(0.0, float(dynamic_signal_weight))
        self.dynamic_zone_weight = max(0.0, float(dynamic_zone_weight))
        self.dynamic_breakout_weight = max(0.0, float(dynamic_breakout_weight))
        self.enable_confidence_gate = bool(enable_confidence_gate)
        self.confidence_gate_threshold = float(np.clip(float(confidence_gate_threshold), 0.0, 1.0))
        self.confidence_gate_flatten_on_reverse = bool(confidence_gate_flatten_on_reverse)
        self.regime_hard_stop = bool(regime_hard_stop)
        self.hard_stop_danger_level = int(hard_stop_danger_level)
        self.enforce_directional_rebalance = bool(enforce_directional_rebalance)
        self.rebalance_after_entries = int(rebalance_after_entries)
        self.rebalance_dominance_threshold = float(rebalance_dominance_threshold)
        self.directional_balance_reward_coef = float(directional_balance_reward_coef)

        self.min_trade_duration_hours = max(1, int(min_trade_duration_hours))
        self.preferred_max_trade_duration_hours = max(self.min_trade_duration_hours, int(preferred_max_trade_duration_hours))
        self.hard_max_trade_duration_hours = max(self.preferred_max_trade_duration_hours, int(hard_max_trade_duration_hours))
        self.trade_duration_bonus_coef = float(trade_duration_bonus_coef)
        self.overhold_penalty_coef = float(overhold_penalty_coef)
        self.overhold_step_penalty_coef = float(overhold_step_penalty_coef)

        self.hold_penalty_base = max(0.0, float(hold_penalty_base))
        self.hold_penalty_growth = max(0.0, float(hold_penalty_growth))
        self.hold_penalty_cap = max(0.0, float(hold_penalty_cap))
        self.hold_penalty_only_when_flat = bool(hold_penalty_only_when_flat)
        self.hold_in_position_grace_hours = max(1, int(hold_in_position_grace_hours))
        self.hold_in_position_penalty_coef = max(0.0, float(hold_in_position_penalty_coef))
        self.hold_in_position_penalty_growth = max(0.0, float(hold_in_position_penalty_growth))
        self.hold_in_position_penalty_cap = max(0.0, float(hold_in_position_penalty_cap))
        self.hold_missed_move_coef = max(0.0, float(hold_missed_move_coef))
        self.hold_missed_move_atr_floor = max(0.0, float(hold_missed_move_atr_floor))

        self.target_active_steps_pct = float(np.clip(float(target_active_steps_pct), 0.001, 0.999))
        self.target_active_steps_tolerance = float(np.clip(float(target_active_steps_tolerance), 0.001, 0.25))
        self.active_target_reward_coef = max(0.0, float(active_target_reward_coef))
        self.activity_hold_penalty_coef = max(0.0, float(activity_hold_penalty_coef))
        self.entry_quality_coef = max(0.0, float(entry_quality_coef))
        self.entry_quality_threshold = float(np.clip(float(entry_quality_threshold), -1.0, 1.0))
        self.entry_quality_noise_scale = float(np.clip(float(entry_quality_noise_scale), 0.0, 1.0))

        self.small_close_pnl_threshold = max(0.0, float(small_close_pnl_threshold))
        self.small_close_pnl_penalty_coef = max(0.0, float(small_close_pnl_penalty_coef))
        self.good_close_pnl_threshold = max(0.0, float(good_close_pnl_threshold))
        self.good_close_bonus_coef = max(0.0, float(good_close_bonus_coef))
        self.breakout_lookbacks = self._parse_breakout_lookbacks(breakout_lookbacks)
        self.breakout_enabled = len(self.breakout_lookbacks) > 0

        # support/resistance zones (features for the agent)
        # parameters may come from function arguments or fall back to env vars
        if enable_zones is None:
            self.enable_zones = bool(int(os.getenv('RL_ENABLE_ZONES', '1')))
        else:
            self.enable_zones = bool(enable_zones)
        if zone_tol is None:
            self.zone_tol = float(os.getenv('RL_ZONE_TOL', '0.002'))
        else:
            self.zone_tol = float(zone_tol)
        if zone_lookback is None:
            self.zone_lookback = int(os.getenv('RL_ZONE_LOOKBACK', '5'))
        else:
            self.zone_lookback = int(zone_lookback)
        if self.enable_zones:
            # compute on full dataset once at initialization
            sup, res = compute_support_resistance(
                self.df, price_col='Close',
                lookback=self.zone_lookback, tol=self.zone_tol,
            )
            self.support_zones = sup
            self.resistance_zones = res
        else:
            self.support_zones, self.resistance_zones = [], []
        self.breakout_alignment_bonus_coef = max(0.0, float(breakout_alignment_bonus_coef))
        self.breakout_misaligned_penalty_coef = max(0.0, float(breakout_misaligned_penalty_coef))
        self.breakout_strength_bonus_coef = max(0.0, float(breakout_strength_bonus_coef))
        self.breakout_selector_bonus_coef = max(0.0, float(breakout_selector_bonus_coef))
        self.breakout_entry_strength_quantile = float(np.clip(float(breakout_entry_strength_quantile), 0.50, 0.95))
        self.breakout_strength_quantile_window = max(200, int(breakout_strength_quantile_window))
        self.breakout_block_misaligned_entries = bool(breakout_block_misaligned_entries)
        self.breakout_block_weak_entries = bool(breakout_block_weak_entries)
        self.atr_stop_multiple = max(0.2, float(atr_stop_multiple))
        self.hard_time_stop_hours = max(4, int(hard_time_stop_hours))
        self.reward_shaping_weight = float(np.clip(float(reward_shaping_weight), 0.0, 1.0))
        self.turnover_penalty_coef = max(0.0, float(turnover_penalty_coef))
        self.drawdown_penalty_coef = max(0.0, float(drawdown_penalty_coef))
        self.drawdown_penalty_threshold = float(np.clip(float(drawdown_penalty_threshold), 0.0, 0.95))
        self.low_confidence_open_penalty_coef = max(0.0, float(low_confidence_open_penalty_coef))
        self.high_confidence_open_bonus_coef = max(0.0, float(high_confidence_open_bonus_coef))
        self.confidence_reward_threshold = float(np.clip(float(confidence_reward_threshold), 0.0, 1.0))
        self.short_gap_penalty_coef = max(0.0, float(short_gap_penalty_coef))
        self.short_gap_threshold = max(1, int(short_gap_threshold))
        self.short_reward_bonus_coef = max(0.0, float(short_reward_bonus_coef))
        self.long_reward_bonus_coef = max(0.0, float(long_reward_bonus_coef))
        self.breakout_prior_n = int(breakout_prior_n)
        self.breakout_prior_bonus_coef = max(0.0, float(breakout_prior_bonus_coef))
        self.last_selected_breakout_idx = 0
        self.last_selected_breakout_n = int(self.breakout_lookbacks[0]) if self.breakout_enabled else 0
        self._prepare_dynamic_sizing_features()
        self._prepare_breakout_columns()

        self.random_start_on_reset = bool(random_start_on_reset)
        self.min_episode_steps = max(10, int(min_episode_steps))
        self.reward_clip_abs = max(0.0, float(reward_clip_abs))
        self.equity_from_reward = bool(equity_from_reward)

        self.base_obs_dim = 15
        # zone features (dist/support strength, dist/res strength)
        self.zone_obs_dim = 4 if self.enable_zones else 0
        self.breakout_obs_dim = (2 * len(self.breakout_lookbacks) + 2) if self.breakout_enabled else 0
        self.observation_space = spaces.Box(
            low=-5,
            high=5,
            shape=(self.base_obs_dim + self.zone_obs_dim + self.breakout_obs_dim,),
            dtype=np.float32,
        )
        if self.breakout_enabled:
            self.action_space = spaces.MultiDiscrete(np.array([4, len(self.breakout_lookbacks)], dtype=np.int64))
        else:
            self.action_space = spaces.Discrete(4)

        self.current_step = self.window_size
        self.long_entries = 0
        self.short_entries = 0
        self.position_entry_step = None
        self.entry_atr_norm = 0.0
        self.max_danger_during_trade = 0
        self.peak_equity = self.initial_capital
        self.consecutive_hold_steps = 0
        self.episode_step_count = 0
        self.episode_active_steps = 0
        self.episode_invested_steps = 0

    def _ensure_required_columns(self):
        for col in ["Open", "High", "Low", "Close"]:
            if col not in self.df.columns:
                raise ValueError(f"Required column missing: {col}")
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if "TickVolume" not in self.df.columns:
            if "Volume" in self.df.columns:
                self.df["TickVolume"] = pd.to_numeric(self.df["Volume"], errors="coerce")
            else:
                self.df["TickVolume"] = 0.0

        if "RET" not in self.df.columns:
            close = pd.to_numeric(self.df["Close"], errors="coerce")
            self.df["RET"] = np.log(close / close.shift(1))

        if "ATR" not in self.df.columns:
            high = pd.to_numeric(self.df["High"], errors="coerce")
            low = pd.to_numeric(self.df["Low"], errors="coerce")
            close = pd.to_numeric(self.df["Close"], errors="coerce")
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.df["ATR"] = tr.rolling(14).mean()

        if "ROLL_STD_RET" not in self.df.columns:
            self.df["ROLL_STD_RET"] = pd.to_numeric(self.df["RET"], errors="coerce").rolling(100).std()

        for col, fill in [("HURST", np.nan), ("ADF_P", np.nan), ("ACF1", np.nan)]:
            if col not in self.df.columns:
                self.df[col] = fill

        if "REGIME" not in self.df.columns:
            self.df["REGIME"] = 2

        self.df["REGIME"] = pd.to_numeric(self.df["REGIME"], errors="coerce").fillna(2).astype(int)

    @staticmethod
    def _parse_breakout_lookbacks(raw_lookbacks: Iterable[int] | None) -> tuple[int, ...]:
        if raw_lookbacks is None:
            return tuple()
        parsed: list[int] = []
        for value in raw_lookbacks:
            try:
                v = int(value)
            except Exception:
                continue
            if v >= 4:
                parsed.append(v)
        if not parsed:
            return tuple()
        return tuple(sorted(set(parsed)))

    def _zone_features(self, step_idx: int):
        """Return 4-zone features at given step.

        Four floats: distance and strength for nearest support, distance and
        strength for nearest resistance (normalized to price).  If no zones
        exist yet the values are zero.
        """
        price = float(self.df.iloc[step_idx]["Close"])

        def nearest(zones):
            best_dist = 1.0
            best_str = 0.0
            for z in zones:
                if z["last_touch"] >= step_idx:
                    continue
                d = abs(price - z["price"]) / (z["price"] + 1e-9)
                if d < best_dist:
                    best_dist = d
                    best_str = float(z.get("touches", 0))
            return best_dist, best_str

        dsup, ssup = nearest(self.support_zones)
        dres, sres = nearest(self.resistance_zones)
        return np.array([dsup, dres, ssup, sres], dtype=np.float32)

    def _prepare_breakout_columns(self):
        if not self.breakout_enabled:
            return
        close = pd.to_numeric(self.df["Close"], errors="coerce")
        atr_key = "ATR_H1" if "ATR_H1" in self.df.columns else "ATR"
        atr = pd.to_numeric(self.df.get(atr_key, 0.0), errors="coerce").fillna(0.0)
        atr_norm = (atr / (close.abs() + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        for lb in self.breakout_lookbacks:
            hh = close.rolling(lb).max().shift(1)
            ll = close.rolling(lb).min().shift(1)
            sig = np.where(close > hh, 1, np.where(close < ll, -1, 0))
            up_strength = (close - hh) / (hh.abs() + 1e-9)
            dn_strength = (ll - close) / (ll.abs() + 1e-9)
            raw_strength = np.where(sig > 0, up_strength, np.where(sig < 0, dn_strength, 0.0))
            strength = pd.Series(raw_strength, index=self.df.index)
            norm_strength = strength / (atr_norm + 1e-9)
            abs_strength = pd.to_numeric(norm_strength.abs(), errors="coerce").fillna(0.0)
            qthr = abs_strength.rolling(self.breakout_strength_quantile_window, min_periods=max(200, lb * 2)).quantile(
                self.breakout_entry_strength_quantile
            ).shift(1)
            qthr = qthr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            self.df[f"BRK_SIG_{lb}"] = pd.Series(sig, index=self.df.index).fillna(0).astype(int)
            self.df[f"BRK_STR_{lb}"] = pd.to_numeric(norm_strength, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            self.df[f"BRK_QTHR_{lb}"] = pd.to_numeric(qthr, errors="coerce").fillna(0.0)

    def _prepare_dynamic_sizing_features(self):
        close = pd.to_numeric(self.df["Close"], errors="coerce").abs().replace(0.0, np.nan)
        atr_key = "ATR_H1" if "ATR_H1" in self.df.columns else "ATR"
        atr = pd.to_numeric(self.df.get(atr_key, self.df.get("ATR", 0.0)), errors="coerce")
        atr_norm = (atr / (close + 1e-9)).replace([np.inf, -np.inf], np.nan)
        roll_std = pd.to_numeric(self.df.get("ROLL_STD_RET", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan)
        min_periods = max(12, self.dynamic_reference_window // 4)
        self.df["_ATR_NORM_DYN"] = atr_norm.fillna(0.0)
        atr_fallback = atr_norm.expanding(min_periods=1).median().shift(1)
        roll_fallback = roll_std.expanding(min_periods=1).median().shift(1)
        self.df["_ATR_NORM_MEDIAN_DYN"] = (
            atr_norm.rolling(self.dynamic_reference_window, min_periods=min_periods).median().shift(1).fillna(atr_fallback).fillna(0.0)
        )
        self.df["_ROLL_STD_MEDIAN_DYN"] = (
            roll_std.rolling(self.dynamic_reference_window, min_periods=min_periods).median().shift(1).fillna(roll_fallback).fillna(0.0)
        )

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def _compute_setup_confidence(
        self,
        row: pd.Series,
        regime_signal_score: float,
        atr_norm_now: float,
        roll_std_now: float,
        sup_dist: float,
        sup_strength: float,
        res_dist: float,
        res_strength: float,
        selected_strength_abs: float,
        selected_threshold: float,
        best_breakout_strength: float,
    ) -> tuple[float, dict[str, float]]:
        atr_ref = float(pd.to_numeric(row.get("_ATR_NORM_MEDIAN_DYN", atr_norm_now), errors="coerce"))
        roll_ref = float(pd.to_numeric(row.get("_ROLL_STD_MEDIAN_DYN", roll_std_now), errors="coerce"))
        if not np.isfinite(atr_ref) or atr_ref <= 1e-9:
            atr_ref = max(1e-6, atr_norm_now)
        if not np.isfinite(roll_ref) or roll_ref <= 1e-9:
            roll_ref = max(1e-6, roll_std_now)

        atr_ratio = atr_norm_now / max(1e-6, atr_ref)
        vol_ratio = roll_std_now / max(1e-6, roll_ref)
        volatility_pressure = max(float(atr_ratio), float(vol_ratio))
        volatility_score = self._clip01(1.10 - 0.30 * max(0.0, volatility_pressure - 1.0))

        signal_score = self._clip01(abs(regime_signal_score))

        zone_distance_scale = max(0.0015, float(self.zone_tol) * 4.0)
        support_closeness = self._clip01(1.0 - abs(float(sup_dist)) / zone_distance_scale)
        resistance_closeness = self._clip01(1.0 - abs(float(res_dist)) / zone_distance_scale)
        support_score = support_closeness * self._clip01(float(sup_strength) / 3.0)
        resistance_score = resistance_closeness * self._clip01(float(res_strength) / 3.0)
        zone_score = max(float(support_score), float(resistance_score)) if self.enable_zones else 0.0

        breakout_score = 0.0
        if self.breakout_enabled:
            if selected_threshold > 1e-9:
                breakout_score = self._clip01(selected_strength_abs / max(selected_threshold * 1.35, 1e-9))
            elif best_breakout_strength > 1e-9:
                breakout_score = self._clip01(selected_strength_abs / max(best_breakout_strength, 1e-9))

        weights = {
            "volatility": self.dynamic_volatility_weight,
            "signal": self.dynamic_signal_weight,
            "zone": self.dynamic_zone_weight if self.enable_zones else 0.0,
            "breakout": self.dynamic_breakout_weight if self.breakout_enabled else 0.0,
        }
        weight_sum = sum(weights.values())
        if weight_sum <= 1e-9:
            confidence = 1.0
        else:
            confidence = (
                weights["volatility"] * volatility_score
                + weights["signal"] * signal_score
                + weights["zone"] * zone_score
                + weights["breakout"] * breakout_score
            ) / weight_sum

        components = {
            "volatility_score": float(volatility_score),
            "signal_score": float(signal_score),
            "zone_score": float(zone_score),
            "breakout_score": float(breakout_score),
            "atr_ratio": float(atr_ratio),
            "roll_std_ratio": float(vol_ratio),
        }
        return self._clip01(confidence), components

    def _get_breakout_state(self, step_idx: int, breakout_idx: int):
        if not self.breakout_enabled:
            return {
                "selected_idx": -1,
                "selected_n": 0,
                "selected_signal": 0,
                "selected_strength": 0.0,
                "consensus": 0.0,
                "best_abs_strength": 0.0,
                "signals": [],
                "strengths": [],
            }

        row = self.df.iloc[step_idx]
        signals: list[int] = []
        strengths: list[float] = []
        for lb in self.breakout_lookbacks:
            sig = int(pd.to_numeric(row.get(f"BRK_SIG_{lb}", 0), errors="coerce"))
            strength = float(pd.to_numeric(row.get(f"BRK_STR_{lb}", 0.0), errors="coerce"))
            if not np.isfinite(strength):
                strength = 0.0
            signals.append(int(np.clip(sig, -1, 1)))
            strengths.append(float(strength))

        idx = int(np.clip(int(breakout_idx), 0, len(self.breakout_lookbacks) - 1))
        selected_n = int(self.breakout_lookbacks[idx])
        selected_signal = int(signals[idx])
        selected_strength = float(strengths[idx])
        consensus = float(np.mean(signals)) if signals else 0.0
        best_abs_strength = float(np.max(np.abs(strengths))) if strengths else 0.0
        return {
            "selected_idx": idx,
            "selected_n": selected_n,
            "selected_signal": selected_signal,
            "selected_strength": selected_strength,
            "consensus": consensus,
            "best_abs_strength": best_abs_strength,
            "signals": signals,
            "strengths": strengths,
        }

    def _parse_action(self, action):
        if not self.breakout_enabled:
            try:
                trade_action = int(action)
            except Exception:
                try:
                    trade_action = int(np.asarray(action).reshape(-1)[0])
                except Exception:
                    trade_action = 0
            return int(np.clip(trade_action, 0, 3)), -1

        flat = np.asarray(action).reshape(-1)
        if flat.size >= 2:
            trade_action = int(flat[0])
            breakout_idx = int(flat[1])
        elif flat.size == 1:
            trade_action = int(flat[0])
            breakout_idx = int(self.last_selected_breakout_idx)
        else:
            trade_action = 0
            breakout_idx = int(self.last_selected_breakout_idx)

        trade_action = int(np.clip(trade_action, 0, 3))
        breakout_idx = int(np.clip(breakout_idx, 0, len(self.breakout_lookbacks) - 1))
        return trade_action, breakout_idx

    def _normalize(self, value, ref):
        ref_mean = float(np.nanmean(ref)) if len(ref) > 0 else 0.0
        ref_std = float(np.nanstd(ref)) if len(ref) > 0 else 1.0
        return (float(value) - ref_mean) / (ref_std + 1e-8)

    def _get_zone_features(self, price_val: float, step_idx: int):
        """Return (sup_dist, sup_strength, res_dist, res_strength) for given price and step.

        Distances are signed fractional differences from the nearest zone price.
        Zones with last_touch >= step_idx are ignored (they have not yet formed).
        """
        if not self.enable_zones or price_val is None:
            return 0.0, 0.0, 0.0, 0.0
        sup_dist = 0.0
        sup_strength = 0.0
        res_dist = 0.0
        res_strength = 0.0

        best_d = float("inf")
        for z in self.support_zones:
            if z.get("last_touch", 0) >= step_idx:
                continue
            d = abs(price_val - z["price"]) / max(1e-9, price_val)
            if d < best_d:
                best_d = d
                sup_dist = (price_val - z["price"]) / max(1e-9, price_val)
                sup_strength = float(z.get("touches", 0))
        best_d = float("inf")
        for z in self.resistance_zones:
            if z.get("last_touch", 0) >= step_idx:
                continue
            d = abs(price_val - z["price"]) / max(1e-9, price_val)
            if d < best_d:
                best_d = d
                res_dist = (price_val - z["price"]) / max(1e-9, price_val)
                res_strength = float(z.get("touches", 0))
        return sup_dist, sup_strength, res_dist, res_strength

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]

        price_series = pd.to_numeric(window["Close"], errors="coerce")
        row_close = float(pd.to_numeric(row.get("Close"), errors="coerce"))
        atr_key = "ATR_H1" if "ATR_H1" in row.index else "ATR"
        row_atr = float(pd.to_numeric(row.get(atr_key), errors="coerce"))

        price_norm = self._normalize(row_close, price_series.to_numpy(dtype=float))
        atr_norm = row_atr / max(1e-9, abs(row_close)) if not np.isnan(row_atr) else 0.0

        vol_series = pd.to_numeric(window.get("TickVolume", 0.0), errors="coerce")
        row_vol = float(pd.to_numeric(row.get("TickVolume", 0.0), errors="coerce"))
        vol_norm = self._normalize(row_vol, vol_series.to_numpy(dtype=float))

        hurst = float(pd.to_numeric(row.get("HURST", np.nan), errors="coerce"))
        adf_p = float(pd.to_numeric(row.get("ADF_P", np.nan), errors="coerce"))
        acf1_val = float(pd.to_numeric(row.get("ACF1", np.nan), errors="coerce"))

        if np.isnan(hurst):
            hurst = 0.5
        if np.isnan(adf_p):
            adf_p = 0.5
        if np.isnan(acf1_val):
            acf1_val = 0.0

        activity_ratio = float(self.episode_active_steps) / float(max(1, self.episode_step_count)) if self.episode_step_count > 0 else 0.0
        activity_pressure = float(np.clip((self.target_active_steps_pct - activity_ratio) / max(1e-6, self.target_active_steps_tolerance), -3.0, 3.0))
        hold_streak_norm = float(np.clip(np.log1p(float(self.consecutive_hold_steps)) / np.log1p(96.0), 0.0, 1.0))

        if self.position != 0 and self.position_entry_step is not None:
            position_age_hours = max(0.0, float(self.current_step - int(self.position_entry_step) + 1))
        else:
            position_age_hours = 0.0

        position_age_norm = float(np.clip(position_age_hours / float(max(1, self.preferred_max_trade_duration_hours)), 0.0, 3.0))

        z_val = zscore(price_series).iloc[-1] if len(price_series) > 0 else 0.0
        rsi_val = rsi(price_series).iloc[-1] / 100.0 if len(price_series) > 0 else 0.5

        obs_vals = [
            price_norm,
            z_val,
            rsi_val,
            atr_norm,
            hurst,
            adf_p,
            acf1_val,
            float(pd.to_numeric(row.get("RET", 0.0), errors="coerce")),
            float(pd.to_numeric(row.get("ROLL_STD_RET", 0.0), errors="coerce")),
            vol_norm,
            float(self.position),
            activity_ratio,
            activity_pressure,
            hold_streak_norm,
            position_age_norm,
        ]

        if self.breakout_enabled:
            brk_state = self._get_breakout_state(self.current_step, self.last_selected_breakout_idx)
            for sig, strength in zip(brk_state["signals"], brk_state["strengths"]):
                obs_vals.append(float(sig))
                obs_vals.append(float(np.clip(strength, -5.0, 5.0)))
            obs_vals.append(float(np.clip(brk_state["consensus"], -1.0, 1.0)))
            obs_vals.append(float(np.clip(brk_state["best_abs_strength"], 0.0, 5.0)))

        # zone-based features: distance and strength for nearest support/resistance
        if self.enable_zones:
            sup_dist, sup_strength, res_dist, res_strength = self._get_zone_features(row_close, self.current_step)
            obs_vals.append(float(np.clip(sup_dist, -5.0, 5.0)))
            obs_vals.append(float(np.clip(sup_strength, 0.0, 100.0)))
            obs_vals.append(float(np.clip(res_dist, -5.0, 5.0)))
            obs_vals.append(float(np.clip(res_strength, 0.0, 100.0)))

        obs = np.array(obs_vals, dtype=np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        obs = np.clip(obs, -5.0, 5.0)
        regime = int(pd.to_numeric(row.get("REGIME", 2), errors="coerce"))
        return obs.astype(np.float32), regime
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.position = 0
        self.entry_price = None
        self.position_size = 0.0
        self.position_size_at_entry = 0.0
        self.entry_atr_norm = 0.0
        self.equity = self.initial_capital

        if self.random_start_on_reset and len(self.df) > (self.window_size + self.min_episode_steps + 3):
            max_start = len(self.df) - self.min_episode_steps - 2
            self.current_step = int(self.np_random.integers(self.window_size, max_start))
        else:
            self.current_step = self.window_size

        self.long_entries = 0
        self.short_entries = 0
        self.position_entry_step = None
        self.entry_atr_norm = 0.0
        self.max_danger_during_trade = 0
        self.peak_equity = self.initial_capital
        self.consecutive_hold_steps = 0
        self.episode_step_count = 0
        self.episode_active_steps = 0
        self.episode_invested_steps = 0
        self.last_selected_breakout_idx = 0
        self.last_selected_breakout_n = int(self.breakout_lookbacks[0]) if self.breakout_enabled else 0

        obs, _ = self._get_observation()
        return obs, {}

    def _execution_cost_rate(self, row: pd.Series) -> float:
        close_now = float(pd.to_numeric(row.get("Close"), errors="coerce"))
        atr_now = float(pd.to_numeric(row.get("ATR_H1", row.get("ATR", 0.0)), errors="coerce"))
        atr_norm = atr_now / max(1e-9, abs(close_now)) if not np.isnan(atr_now) else 0.0
        static_cost = (self.spread_bps + self.slippage_bps) / 10000.0
        dynamic_cost = self.slippage_atr_coef * max(0.0, atr_norm)
        return max(0.0, static_cost + dynamic_cost)

    @staticmethod
    def _fill_price(raw_price: float, side: int, cost_rate: float) -> float:
        if side > 0:
            return float(raw_price * (1.0 + cost_rate))
        if side < 0:
            return float(raw_price * (1.0 - cost_rate))
        return float(raw_price)

    def _execution_price(self, exec_idx: int, fallback_close: float) -> float:
        row_exec = self.df.iloc[exec_idx]
        open_px = pd.to_numeric(row_exec.get("Open"), errors="coerce")
        if not pd.isna(open_px):
            return float(open_px)
        close_px = pd.to_numeric(row_exec.get("Close"), errors="coerce")
        if not pd.isna(close_px):
            return float(close_px)
        return float(fallback_close)

    def step(self, action):
        action, breakout_idx = self._parse_action(action)

        if self.current_step >= len(self.df) - 1:
            obs, _ = self._get_observation()
            return obs, 0.0, True, False, {"reason": "end_of_data"}

        obs, regime = self._get_observation()
        row = self.df.iloc[self.current_step]
        state_price = float(pd.to_numeric(row.get("Close"), errors="coerce"))
        # compute zone diagnostics for info
        if self.enable_zones:
            sup_dist, sup_strength, res_dist, res_strength = self._get_zone_features(state_price, self.current_step)
        else:
            sup_dist = sup_strength = res_dist = res_strength = 0.0
        atr_now = float(pd.to_numeric(row.get("ATR_H1", row.get("ATR", 0.0)), errors="coerce"))
        atr_norm_now = atr_now / max(1e-9, abs(state_price)) if np.isfinite(atr_now) else 0.0
        atr_norm_now = max(0.0, float(atr_norm_now))
        breakout_state = self._get_breakout_state(self.current_step, breakout_idx)
        if self.breakout_enabled:
            self.last_selected_breakout_idx = int(breakout_state["selected_idx"])
            self.last_selected_breakout_n = int(breakout_state["selected_n"])

        exec_idx = int(self.current_step + self.execution_delay_bars)
        if exec_idx >= len(self.df):
            exec_idx = len(self.df) - 1

        execution_price_raw = self._execution_price(exec_idx, fallback_close=state_price)
        cost_rate = self._execution_cost_rate(row)
        info = {"regime": regime}

        danger_level = 0
        if "HYBRID_CLASS" in row.index and not pd.isna(row["HYBRID_CLASS"]):
            try:
                danger_level = int(row["HYBRID_CLASS"]) if not np.isnan(row["HYBRID_CLASS"]) else 0
            except Exception:
                danger_level = 0
        elif "STABILITY" in row.index and not pd.isna(row["STABILITY"]):
            try:
                danger_level = int(row["STABILITY"]) if not np.isnan(row["STABILITY"]) else 0
            except Exception:
                danger_level = 0
        elif "REGIME" in row.index and not pd.isna(row["REGIME"]):
            try:
                regime_label = int(row["REGIME"])
                danger_level = 2 if regime_label == 3 else (1 if regime_label == 2 else 0)
            except Exception:
                danger_level = 0

        if self.enable_regime_sizing:
            try:
                self.position_size = float(self.base_position_size * (self.sizing_shrink ** max(0, int(danger_level))))
            except Exception:
                self.position_size = float(self.base_position_size)
        else:
            self.position_size = float(self.base_position_size)
        self.position_size = max(0.0, min(1.0, self.position_size))
        if getattr(self, "exposure_scaler", None) is not None:
            timestamp = row.get("Time") or row.get("time")
            multiplier = self.exposure_scaler.get_scale(timestamp, regime, danger_level)
            self.position_size = float(np.clip(self.position_size * multiplier, 0.0, 1.0))

        ret_now = float(pd.to_numeric(row.get("RET"), errors="coerce"))
        roll_std_now = float(pd.to_numeric(row.get("ROLL_STD_RET"), errors="coerce"))
        if np.isnan(ret_now):
            ret_now = 0.0
        if np.isnan(roll_std_now) or roll_std_now <= 1e-9:
            roll_std_now = max(1e-6, abs(ret_now))
        momentum_score = float(np.clip(ret_now / roll_std_now, -3.0, 3.0) / 3.0)
        if regime == 0:
            regime_signal_score = -momentum_score
        elif regime == 1:
            regime_signal_score = momentum_score
        elif regime == 2:
            regime_signal_score = self.entry_quality_noise_scale * momentum_score
        else:
            regime_signal_score = 0.25 * momentum_score

        base_position_size_pre_dynamic = float(self.position_size)

        info["hard_stop_applied"] = False
        info["forced_action"] = False
        info["forced_reason"] = ""
        info["requested_action"] = int(action)
        info["requested_breakout_idx"] = int(breakout_idx)
        info["selected_breakout_idx"] = int(breakout_state["selected_idx"])
        info["selected_breakout_n"] = int(breakout_state["selected_n"])
        info["selected_breakout_signal"] = int(breakout_state["selected_signal"])
        info["selected_breakout_strength"] = float(breakout_state["selected_strength"])
        info["breakout_consensus"] = float(breakout_state["consensus"])
        info["breakout_best_abs_strength"] = float(breakout_state["best_abs_strength"])
        selected_n = int(breakout_state["selected_n"])
        selected_signal = int(breakout_state["selected_signal"])
        selected_strength = float(breakout_state["selected_strength"])
        selected_strength_abs = abs(selected_strength)
        if selected_n > 0:
            selected_threshold = float(pd.to_numeric(row.get(f"BRK_QTHR_{selected_n}", 0.0), errors="coerce"))
            if not np.isfinite(selected_threshold):
                selected_threshold = 0.0
        else:
            selected_threshold = 0.0
        info["selected_breakout_strength_threshold"] = float(selected_threshold)
        if selected_threshold > 0:
            strength_ok = selected_strength_abs >= selected_threshold
        else:
            strength_ok = selected_strength_abs > 0 or action == 2
        setup_confidence, confidence_components = self._compute_setup_confidence(
            row=row,
            regime_signal_score=regime_signal_score,
            atr_norm_now=atr_norm_now,
            roll_std_now=roll_std_now,
            sup_dist=sup_dist,
            sup_strength=sup_strength,
            res_dist=res_dist,
            res_strength=res_strength,
            selected_strength_abs=selected_strength_abs,
            selected_threshold=selected_threshold,
            best_breakout_strength=float(breakout_state["best_abs_strength"]),
        )
        dynamic_size_multiplier = 1.0
        if self.enable_dynamic_sizing:
            dynamic_size_multiplier = self.dynamic_size_floor + (self.dynamic_size_ceiling - self.dynamic_size_floor) * (
                setup_confidence ** self.dynamic_confidence_power
            )
            self.position_size = float(np.clip(base_position_size_pre_dynamic * dynamic_size_multiplier, 0.0, 1.0))
        info["breakout_entry_strength_ok"] = bool(strength_ok)
        info["breakout_entry_signal_ok"] = bool((action == 1 and selected_signal > 0) or (action == 2 and selected_signal < 0))
        info["breakout_entry_allowed"] = True
        info["invalid_close_blocked"] = False
        info["confidence_gate_blocked"] = False

        if self.enable_confidence_gate and action in [1, 2]:
            opening = (action == 1 and self.position <= 0) or (action == 2 and self.position >= 0)
            low_confidence = setup_confidence < self.confidence_gate_threshold
            if opening and low_confidence:
                reversing = (action == 1 and self.position == -1) or (action == 2 and self.position == 1)
                if reversing and self.confidence_gate_flatten_on_reverse:
                    action = 3
                    info["forced_reason"] = "confidence_gate_close_only"
                else:
                    action = 0
                    info["forced_reason"] = "confidence_gate_block"
                info["forced_action"] = True
                info["confidence_gate_blocked"] = True

        # Hard risk controls: ATR stop and max holding time force close.
        if self.position != 0 and self.entry_price is not None and self.position_entry_step is not None:
            pre_trade_age = max(1, exec_idx - int(self.position_entry_step) + 1)
            if self.position == 1:
                pre_unrealized = ((state_price - self.entry_price) / max(1e-9, self.entry_price)) * self.position_size_at_entry
            else:
                pre_unrealized = ((self.entry_price - state_price) / max(1e-9, self.entry_price)) * self.position_size_at_entry

            stop_ref = max(float(self.entry_atr_norm), float(atr_norm_now), 1e-6)
            atr_stop_threshold = self.atr_stop_multiple * stop_ref
            if pre_unrealized <= -atr_stop_threshold:
                action = 3
                info["forced_action"] = True
                info["forced_reason"] = "atr_stop"
                info["hard_stop_applied"] = True
            elif pre_trade_age >= self.hard_time_stop_hours:
                action = 3
                info["forced_action"] = True
                info["forced_reason"] = "time_stop"
                info["hard_stop_applied"] = True

        # Entry-quality gate: only open/reverse when selected breakout has aligned signal + strong strength.
        if self.breakout_enabled and action in [1, 2]:
            opening = (action == 1 and self.position <= 0) or (action == 2 and self.position >= 0)
            if opening:
                signal_ok = (action == 1 and selected_signal > 0) or (action == 2 and selected_signal < 0)
                strength_ok = (selected_strength_abs >= selected_threshold) if selected_threshold > 0 else (selected_strength_abs > 0)
                entry_allowed = True
                if self.breakout_block_misaligned_entries and not signal_ok:
                    entry_allowed = False
                if self.breakout_block_weak_entries and not strength_ok:
                    entry_allowed = False

                info["breakout_entry_signal_ok"] = bool(signal_ok)
                info["breakout_entry_strength_ok"] = bool(strength_ok)
                info["breakout_entry_allowed"] = bool(entry_allowed)

                if not entry_allowed:
                    if (action == 1 and self.position == -1) or (action == 2 and self.position == 1):
                        action = 3
                        info["forced_reason"] = "entry_filter_close_only"
                    else:
                        action = 0
                        info["forced_reason"] = "entry_filter_block"
                    info["forced_action"] = True

        if action == 3 and self.position == 0:
            action = 0
            info["invalid_close_blocked"] = True

        reward = 0.0
        done = False
        prev_position = int(self.position)
        realized_pnl = 0.0
        short_trade_closed = False
        short_trade_pnl = 0.0
        long_trade_closed = False
        long_trade_pnl = 0.0
        opened_direction = 0
        closed_trade_duration_hours = 0
        closed_trade_extreme = False
        duration_reward_component = 0.0
        hold_penalty_component = 0.0
        hold_in_position_penalty_component = 0.0
        hold_missed_move_component = 0.0
        activity_reward_component = 0.0
        entry_quality_component = 0.0
        close_quality_component = 0.0
        breakout_alignment_component = 0.0
        breakout_selector_component = 0.0
        breakout_prior_component = 0.0
        turnover_penalty_component = 0.0
        confidence_reward_component = 0.0
        drawdown_penalty_component = 0.0
        activity_ratio = 0.0
        activity_closeness = 0.0
        activity_gap = 0.0
        ongoing_hours = 0
        commission_component = 0.0
        if action == 1:
            if self.position <= 0:
                if self.position == -1 and self.entry_price is not None:
                    exit_price = self._fill_price(execution_price_raw, side=1, cost_rate=cost_rate)
                    pnl = ((self.entry_price - exit_price) / max(1e-9, self.entry_price)) * self.position_size_at_entry
                    realized_pnl += pnl
                    short_trade_closed = True
                    short_trade_pnl += pnl
                    commission_component += self.commission * self.position_size_at_entry
                    if self.position_entry_step is not None:
                        closed_trade_duration_hours = max(1, exec_idx - int(self.position_entry_step) + 1)
                        closed_trade_extreme = bool(int(self.max_danger_during_trade) >= 2)

                entry_fill = self._fill_price(execution_price_raw, side=1, cost_rate=cost_rate)
                self.position = 1
                self.entry_price = entry_fill
                self.position_size_at_entry = float(self.position_size)
                self.long_entries += 1
                opened_direction = 1
                self.position_entry_step = exec_idx
                self.entry_atr_norm = float(max(1e-6, atr_norm_now))
                self.max_danger_during_trade = int(max(0, danger_level))
                commission_component += self.commission * self.position_size_at_entry

        elif action == 2:
            if self.position >= 0:
                if self.position == 1 and self.entry_price is not None:
                    exit_price = self._fill_price(execution_price_raw, side=-1, cost_rate=cost_rate)
                    pnl = ((exit_price - self.entry_price) / max(1e-9, self.entry_price)) * self.position_size_at_entry
                    realized_pnl += pnl
                    long_trade_closed = True
                    long_trade_pnl += pnl
                    commission_component += self.commission * self.position_size_at_entry
                    if self.position_entry_step is not None:
                        closed_trade_duration_hours = max(1, exec_idx - int(self.position_entry_step) + 1)
                        closed_trade_extreme = bool(int(self.max_danger_during_trade) >= 2)

                entry_fill = self._fill_price(execution_price_raw, side=-1, cost_rate=cost_rate)
                self.position = -1
                self.entry_price = entry_fill
                self.position_size_at_entry = float(self.position_size)
                self.short_entries += 1
                opened_direction = -1
                self.position_entry_step = exec_idx
                self.entry_atr_norm = float(max(1e-6, atr_norm_now))
                self.max_danger_during_trade = int(max(0, danger_level))
                commission_component += self.commission * self.position_size_at_entry

        elif action == 3:
            if self.position == 1 and self.entry_price is not None:
                exit_price = self._fill_price(execution_price_raw, side=-1, cost_rate=cost_rate)
                pnl = ((exit_price - self.entry_price) / max(1e-9, self.entry_price)) * self.position_size_at_entry
                realized_pnl += pnl
                long_trade_closed = True
                long_trade_pnl += pnl
                commission_component += self.commission * self.position_size_at_entry
            elif self.position == -1 and self.entry_price is not None:
                exit_price = self._fill_price(execution_price_raw, side=1, cost_rate=cost_rate)
                pnl = ((self.entry_price - exit_price) / max(1e-9, self.entry_price)) * self.position_size_at_entry
                realized_pnl += pnl
                short_trade_closed = True
                short_trade_pnl += pnl
                commission_component += self.commission * self.position_size_at_entry

            if self.position != 0 and self.position_entry_step is not None:
                closed_trade_duration_hours = max(1, exec_idx - int(self.position_entry_step) + 1)
                closed_trade_extreme = bool(int(self.max_danger_during_trade) >= 2)

            self.position = 0
            self.entry_price = None
            self.position_size_at_entry = 0.0
            self.position_entry_step = None
            self.entry_atr_norm = 0.0
            self.max_danger_during_trade = 0

        if action == 0 and self.position == 0:
            self.consecutive_hold_steps += 1
        else:
            self.consecutive_hold_steps = 0

        self.episode_step_count += 1
        if action != 0:
            self.episode_active_steps += 1
        if self.position != 0:
            self.episode_invested_steps += 1
        activity_ratio = self.episode_active_steps / max(1, self.episode_step_count)
        invested_ratio = self.episode_invested_steps / max(1, self.episode_step_count)
        if action == 2:
            self.steps_since_short = 0
        else:
            self.steps_since_short += 1
        short_gap_penalty = 0.0
        if self.short_gap_penalty_coef > 0.0 and self.steps_since_short > self.short_gap_threshold:
            short_gap_penalty = self.short_gap_penalty_coef * (self.steps_since_short - self.short_gap_threshold)

        if self.position != 0:
            self.max_danger_during_trade = int(max(int(self.max_danger_during_trade), int(max(0, danger_level))))

        if self.position == 1 and self.entry_price is not None:
            unrealized = ((state_price - self.entry_price) / max(1e-9, self.entry_price)) * self.position_size_at_entry
        elif self.position == -1 and self.entry_price is not None:
            unrealized = ((self.entry_price - state_price) / max(1e-9, self.entry_price)) * self.position_size_at_entry
        else:
            unrealized = 0.0

        reward += (realized_pnl - commission_component)
        if action != 0:
            reward += 0.00015 * unrealized
        short_bonus = 0.0
        long_bonus = 0.0
        if self.short_reward_bonus_coef > 0.0 and short_trade_closed and short_trade_pnl > 0:
            short_bonus = self.short_reward_bonus_coef * short_trade_pnl
            reward += short_bonus
        if self.long_reward_bonus_coef > 0.0 and long_trade_closed and long_trade_pnl > 0:
            long_bonus = self.long_reward_bonus_coef * long_trade_pnl
            reward += long_bonus
        if (prev_position == 1 and action == 2) or (prev_position == -1 and action == 1):
            reward -= self.reverse_action_penalty_coef * max(self.position_size, self.position_size_at_entry)

        if regime == 0 and self.position != 0:
            reward -= 0.0005 * abs(unrealized)
        elif regime == 1 and self.position != 0 and action != 0:
            reward += 0.00035 * abs(unrealized)
        elif regime == 2 and action in [1, 2]:
            reward -= 0.001

        if opened_direction != 0 and self.directional_balance_reward_coef != 0.0:
            total_entries = max(1, self.long_entries + self.short_entries)
            imbalance = (self.long_entries - self.short_entries) / total_entries
            reward += self.directional_balance_reward_coef * (-imbalance if opened_direction == 1 else imbalance)

        if opened_direction != 0 and self.entry_quality_coef > 0.0:
            direction_score = regime_signal_score if opened_direction == 1 else -regime_signal_score
            if direction_score >= self.entry_quality_threshold:
                strength = direction_score - self.entry_quality_threshold
                entry_quality_component += self.entry_quality_coef * (0.35 + strength)
            else:
                weakness = self.entry_quality_threshold - direction_score
                entry_quality_component -= self.entry_quality_coef * (0.50 + weakness)
            reward += entry_quality_component

        if opened_direction != 0:
            if self.high_confidence_open_bonus_coef > 0.0 and setup_confidence >= self.confidence_reward_threshold:
                bonus_strength = (setup_confidence - self.confidence_reward_threshold) / max(1e-6, 1.0 - self.confidence_reward_threshold)
                confidence_reward_component += self.high_confidence_open_bonus_coef * (0.35 + 0.65 * min(1.0, bonus_strength))
            if self.low_confidence_open_penalty_coef > 0.0 and setup_confidence < self.confidence_reward_threshold:
                weakness = (self.confidence_reward_threshold - setup_confidence) / max(1e-6, self.confidence_reward_threshold)
                confidence_reward_component -= self.low_confidence_open_penalty_coef * (0.35 + 0.65 * min(1.0, weakness))
            reward += confidence_reward_component

        if self.breakout_enabled and action in [1, 2]:
            best_abs_strength = float(breakout_state["best_abs_strength"])
            aligned = (action == 1 and selected_signal > 0) or (action == 2 and selected_signal < 0)

            if selected_signal == 0:
                breakout_alignment_component -= 0.25 * self.breakout_misaligned_penalty_coef
            elif aligned:
                breakout_alignment_component += self.breakout_alignment_bonus_coef
                breakout_alignment_component += self.breakout_strength_bonus_coef * min(3.0, abs(selected_strength))
            else:
                breakout_alignment_component -= self.breakout_misaligned_penalty_coef * (0.6 + min(2.0, abs(selected_strength)))

            if self.breakout_selector_bonus_coef > 0.0 and best_abs_strength > 1e-9:
                selector_ratio = abs(selected_strength) / best_abs_strength
                breakout_selector_component += self.breakout_selector_bonus_coef * (2.0 * selector_ratio - 1.0)

            if self.breakout_prior_bonus_coef > 0.0 and self.breakout_prior_n in self.breakout_lookbacks:
                if selected_n == self.breakout_prior_n:
                    breakout_prior_component += self.breakout_prior_bonus_coef
                else:
                    breakout_prior_component -= 0.35 * self.breakout_prior_bonus_coef

            reward += breakout_alignment_component + breakout_selector_component + breakout_prior_component

        # Use invested_ratio (position != 0) as primary target instead of active_steps
        band_low = max(0.0, self.target_active_steps_pct - self.target_active_steps_tolerance)
        band_high = min(1.0, self.target_active_steps_pct + self.target_active_steps_tolerance)
        invested_band_low = max(0.0, self.target_active_steps_pct - self.target_active_steps_tolerance)
        invested_band_high = min(1.0, self.target_active_steps_pct + self.target_active_steps_tolerance)
        invested_closeness = max(0.0, 1.0 - abs(invested_ratio - self.target_active_steps_pct) / max(1e-6, self.target_active_steps_tolerance))
        activity_closeness = max(0.0, 1.0 - abs(activity_ratio - self.target_active_steps_pct) / max(1e-6, self.target_active_steps_tolerance))

        if action != 0:
            if activity_ratio < band_low and self.active_target_reward_coef > 0.0:
                deficit = (band_low - activity_ratio) / max(1e-6, band_low)
                activity_gap = float(deficit)
                activity_reward_component += self.active_target_reward_coef * (0.6 + 1.8 * min(1.0, deficit))
            elif activity_ratio > band_high and self.activity_hold_penalty_coef > 0.0:
                excess = (activity_ratio - band_high) / max(1e-6, 1.0 - band_high)
                activity_gap = float(excess)
                activity_reward_component -= self.activity_hold_penalty_coef * (0.6 + 1.8 * min(1.0, excess))
            elif self.active_target_reward_coef > 0.0:
                activity_reward_component += 0.15 * self.active_target_reward_coef * activity_closeness
        else:
            if activity_ratio < band_low and self.activity_hold_penalty_coef > 0.0:
                deficit = (band_low - activity_ratio) / max(1e-6, band_low)
                activity_gap = float(deficit)
                activity_reward_component -= self.activity_hold_penalty_coef * (0.6 + 1.8 * min(1.0, deficit))
            elif activity_ratio > band_high and self.activity_hold_penalty_coef > 0.0:
                excess = (activity_ratio - band_high) / max(1e-6, 1.0 - band_high)
                activity_gap = float(excess)
                activity_reward_component += 0.25 * self.activity_hold_penalty_coef * (0.4 + 1.2 * min(1.0, excess))

        reward += activity_reward_component
        reward -= short_gap_penalty

        if closed_trade_duration_hours > 0:
            d = int(closed_trade_duration_hours)
            if d < self.min_trade_duration_hours:
                short_ratio = (self.min_trade_duration_hours - d) / max(1, self.min_trade_duration_hours)
                duration_reward_component -= self.trade_duration_bonus_coef * 0.35 * short_ratio
            elif d <= self.preferred_max_trade_duration_hours:
                duration_reward_component += self.trade_duration_bonus_coef
            elif d <= self.hard_max_trade_duration_hours:
                over = d - self.preferred_max_trade_duration_hours
                if closed_trade_extreme:
                    duration_reward_component += self.trade_duration_bonus_coef * 0.30
                else:
                    duration_reward_component -= self.overhold_penalty_coef * (over / max(1, self.hard_max_trade_duration_hours - self.preferred_max_trade_duration_hours))
            else:
                over_hard = d - self.hard_max_trade_duration_hours
                if closed_trade_extreme:
                    duration_reward_component -= self.overhold_penalty_coef * 0.25 * (1.0 + over_hard / max(1, self.hard_max_trade_duration_hours))
                else:
                    duration_reward_component -= self.overhold_penalty_coef * (1.0 + over_hard / max(1, self.preferred_max_trade_duration_hours))

        if self.position != 0 and self.position_entry_step is not None:
            ongoing_hours = max(1, exec_idx - int(self.position_entry_step) + 1)
            ongoing_extreme = bool(int(self.max_danger_during_trade) >= 2)
            if ongoing_hours > self.preferred_max_trade_duration_hours and not ongoing_extreme:
                over = ongoing_hours - self.preferred_max_trade_duration_hours
                duration_reward_component -= self.overhold_step_penalty_coef * min(over, 96)
            info["ongoing_trade_duration_hours"] = int(ongoing_hours)
            info["ongoing_trade_extreme"] = ongoing_extreme
        else:
            info["ongoing_trade_duration_hours"] = 0
            info["ongoing_trade_extreme"] = False

        reward += duration_reward_component

        if closed_trade_duration_hours > 0:
            abs_realized = abs(float(realized_pnl))
            if self.small_close_pnl_penalty_coef > 0.0 and self.small_close_pnl_threshold > 0.0 and abs_realized < self.small_close_pnl_threshold:
                deficit = (self.small_close_pnl_threshold - abs_realized) / max(1e-9, self.small_close_pnl_threshold)
                close_quality_component -= self.small_close_pnl_penalty_coef * (0.35 + 0.65 * deficit)
            if self.good_close_bonus_coef > 0.0 and self.good_close_pnl_threshold > 0.0 and realized_pnl > self.good_close_pnl_threshold:
                excess = (float(realized_pnl) - self.good_close_pnl_threshold) / max(1e-9, self.good_close_pnl_threshold)
                close_quality_component += self.good_close_bonus_coef * (0.30 + 0.70 * min(2.0, max(0.0, excess)))
            reward += close_quality_component
        # Hold penalty: apply ONLY when position is flat (not investing)
        # When position != 0, a HOLD is legitimate (staying invested)
        hold_after_grace = self.position == 0 or ongoing_hours > int(self.hold_in_position_grace_hours)
        hold_pressure_enabled = bool(invested_ratio <= invested_band_high)
        hold_penalty_enabled = (
            action == 0
            and self.position == 0  # Only penalize hold when FLAT (not investing)
            and self.hold_penalty_base > 0.0
            and hold_after_grace
            and hold_pressure_enabled
        )
        if hold_penalty_enabled:
            hold_streak = float(max(1, self.consecutive_hold_steps))
            exponent = min(50.0, self.hold_penalty_growth * (hold_streak - 1.0))
            raw_penalty = self.hold_penalty_base * float(np.exp(exponent))
            hold_penalty_component = min(self.hold_penalty_cap, raw_penalty)
            reward -= hold_penalty_component

        if action == 0 and self.position != 0 and ongoing_hours > int(self.hold_in_position_grace_hours) and self.hold_in_position_penalty_coef > 0.0 and (hold_pressure_enabled or ongoing_hours > self.preferred_max_trade_duration_hours):
            over = ongoing_hours - int(self.hold_in_position_grace_hours)
            exponent = min(50.0, self.hold_in_position_penalty_growth * float(over))
            raw_penalty = self.hold_in_position_penalty_coef * float(np.exp(exponent))
            hold_in_position_penalty_component = min(self.hold_in_position_penalty_cap, raw_penalty)
            reward -= hold_in_position_penalty_component

        if action == 0 and self.position == 0 and self.hold_missed_move_coef > 0.0 and hold_pressure_enabled:
            denom = max(1e-9, abs(float(state_price)))
            realized_move = abs(float(execution_price_raw) - float(state_price)) / denom
            atr_norm_now = float(pd.to_numeric(row.get("ATR"), errors="coerce") / denom) if "ATR" in row.index else 0.0
            atr_norm_now = max(0.0, atr_norm_now) if not np.isnan(atr_norm_now) else 0.0
            move_floor = self.hold_missed_move_atr_floor * atr_norm_now
            missed = max(0.0, realized_move - move_floor)
            hold_missed_move_component = self.hold_missed_move_coef * missed
            reward -= hold_missed_move_component

        if action != 0 and self.turnover_penalty_coef > 0.0:
            turnover_scale = max(0.05, float(self.position_size), float(self.position_size_at_entry))
            if action in [1, 2]:
                turnover_scale *= 1.10
            turnover_penalty_component = self.turnover_penalty_coef * turnover_scale
            reward -= turnover_penalty_component

        if self.equity_from_reward:
            equity_return_component = float(reward)
            shaping_component_total = 0.0
            raw_reward = float(reward)
        else:
            equity_return_component = float(realized_pnl - commission_component)
            shaping_component_total = float(reward - equity_return_component)
            raw_reward = float(equity_return_component + self.reward_shaping_weight * shaping_component_total)

        projected_equity = float(self.equity * (1.0 + equity_return_component))
        projected_peak = float(max(self.peak_equity, projected_equity))
        current_drawdown = max(0.0, 1.0 - projected_equity / max(1e-9, projected_peak))
        if self.drawdown_penalty_coef > 0.0 and current_drawdown > self.drawdown_penalty_threshold:
            excess = (current_drawdown - self.drawdown_penalty_threshold) / max(1e-6, 1.0 - self.drawdown_penalty_threshold)
            drawdown_penalty_component = self.drawdown_penalty_coef * (0.35 + 0.65 * min(1.0, excess))
            raw_reward -= drawdown_penalty_component

        reward = float(np.clip(raw_reward, -self.reward_clip_abs, self.reward_clip_abs)) if self.reward_clip_abs > 0.0 else raw_reward
        self.equity = projected_equity
        self.peak_equity = projected_peak

        info.update({
            "danger_level": int(max(0, danger_level)),
            "position_size": float(self.position_size),
            "base_position_size_pre_dynamic": float(base_position_size_pre_dynamic),
            "dynamic_sizing_enabled": bool(self.enable_dynamic_sizing),
            "dynamic_size_multiplier": float(dynamic_size_multiplier),
            "setup_confidence": float(setup_confidence),
            "volatility_confidence": float(confidence_components["volatility_score"]),
            "signal_confidence": float(confidence_components["signal_score"]),
            "zone_confidence": float(confidence_components["zone_score"]),
            "breakout_confidence": float(confidence_components["breakout_score"]),
            "confidence_gate_threshold": float(self.confidence_gate_threshold),
            "long_entries": int(self.long_entries),
            "short_entries": int(self.short_entries),
            "executed_action": int(action),
            "decision_step": int(self.current_step),
            "executed_step": int(exec_idx),
            "price": float(state_price),
            "execution_price": float(execution_price_raw),
            "position": int(self.position),
            "equity": float(self.equity),
            "trade_duration_hours": int(closed_trade_duration_hours),
            "trade_duration_extreme": bool(closed_trade_extreme),
            "duration_reward_component": float(duration_reward_component),
            "hold_streak": int(self.consecutive_hold_steps),
            "hold_penalty_component": float(hold_penalty_component),
            "hold_in_position_penalty_component": float(hold_in_position_penalty_component),
            "commission_component": float(commission_component),
            "equity_return_component": float(equity_return_component),
            "raw_reward": float(raw_reward),
            "reward_clipped": bool(abs(raw_reward - reward) > 1e-12),
            "shaping_component_total": float(shaping_component_total),
            "hold_missed_move_component": float(hold_missed_move_component),
            "episode_step_count": int(self.episode_step_count),
            "episode_active_steps": int(self.episode_active_steps),
            "episode_invested_steps": int(self.episode_invested_steps),
            "activity_ratio": float(activity_ratio),
            "invested_ratio": float(invested_ratio),
            "activity_target": float(self.target_active_steps_pct),
            "activity_band_low": float(band_low),
            "activity_band_high": float(band_high),
            "activity_gap": float(activity_gap),
            "activity_closeness": float(activity_closeness),
            "activity_reward_component": float(activity_reward_component),
            "invested_ratio": float(invested_ratio),
            "invested_closeness": float(invested_closeness),
            "hold_pressure_enabled": bool(hold_pressure_enabled),
            "regime_signal_score": float(regime_signal_score),
            "entry_quality_component": float(entry_quality_component),
            "confidence_reward_component": float(confidence_reward_component),
            "close_quality_component": float(close_quality_component),
            "breakout_alignment_component": float(breakout_alignment_component),
            "breakout_selector_component": float(breakout_selector_component),
            "breakout_prior_component": float(breakout_prior_component),
            "selected_breakout_idx": int(breakout_state["selected_idx"]),
            "selected_breakout_n": int(breakout_state["selected_n"]),
            "selected_breakout_signal": int(breakout_state["selected_signal"]),
            "selected_breakout_strength": float(breakout_state["selected_strength"]),
            "breakout_consensus": float(breakout_state["consensus"]),
            "breakout_best_abs_strength": float(breakout_state["best_abs_strength"]),
            "turnover_penalty_component": float(turnover_penalty_component),
            "cost_rate": float(cost_rate),
            "support_dist": float(sup_dist),
            "support_strength": float(sup_strength),
            "resistance_dist": float(res_dist),
            "resistance_strength": float(res_strength),
            "current_drawdown": float(current_drawdown),
            "drawdown_penalty_component": float(drawdown_penalty_component),
            "short_gap_penalty": float(short_gap_penalty),
            "short_reward_bonus": float(short_bonus),
            "long_reward_bonus": float(long_bonus),
        })
        if isinstance(self.max_drawdown, (int, float)) and 0 < self.max_drawdown < 1:
            if self.equity <= self.initial_capital * (1 - self.max_drawdown):
                done = True
                info["reason"] = "max_drawdown"

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            info["reason"] = "end_of_data"

        next_obs, _ = self._get_observation()
        return next_obs, reward, done, False, info
