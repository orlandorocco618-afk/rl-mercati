import os
import json
import pandas as pd
import sys

sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn

from env.trading_env import TradingEnv
from utils.kalman import apply_kalman_filter
from utils.data_utils import merge_h1_with_d1_regimes
from utils.error_handler import (
    log_error, print_step_summary, print_final_report,
    check_file_exists, validate_dataframe
)


# ============================================================
#  DATA LOADING
# ============================================================

def load_data(
    h1_path: str = "data/xauusd_h1_clean.csv",
    d1_path: str = "data/xauusd_d1_clean.csv"
) -> pd.DataFrame:
    """
    Loads H1 for trading and D1 for the regime computation.
    D1 regimes are assigned to every H1 candle of the same date.
    """
    df = merge_h1_with_d1_regimes(h1_path, d1_path)
    return df


def split_train_val_test_by_date(
    df: pd.DataFrame,
    train_end: str = "2022-12-31 23:00:00",
    val_end: str = "2023-12-31 23:00:00",
):
    if "Time" not in df.columns:
        raise ValueError("Time column missing for the temporal split")
    out = df.copy()
    out["Time"] = pd.to_datetime(out["Time"], errors="coerce")
    out = out.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    if val_end_ts <= train_end_ts:
        raise ValueError("val_end must come after train_end")

    df_train = out.loc[out["Time"] <= train_end_ts].reset_index(drop=True)
    df_val = out.loc[(out["Time"] > train_end_ts) & (out["Time"] <= val_end_ts)].reset_index(drop=True)
    df_test = out.loc[out["Time"] > val_end_ts].reset_index(drop=True)
    return df_train, df_val, df_test


# ============================================================
#  ENVIRONMENT CREATION
# ============================================================

def make_env(
    df,
    window_size=100,
    initial_capital=10000.0,
    reverse_action_penalty_coef: float = 0.002,
    target_active_steps_pct: float = 0.15,
    target_active_steps_tolerance: float = 0.03,
    active_target_reward_coef: float = 0.0022,
    activity_hold_penalty_coef: float = 0.0028,
    hold_penalty_base: float = 0.000012,
    hold_penalty_growth: float = 0.05,
    hold_penalty_cap: float = 0.0022,
    hold_in_position_grace_hours: int = 8,
    hold_in_position_penalty_coef: float = 0.00045,
    hold_in_position_penalty_growth: float = 0.055,
    hold_in_position_penalty_cap: float = 0.008,
    hold_missed_move_coef: float = 1.0,
    entry_quality_coef: float = 0.0,
    entry_quality_threshold: float = 0.15,
    entry_quality_noise_scale: float = 0.5,
    small_close_pnl_threshold: float = 0.00008,
    small_close_pnl_penalty_coef: float = 0.0,
    good_close_pnl_threshold: float = 0.00025,
    good_close_bonus_coef: float = 0.0,
    breakout_entry_strength_quantile: float = 0.55,
    breakout_strength_quantile_window: int = 1500,
    atr_stop_multiple: float = 2.5,
    hard_time_stop_hours: int = 36,
    reward_shaping_weight: float = 0.12,
    turnover_penalty_coef: float = 0.00035,
    breakout_prior_n: int = 96,
    breakout_prior_bonus_coef: float = 0.00025,
    min_trade_duration_hours: int = 0,
    preferred_max_trade_duration_hours: int = 24,
    trade_duration_bonus_coef: float = 0.0015,
    extra_env_kwargs: dict | None = None,
):
    def _init():
        env_kwargs = {
            "df": df,
            "window_size": window_size,
            "initial_capital": initial_capital,
            "max_drawdown": 0.30,
            "allow_hold_when_flat": True,
            "reverse_action_penalty_coef": reverse_action_penalty_coef,
            "random_start_on_reset": True,
            "min_episode_steps": 1500,
            "target_active_steps_pct": target_active_steps_pct,
            "target_active_steps_tolerance": target_active_steps_tolerance,
            "active_target_reward_coef": active_target_reward_coef,
            "activity_hold_penalty_coef": activity_hold_penalty_coef,
            "hold_penalty_base": hold_penalty_base,
            "hold_penalty_growth": hold_penalty_growth,
            "hold_penalty_cap": hold_penalty_cap,
            "hold_penalty_only_when_flat": False,
            "hold_in_position_grace_hours": hold_in_position_grace_hours,
            "hold_in_position_penalty_coef": hold_in_position_penalty_coef,
            "hold_in_position_penalty_growth": hold_in_position_penalty_growth,
            "hold_in_position_penalty_cap": hold_in_position_penalty_cap,
            "hold_missed_move_coef": hold_missed_move_coef,
            "entry_quality_coef": entry_quality_coef,
            "entry_quality_threshold": entry_quality_threshold,
            "entry_quality_noise_scale": entry_quality_noise_scale,
            "small_close_pnl_threshold": small_close_pnl_threshold,
            "small_close_pnl_penalty_coef": small_close_pnl_penalty_coef,
            "good_close_pnl_threshold": good_close_pnl_threshold,
            "good_close_bonus_coef": good_close_bonus_coef,
            "breakout_entry_strength_quantile": breakout_entry_strength_quantile,
            "breakout_strength_quantile_window": breakout_strength_quantile_window,
            "breakout_block_misaligned_entries": True,
            "breakout_block_weak_entries": True,
            "atr_stop_multiple": atr_stop_multiple,
            "hard_time_stop_hours": hard_time_stop_hours,
            "reward_shaping_weight": reward_shaping_weight,
            "turnover_penalty_coef": turnover_penalty_coef,
            "breakout_prior_n": breakout_prior_n,
            "breakout_prior_bonus_coef": breakout_prior_bonus_coef,
            "min_trade_duration_hours": min_trade_duration_hours,
            "preferred_max_trade_duration_hours": preferred_max_trade_duration_hours,
            "trade_duration_bonus_coef": trade_duration_bonus_coef,
            "regime_hard_stop": False,
            "enforce_directional_rebalance": True,
            "rebalance_after_entries": 40,
            "rebalance_dominance_threshold": 0.70,
            "directional_balance_reward_coef": 0.001,
        }
        if extra_env_kwargs:
            env_kwargs.update(extra_env_kwargs)
        return TradingEnv(**env_kwargs)
    return _init


# ============================================================
#  TRAINING PPO
# ============================================================


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _get_env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return sorted(set(int(x) for x in default if int(x) >= 4))
    values: list[int] = []
    for token in raw.replace(";", ",").split(","):
        t = token.strip()
        if not t:
            continue
        try:
            v = int(t)
        except Exception:
            continue
        if v >= 4:
            values.append(v)
    if not values:
        return sorted(set(int(x) for x in default if int(x) >= 4))
    return sorted(set(values))


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def train_agent(
    data_path: str,
    model_path: str = "models/ppo_trading.zip",
    timesteps: int = 300_000,
    tensorboard_log: str = "logs/"
):

    print("Loading data...")
    df = load_data()

    df = apply_kalman_filter(df, price_col="Close")
    df["Close"] = df["Close_KF"]
    train_end = os.getenv("RL_TRAIN_END", "2022-12-31 23:00:00").strip()
    val_end = os.getenv("RL_VAL_END", "2023-12-31 23:00:00").strip()
    df_train, _, _ = split_train_val_test_by_date(df, train_end=train_end, val_end=val_end)

    print("Creating the RL environment...")
    env = DummyVecEnv([make_env(df_train)])

    print("Initializing the PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        target_kl=0.03,
        max_grad_norm=0.5,
    )

    model.set_random_seed(42)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="models/checkpoints/",
        name_prefix="ppo_trading"
    )

    print("Starting training...")
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    print(f"Saving the final model to: {model_path}")
    model.save(model_path)

    env.close()
    print("Training completed.")


# ============================================================
#  ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    errors = []
    status = "FAILURE"
    
    try:
        train_timesteps = _get_env_int("RL_TRAIN_TIMESTEPS", 300_000)
        checkpoint_freq = _get_env_int("RL_CHECKPOINT_FREQ", 50_000)
        seed = _get_env_int("RL_SEED", 42)
        learning_rate = _get_env_float("RL_LR", 3e-4)
        ent_coef = _get_env_float("RL_ENT_COEF", 0.02)
        reverse_action_penalty_coef = _get_env_float("RL_REVERSE_ACTION_PENALTY_COEF", 0.002)
        target_active_steps_pct = _get_env_float("RL_TARGET_ACTIVE_STEPS_PCT", 0.15)
        target_active_steps_tolerance = _get_env_float("RL_TARGET_ACTIVE_STEPS_TOLERANCE", 0.03)
        active_target_reward_coef = _get_env_float("RL_ACTIVE_TARGET_REWARD_COEF", 0.0022)
        activity_hold_penalty_coef = _get_env_float("RL_ACTIVITY_HOLD_PENALTY_COEF", 0.0028)
        hold_penalty_base = _get_env_float("RL_HOLD_PENALTY_BASE", 0.000012)
        hold_penalty_growth = _get_env_float("RL_HOLD_PENALTY_GROWTH", 0.05)
        hold_penalty_cap = _get_env_float("RL_HOLD_PENALTY_CAP", 0.0022)
        hold_in_position_grace_hours = _get_env_int("RL_HOLD_IN_POSITION_GRACE_HOURS", 8)
        hold_in_position_penalty_coef = _get_env_float("RL_HOLD_IN_POSITION_PENALTY_COEF", 0.00045)
        hold_in_position_penalty_growth = _get_env_float("RL_HOLD_IN_POSITION_PENALTY_GROWTH", 0.055)
        hold_in_position_penalty_cap = _get_env_float("RL_HOLD_IN_POSITION_PENALTY_CAP", 0.008)
        hold_missed_move_coef = _get_env_float("RL_HOLD_MISSED_MOVE_COEF", 1.0)
        entry_quality_coef = _get_env_float("RL_ENTRY_QUALITY_COEF", 0.0)
        entry_quality_threshold = _get_env_float("RL_ENTRY_QUALITY_THRESHOLD", 0.15)
        entry_quality_noise_scale = _get_env_float("RL_ENTRY_QUALITY_NOISE_SCALE", 0.5)
        small_close_pnl_threshold = _get_env_float("RL_SMALL_CLOSE_PNL_THRESHOLD", 0.00008)
        small_close_pnl_penalty_coef = _get_env_float("RL_SMALL_CLOSE_PNL_PENALTY_COEF", 0.0)
        good_close_pnl_threshold = _get_env_float("RL_GOOD_CLOSE_PNL_THRESHOLD", 0.00025)
        good_close_bonus_coef = _get_env_float("RL_GOOD_CLOSE_BONUS_COEF", 0.0)
        breakout_lookbacks = _get_env_int_list("RL_BREAKOUT_LOOKBACKS", [48, 72, 96, 144])
        breakout_alignment_bonus_coef = _get_env_float("RL_BREAKOUT_ALIGNMENT_BONUS_COEF", 0.0008)
        breakout_misaligned_penalty_coef = _get_env_float("RL_BREAKOUT_MISALIGNED_PENALTY_COEF", 0.0009)
        breakout_strength_bonus_coef = _get_env_float("RL_BREAKOUT_STRENGTH_BONUS_COEF", 0.00035)
        breakout_selector_bonus_coef = _get_env_float("RL_BREAKOUT_SELECTOR_BONUS_COEF", 0.00025)
        breakout_entry_strength_quantile = _get_env_float("RL_BREAKOUT_ENTRY_STRENGTH_QUANTILE", 0.55)
        breakout_strength_quantile_window = _get_env_int("RL_BREAKOUT_STRENGTH_QUANTILE_WINDOW", 1500)
        breakout_block_weak_entries = _get_env_bool("RL_BREAKOUT_BLOCK_WEAK_ENTRIES", True)
        breakout_block_misaligned_entries = _get_env_bool("RL_BREAKOUT_BLOCK_MISALIGNED_ENTRIES", True)
        atr_stop_multiple = _get_env_float("RL_ATR_STOP_MULTIPLE", 2.5)
        hard_time_stop_hours = _get_env_int("RL_HARD_TIME_STOP_HOURS", 36)
        reward_shaping_weight = _get_env_float("RL_REWARD_SHAPING_WEIGHT", 0.12)
        turnover_penalty_coef = _get_env_float("RL_TURNOVER_PENALTY_COEF", 0.00035)
        enable_dynamic_sizing = _get_env_bool("RL_ENABLE_DYNAMIC_SIZING", True)
        dynamic_reference_window = _get_env_int("RL_DYNAMIC_REFERENCE_WINDOW", 96)
        dynamic_size_floor = _get_env_float("RL_DYNAMIC_SIZE_FLOOR", 0.70)
        dynamic_size_ceiling = _get_env_float("RL_DYNAMIC_SIZE_CEILING", 1.35)
        dynamic_confidence_power = _get_env_float("RL_DYNAMIC_CONFIDENCE_POWER", 1.10)
        dynamic_volatility_weight = _get_env_float("RL_DYNAMIC_VOLATILITY_WEIGHT", 0.45)
        dynamic_signal_weight = _get_env_float("RL_DYNAMIC_SIGNAL_WEIGHT", 0.20)
        dynamic_zone_weight = _get_env_float("RL_DYNAMIC_ZONE_WEIGHT", 0.20)
        dynamic_breakout_weight = _get_env_float("RL_DYNAMIC_BREAKOUT_WEIGHT", 0.15)
        enable_confidence_gate = _get_env_bool("RL_ENABLE_CONFIDENCE_GATE", False)
        confidence_gate_threshold = _get_env_float("RL_CONFIDENCE_GATE_THRESHOLD", 0.60)
        drawdown_penalty_coef = _get_env_float("RL_DRAWDOWN_PENALTY_COEF", 0.0008)
        drawdown_penalty_threshold = _get_env_float("RL_DRAWDOWN_PENALTY_THRESHOLD", 0.03)
        low_confidence_open_penalty_coef = _get_env_float("RL_LOW_CONFIDENCE_OPEN_PENALTY_COEF", 0.00035)
        high_confidence_open_bonus_coef = _get_env_float("RL_HIGH_CONFIDENCE_OPEN_BONUS_COEF", 0.00015)
        confidence_reward_threshold = _get_env_float("RL_CONFIDENCE_REWARD_THRESHOLD", 0.58)
        breakout_prior_n = _get_env_int("RL_BREAKOUT_PRIOR_N", 96)
        breakout_prior_bonus_coef = _get_env_float("RL_BREAKOUT_PRIOR_BONUS_COEF", 0.00025)
        min_trade_duration_hours = _get_env_int("RL_MIN_TRADE_DURATION_HOURS", 0)
        preferred_max_trade_duration_hours = _get_env_int("RL_PREFERRED_MAX_TRADE_DURATION_HOURS", 24)
        trade_duration_bonus_coef = _get_env_float("RL_TRADE_DURATION_BONUS_COEF", 0.0015)
        spread_bps = _get_env_float("RL_SPREAD_BPS", 2.0)
        slippage_bps = _get_env_float("RL_SLIPPAGE_BPS", 1.0)
        slippage_atr_coef = _get_env_float("RL_SLIPPAGE_ATR_COEF", 0.0)
        execution_delay_bars = _get_env_int("RL_EXECUTION_DELAY_BARS", 1)
        train_end = os.getenv("RL_TRAIN_END", "2022-12-31 23:00:00").strip()
        val_end = os.getenv("RL_VAL_END", "2023-12-31 23:00:00").strip()
        resume_model_path = os.getenv("RL_RESUME_MODEL", "").strip()
        skip_post_analysis = _get_env_bool("RL_SKIP_POST_ANALYSIS", False)
        print(
            f"[CONFIG] timesteps={train_timesteps}, checkpoint_freq={checkpoint_freq}, "
            f"lr={learning_rate}, ent_coef={ent_coef}, seed={seed}"
        )
        print(
            f"[CONFIG] activity_target={target_active_steps_pct:.3f}±{target_active_steps_tolerance:.3f}, "
            f"active_coef={active_target_reward_coef:.6f}, hold_coef={activity_hold_penalty_coef:.6f}"
        )
        print(
            f"[CONFIG] hold_exp_base={hold_penalty_base:.6f}, hold_exp_growth={hold_penalty_growth:.4f}, "
            f"hold_exp_cap={hold_penalty_cap:.6f}, in_pos_grace_h={hold_in_position_grace_hours}"
        )
        print(f"[CONFIG] reverse_penalty_coef={reverse_action_penalty_coef:.6f}")
        print(
            f"[CONFIG] quality: entry_coef={entry_quality_coef:.6f}, entry_thr={entry_quality_threshold:.3f}, "
            f"small_close_pen={small_close_pnl_penalty_coef:.6f}, good_close_bonus={good_close_bonus_coef:.6f}"
        )
        print(
            f"[CONFIG] breakout_n_candidates={breakout_lookbacks}, "
            f"align_bonus={breakout_alignment_bonus_coef:.6f}, "
            f"misalign_penalty={breakout_misaligned_penalty_coef:.6f}, "
            f"selector_bonus={breakout_selector_bonus_coef:.6f}"
        )
        print(
            f"[CONFIG] breakout_entry_q={breakout_entry_strength_quantile:.2f}, "
            f"q_window={breakout_strength_quantile_window}, "
            f"atr_stop_mult={atr_stop_multiple:.2f}, time_stop_h={hard_time_stop_hours}, "
            f"reward_shaping_weight={reward_shaping_weight:.3f}"
        )
        print(
            f"[CONFIG] turnover_penalty={turnover_penalty_coef:.6f}, "
            f"breakout_prior_n={breakout_prior_n}, prior_bonus={breakout_prior_bonus_coef:.6f}"
        )
        print(
            f"[CONFIG] dynamic_sizing={enable_dynamic_sizing}, floor={dynamic_size_floor:.2f}, "
            f"ceiling={dynamic_size_ceiling:.2f}, gate={enable_confidence_gate}, "
            f"dd_penalty={drawdown_penalty_coef:.6f}"
        )
        print(
            f"[CONFIG] trade_duration: min_h={min_trade_duration_hours}, "
            f"pref_max_h={preferred_max_trade_duration_hours}, bonus_coef={trade_duration_bonus_coef:.6f}"
        )
        print(
            f"[CONFIG] costs: spread_bps={spread_bps:.3f}, slippage_bps={slippage_bps:.3f}, "
            f"slippage_atr_coef={slippage_atr_coef:.5f}, execution_delay_bars={execution_delay_bars}"
        )
        print(f"[CONFIG] split: train<= {train_end}, val<= {val_end}, test> {val_end}")
        if resume_model_path:
            print(f"[CONFIG] resume_model={resume_model_path}")

        print("="*60)
        print("[STEP 1/8] Loading H1+D1 data...")
        print("="*60)
        
        check_file_exists("data/xauusd_h1_clean.csv", critical=True)
        check_file_exists("data/xauusd_d1_clean.csv", critical=True)
        
        df = None
        try:
            df = load_data()
            # In production we allow NaN from rolling features (legitimate from indicators)
            import os
            quick = os.getenv("RL_QUICK_TEST") == "1"
            if not quick:
                validate_dataframe(df, critical=False, allow_nan=True)  # Allow NaN from rolling
            print_step_summary("load_data", status="OK", rows=len(df))
        except Exception as e:
            error_msg = log_error(e, "load_data", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 2/8] Applying the Kalman filter...")
        print("="*60)
        try:
            df = apply_kalman_filter(df, price_col="Close")
            df["Close"] = df["Close_KF"]
            df_train, df_val, df_test = split_train_val_test_by_date(df, train_end=train_end, val_end=val_end)
            if len(df_train) < 2000 or len(df_test) < 500:
                raise ValueError(
                    f"Invalid split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
                )
            print_step_summary(
                "kalman_filter",
                status="OK",
                train_rows=len(df_train),
                val_rows=len(df_val),
                test_rows=len(df_test),
            )
        except Exception as e:
            error_msg = log_error(e, "kalman_filter", critical=True)
            errors.append(error_msg)
            raise

        shared_env_kwargs = {
            "spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "slippage_atr_coef": slippage_atr_coef,
            "execution_delay_bars": execution_delay_bars,
            "recompute_regimes": False,
            "breakout_lookbacks": breakout_lookbacks,
            "breakout_alignment_bonus_coef": breakout_alignment_bonus_coef,
            "breakout_misaligned_penalty_coef": breakout_misaligned_penalty_coef,
            "breakout_strength_bonus_coef": breakout_strength_bonus_coef,
            "breakout_selector_bonus_coef": breakout_selector_bonus_coef,
            "breakout_entry_strength_quantile": breakout_entry_strength_quantile,
            "breakout_strength_quantile_window": breakout_strength_quantile_window,
            "breakout_block_misaligned_entries": breakout_block_misaligned_entries,
            "breakout_block_weak_entries": breakout_block_weak_entries,
            "atr_stop_multiple": atr_stop_multiple,
            "hard_time_stop_hours": hard_time_stop_hours,
            "reward_shaping_weight": reward_shaping_weight,
            "turnover_penalty_coef": turnover_penalty_coef,
            "enable_dynamic_sizing": enable_dynamic_sizing,
            "dynamic_reference_window": dynamic_reference_window,
            "dynamic_size_floor": dynamic_size_floor,
            "dynamic_size_ceiling": dynamic_size_ceiling,
            "dynamic_confidence_power": dynamic_confidence_power,
            "dynamic_volatility_weight": dynamic_volatility_weight,
            "dynamic_signal_weight": dynamic_signal_weight,
            "dynamic_zone_weight": dynamic_zone_weight,
            "dynamic_breakout_weight": dynamic_breakout_weight,
            "enable_confidence_gate": enable_confidence_gate,
            "confidence_gate_threshold": confidence_gate_threshold,
            "drawdown_penalty_coef": drawdown_penalty_coef,
            "drawdown_penalty_threshold": drawdown_penalty_threshold,
            "low_confidence_open_penalty_coef": low_confidence_open_penalty_coef,
            "high_confidence_open_bonus_coef": high_confidence_open_bonus_coef,
            "confidence_reward_threshold": confidence_reward_threshold,
            "breakout_prior_n": breakout_prior_n,
            "breakout_prior_bonus_coef": breakout_prior_bonus_coef,
        }
        
        print("\n" + "="*60)
        print("[STEP 3/8] Creating the RL environment...")
        print("="*60)
        try:
            env = DummyVecEnv([make_env(
                df_train,
                reverse_action_penalty_coef=reverse_action_penalty_coef,
                target_active_steps_pct=target_active_steps_pct,
                target_active_steps_tolerance=target_active_steps_tolerance,
                active_target_reward_coef=active_target_reward_coef,
                activity_hold_penalty_coef=activity_hold_penalty_coef,
                hold_penalty_base=hold_penalty_base,
                hold_penalty_growth=hold_penalty_growth,
                hold_penalty_cap=hold_penalty_cap,
                hold_in_position_grace_hours=hold_in_position_grace_hours,
                hold_in_position_penalty_coef=hold_in_position_penalty_coef,
                hold_in_position_penalty_growth=hold_in_position_penalty_growth,
                hold_in_position_penalty_cap=hold_in_position_penalty_cap,
                hold_missed_move_coef=hold_missed_move_coef,
                entry_quality_coef=entry_quality_coef,
                entry_quality_threshold=entry_quality_threshold,
                entry_quality_noise_scale=entry_quality_noise_scale,
                small_close_pnl_threshold=small_close_pnl_threshold,
                small_close_pnl_penalty_coef=small_close_pnl_penalty_coef,
                good_close_pnl_threshold=good_close_pnl_threshold,
                good_close_bonus_coef=good_close_bonus_coef,
                breakout_entry_strength_quantile=breakout_entry_strength_quantile,
                breakout_strength_quantile_window=breakout_strength_quantile_window,
                atr_stop_multiple=atr_stop_multiple,
                hard_time_stop_hours=hard_time_stop_hours,
                reward_shaping_weight=reward_shaping_weight,
                turnover_penalty_coef=turnover_penalty_coef,
                breakout_prior_n=breakout_prior_n,
                breakout_prior_bonus_coef=breakout_prior_bonus_coef,
                min_trade_duration_hours=min_trade_duration_hours,
                preferred_max_trade_duration_hours=preferred_max_trade_duration_hours,
                trade_duration_bonus_coef=trade_duration_bonus_coef,
                extra_env_kwargs=shared_env_kwargs,
            )])
            print_step_summary("make_env", status="OK")
        except Exception as e:
            error_msg = log_error(e, "make_env", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 4/8] Normalizing observations and rewards...")
        print("="*60)
        try:
            # DISABLED: VecNormalize suspected of introducing NaN into the pipeline
            # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)
            print_step_summary("vecnormalize", status="SKIPPED (debug mode)")
        except Exception as e:
            error_msg = log_error(e, "vecnormalize", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 5/8] Initializing the PPO model...")
        print("="*60)
        try:
            if resume_model_path and os.path.exists(resume_model_path):
                print(f"[RESUME] Loading model from: {resume_model_path}")
                model = PPO.load(resume_model_path, env=env)
                # Ensure resume runs honor runtime hyperparameter overrides.
                model.learning_rate = float(learning_rate)
                model.lr_schedule = get_schedule_fn(float(learning_rate))
                model.ent_coef = float(ent_coef)
                print(f"[RESUME] Override lr={learning_rate}, ent_coef={ent_coef}")
            else:
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    verbose=1,
                    tensorboard_log=None,
                    learning_rate=learning_rate,
                    batch_size=64,
                    n_steps=2048,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=ent_coef,
                    target_kl=0.03,
                    max_grad_norm=0.5,
                )
            model.set_random_seed(seed)
            print_step_summary("ppo_init", status="OK")
        except Exception as e:
            error_msg = log_error(e, "ppo_init", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 6/8] Setup checkpoint callback...")
        print("="*60)
        checkpoint_callback = None
        try:
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path="models/checkpoints/",
                name_prefix="ppo_trading"
            )
            print_step_summary("checkpoint_callback", status="OK")
        except Exception as e:
            error_msg = log_error(e, "checkpoint_callback", critical=False)
            errors.append(error_msg)
            print("[WARN] Callback setup failed, continuing training without checkpoints")
        
        print("\n" + "="*60)
        print(f"[STEP 7/8] Training PPO ({train_timesteps} timesteps)...")
        print("="*60)
        try:
            callbacks = [checkpoint_callback] if checkpoint_callback else []
            model.learn(
                total_timesteps=train_timesteps,
                callback=callbacks,
                progress_bar=False
            )
            print_step_summary("model_training", status="OK", timesteps=train_timesteps)
        except Exception as e:
            error_msg = log_error(e, "model_training", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 8/8] Saving the model and the normalization...")
        print("="*60)
        try:
            os.makedirs("models", exist_ok=True)
            model.save("models/ppo_trading.zip")
            env_config = {
                "window_size": 100,
                "initial_capital": 10000.0,
                "max_drawdown": 0.30,
                "allow_hold_when_flat": True,
                "reverse_action_penalty_coef": reverse_action_penalty_coef,
                "random_start_on_reset": True,
                "min_episode_steps": 1500,
                "target_active_steps_pct": target_active_steps_pct,
                "target_active_steps_tolerance": target_active_steps_tolerance,
                "active_target_reward_coef": active_target_reward_coef,
                "activity_hold_penalty_coef": activity_hold_penalty_coef,
                "hold_penalty_base": hold_penalty_base,
                "hold_penalty_growth": hold_penalty_growth,
                "hold_penalty_cap": hold_penalty_cap,
                "hold_penalty_only_when_flat": False,
                "hold_in_position_grace_hours": hold_in_position_grace_hours,
                "hold_in_position_penalty_coef": hold_in_position_penalty_coef,
                "hold_in_position_penalty_growth": hold_in_position_penalty_growth,
                "hold_in_position_penalty_cap": hold_in_position_penalty_cap,
                "hold_missed_move_coef": hold_missed_move_coef,
                "entry_quality_coef": entry_quality_coef,
                "entry_quality_threshold": entry_quality_threshold,
                "entry_quality_noise_scale": entry_quality_noise_scale,
                "small_close_pnl_threshold": small_close_pnl_threshold,
                "small_close_pnl_penalty_coef": small_close_pnl_penalty_coef,
                "good_close_pnl_threshold": good_close_pnl_threshold,
                "good_close_bonus_coef": good_close_bonus_coef,
                "breakout_lookbacks": breakout_lookbacks,
                "breakout_alignment_bonus_coef": breakout_alignment_bonus_coef,
                "breakout_misaligned_penalty_coef": breakout_misaligned_penalty_coef,
                "breakout_strength_bonus_coef": breakout_strength_bonus_coef,
                "breakout_selector_bonus_coef": breakout_selector_bonus_coef,
                "breakout_entry_strength_quantile": breakout_entry_strength_quantile,
                "breakout_strength_quantile_window": breakout_strength_quantile_window,
                "breakout_block_misaligned_entries": breakout_block_misaligned_entries,
                "breakout_block_weak_entries": breakout_block_weak_entries,
                "atr_stop_multiple": atr_stop_multiple,
                "hard_time_stop_hours": hard_time_stop_hours,
                "reward_shaping_weight": reward_shaping_weight,
                "turnover_penalty_coef": turnover_penalty_coef,
                "breakout_prior_n": breakout_prior_n,
                "breakout_prior_bonus_coef": breakout_prior_bonus_coef,
                "enable_zones": bool(int(os.getenv('RL_ENABLE_ZONES', '1'))),
                "zone_tol": float(os.getenv('RL_ZONE_TOL', '0.002')),
                "zone_lookback": int(os.getenv('RL_ZONE_LOOKBACK', '5')),
                "regime_hard_stop": False,
                "enforce_directional_rebalance": True,
                "rebalance_after_entries": 40,
                "rebalance_dominance_threshold": 0.70,
                "directional_balance_reward_coef": 0.001,
                **shared_env_kwargs,
            }
            with open("models/env_config.json", "w", encoding="utf-8") as f:
                json.dump(env_config, f, indent=2)
            # env.save("models/vecnormalize.pkl")  # VecNormalize disabled
            print_step_summary("model_save", status="OK", 
                             files=["models/ppo_trading.zip", "models/env_config.json"])
            # Post-training: generate actions and analyze
            if skip_post_analysis:
                print_step_summary("post_analysis", status="SKIPPED")
            else:
                try:
                    from backtest.generate_actions import generate_actions
                    from backtest.run_analysis import run_analysis_and_organize

                    print("\n[POST] Generating actions and post-training analysis (TRAIN + TEST)...")
                    generate_actions(
                        model_path="models/ppo_trading.zip",
                        output_path="backtest/data/xauusd_actions_train.csv",
                        allow_hold_when_flat=True,
                        df_override=df_train,
                        env_config=env_config,
                    )
                    generate_actions(
                        model_path="models/ppo_trading.zip",
                        output_path="backtest/data/xauusd_actions.csv",
                        allow_hold_when_flat=True,
                        df_override=df_test,
                        env_config=env_config,
                    )
                    generate_actions(
                        model_path="models/ppo_trading.zip",
                        output_path="backtest/data/xauusd_actions_test.csv",
                        allow_hold_when_flat=True,
                        df_override=df_test,
                        env_config=env_config,
                    )
                    run_analysis_and_organize(
                        actions_csv="backtest/data/xauusd_actions_train.csv",
                        initial_capital=10000.0,
                        save_dir="backtest/plots_in_sample/latest"
                    )
                    run_analysis_and_organize(
                        actions_csv="backtest/data/xauusd_actions.csv",
                        initial_capital=10000.0,
                        save_dir="backtest/plots_oos/latest"
                    )
                    run_edge_scan = os.getenv("RL_RUN_EDGE_SCAN", "1").strip() == "1"
                    if run_edge_scan:
                        from backtest.edge_scanner import run_edge_scanner
                        run_edge_scanner(
                            train_end=train_end,
                            val_end=val_end,
                            output_dir="backtest/plots_oos/edge_scan",
                        )
                    print_step_summary("post_analysis", status="OK")
                except Exception as e:
                    error_msg = log_error(e, "post_analysis", critical=False)
                    errors.append(error_msg)
        except Exception as e:
            error_msg = log_error(e, "model_save", critical=False)
            errors.append(error_msg)
            print("[WARN] Partial save failed")
        
        try:
            env.close()
        except:
            pass
        
        status = "SUCCESS"
        
    except Exception as e:
        error_msg = log_error(e, "train_agent_main", critical=True)
        errors.append(error_msg)
        status = "FAILURE"
    
    print_final_report(status=status, errors=errors)
