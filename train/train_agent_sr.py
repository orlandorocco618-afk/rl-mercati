#!/usr/bin/env python3
"""
Train support/resistance (zone-based) RL agent.

This is a variant of train_agent.py focused exclusively on support/resistance
detection and zone-based features. Breakout features are DISABLED to keep the
focus on zone strategies. Uses the same dataset pipeline and multi-seed methodology.

Environment variables control hyperparameters and dataset splits.
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

from utils.data_utils import merge_h1_with_d1_regimes, load_h1_simple
from utils.kalman import apply_kalman_filter
from utils.error_handler import log_error, print_step_summary, print_final_report
from env.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ============================================================
# CONFIG & PARAMETERS
# ============================================================

CHECKPOINT_DIR = "models/checkpoints_sr"
MODELS_DIR = Path("models")
MODEL_OUTPUT = "models/ppo_sr_zones.zip"
ENV_CONFIG_OUTPUT = "models/env_config_sr.json"

# Training parameters from env vars
train_timesteps = int(os.getenv("RL_TRAIN_TIMESTEPS", "250000"))
seed = int(os.getenv("RL_SEED", "11"))
learning_rate = float(os.getenv("RL_LEARNING_RATE", "0.0001"))
batch_size = int(os.getenv("RL_BATCH_SIZE", "256"))
n_steps = int(os.getenv("RL_N_STEPS", "2048"))
gamma = float(os.getenv("RL_GAMMA", "0.99"))
gae_lambda = float(os.getenv("RL_GAE_LAMBDA", "0.95"))
clip_range = float(os.getenv("RL_CLIP_RANGE", "0.2"))
ent_coef = float(os.getenv("RL_ENT_COEF", "0.01"))
vf_coef = float(os.getenv("RL_VF_COEF", "0.5"))
max_grad_norm = float(os.getenv("RL_MAX_GRAD_NORM", "0.5"))
checkpoint_freq = int(os.getenv("RL_CHECKPOINT_FREQ", "50000"))
skip_post_analysis = bool(int(os.getenv("RL_SKIP_POST_ANALYSIS", "0")))

train_end = os.getenv("RL_TRAIN_END", "2022-12-31 23:00:00")
val_end = os.getenv("RL_VAL_END", "2023-12-31 23:00:00")

# Zone-specific parameters
enable_zones = bool(int(os.getenv("RL_ENABLE_ZONES", "1")))
zone_tol = float(os.getenv("RL_ZONE_TOL", "0.002"))
zone_lookback = int(os.getenv("RL_ZONE_LOOKBACK", "5"))

# Environment parameters (shared with breakout agent but tuned for zones)
reward_shaping_weight = float(os.getenv("RL_REWARD_SHAPING_WEIGHT", "0.12"))
target_active_steps_pct = float(os.getenv("RL_TARGET_ACTIVE_STEPS_PCT", "0.15"))
target_active_steps_tolerance = float(os.getenv("RL_TARGET_ACTIVE_STEPS_TOLERANCE", "0.03"))
active_target_reward_coef = float(os.getenv("RL_ACTIVE_TARGET_REWARD_COEF", "0.004"))
activity_hold_penalty_coef = float(os.getenv("RL_ACTIVITY_HOLD_PENALTY_COEF", "0.004"))
hold_penalty_base = float(os.getenv("RL_HOLD_PENALTY_BASE", "0.000008"))
hold_penalty_growth = float(os.getenv("RL_HOLD_PENALTY_GROWTH", "0.045"))
hold_penalty_cap = float(os.getenv("RL_HOLD_PENALTY_CAP", "0.0015"))
hold_in_position_grace_hours = int(os.getenv("RL_HOLD_IN_POSITION_GRACE_HOURS", "10"))
hold_in_position_penalty_coef = float(os.getenv("RL_HOLD_IN_POSITION_PENALTY_COEF", "0.00035"))
hold_in_position_penalty_growth = float(os.getenv("RL_HOLD_IN_POSITION_PENALTY_GROWTH", "0.045"))
hold_in_position_penalty_cap = float(os.getenv("RL_HOLD_IN_POSITION_PENALTY_CAP", "0.006"))
hold_missed_move_coef = float(os.getenv("RL_HOLD_MISSED_MOVE_COEF", "0.8"))
entry_quality_coef = float(os.getenv("RL_ENTRY_QUALITY_COEF", "0.0"))
entry_quality_threshold = float(os.getenv("RL_ENTRY_QUALITY_THRESHOLD", "0.15"))
entry_quality_noise_scale = float(os.getenv("RL_ENTRY_QUALITY_NOISE_SCALE", "0.5"))
small_close_pnl_threshold = float(os.getenv("RL_SMALL_CLOSE_PNL_THRESHOLD", "0.00008"))
small_close_pnl_penalty_coef = float(os.getenv("RL_SMALL_CLOSE_PNL_PENALTY_COEF", "0.0"))
good_close_pnl_threshold = float(os.getenv("RL_GOOD_CLOSE_PNL_THRESHOLD", "0.00025"))
good_close_bonus_coef = float(os.getenv("RL_GOOD_CLOSE_BONUS_COEF", "0.0"))
reverse_action_penalty_coef = float(os.getenv("RL_REVERSE_ACTION_PENALTY_COEF", "0.002"))
min_trade_duration_hours = int(os.getenv("RL_MIN_TRADE_DURATION_HOURS", "0"))
preferred_max_trade_duration_hours = int(os.getenv("RL_PREFERRED_MAX_TRADE_DURATION_HOURS", "72"))
hard_max_trade_duration_hours = int(os.getenv("RL_HARD_MAX_TRADE_DURATION_HOURS", "96"))
trade_duration_bonus_coef = float(os.getenv("RL_TRADE_DURATION_BONUS_COEF", "0.001"))
overhold_penalty_coef = float(os.getenv("RL_OVERHOLD_PENALTY_COEF", "0.0040"))
atr_stop_multiple = float(os.getenv("RL_ATR_STOP_MULTIPLE", "2.5"))
hard_time_stop_hours = int(os.getenv("RL_HARD_TIME_STOP_HOURS", "36"))
turnover_penalty_coef = float(os.getenv("RL_TURNOVER_PENALTY_COEF", "0.00035"))
enable_dynamic_sizing = bool(int(os.getenv("RL_ENABLE_DYNAMIC_SIZING", "1")))
dynamic_reference_window = int(os.getenv("RL_DYNAMIC_REFERENCE_WINDOW", "96"))
dynamic_size_floor = float(os.getenv("RL_DYNAMIC_SIZE_FLOOR", "0.70"))
dynamic_size_ceiling = float(os.getenv("RL_DYNAMIC_SIZE_CEILING", "1.35"))
dynamic_confidence_power = float(os.getenv("RL_DYNAMIC_CONFIDENCE_POWER", "1.10"))
dynamic_volatility_weight = float(os.getenv("RL_DYNAMIC_VOLATILITY_WEIGHT", "0.40"))
dynamic_signal_weight = float(os.getenv("RL_DYNAMIC_SIGNAL_WEIGHT", "0.15"))
dynamic_zone_weight = float(os.getenv("RL_DYNAMIC_ZONE_WEIGHT", "0.45"))
dynamic_breakout_weight = float(os.getenv("RL_DYNAMIC_BREAKOUT_WEIGHT", "0.0"))
enable_confidence_gate = bool(int(os.getenv("RL_ENABLE_CONFIDENCE_GATE", "0")))
confidence_gate_threshold = float(os.getenv("RL_CONFIDENCE_GATE_THRESHOLD", "0.60"))
drawdown_penalty_coef = float(os.getenv("RL_DRAWDOWN_PENALTY_COEF", "0.0008"))
drawdown_penalty_threshold = float(os.getenv("RL_DRAWDOWN_PENALTY_THRESHOLD", "0.025"))
low_confidence_open_penalty_coef = float(os.getenv("RL_LOW_CONFIDENCE_OPEN_PENALTY_COEF", "0.00035"))
high_confidence_open_bonus_coef = float(os.getenv("RL_HIGH_CONFIDENCE_OPEN_BONUS_COEF", "0.00015"))
confidence_reward_threshold = float(os.getenv("RL_CONFIDENCE_REWARD_THRESHOLD", "0.58"))


def make_env(df, allow_hold_when_flat=True):
    """Factory to create TradingEnv with zone focus (no breakout)."""
    def _init():
        return TradingEnv(
            df=df,
            window_size=100,
            initial_capital=10000.0,
            max_drawdown=0.30,
            allow_hold_when_flat=allow_hold_when_flat,
            random_start_on_reset=True,
            min_episode_steps=1500,
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
            reverse_action_penalty_coef=reverse_action_penalty_coef,
            min_trade_duration_hours=min_trade_duration_hours,
            preferred_max_trade_duration_hours=preferred_max_trade_duration_hours,
            hard_max_trade_duration_hours=hard_max_trade_duration_hours,
            trade_duration_bonus_coef=trade_duration_bonus_coef,
            overhold_penalty_coef=overhold_penalty_coef,
            atr_stop_multiple=atr_stop_multiple,
            hard_time_stop_hours=hard_time_stop_hours,
            reward_shaping_weight=reward_shaping_weight,
            turnover_penalty_coef=turnover_penalty_coef,
            enable_dynamic_sizing=enable_dynamic_sizing,
            dynamic_reference_window=dynamic_reference_window,
            dynamic_size_floor=dynamic_size_floor,
            dynamic_size_ceiling=dynamic_size_ceiling,
            dynamic_confidence_power=dynamic_confidence_power,
            dynamic_volatility_weight=dynamic_volatility_weight,
            dynamic_signal_weight=dynamic_signal_weight,
            dynamic_zone_weight=dynamic_zone_weight,
            dynamic_breakout_weight=dynamic_breakout_weight,
            enable_confidence_gate=enable_confidence_gate,
            confidence_gate_threshold=confidence_gate_threshold,
            drawdown_penalty_coef=drawdown_penalty_coef,
            drawdown_penalty_threshold=drawdown_penalty_threshold,
            low_confidence_open_penalty_coef=low_confidence_open_penalty_coef,
            high_confidence_open_bonus_coef=high_confidence_open_bonus_coef,
            confidence_reward_threshold=confidence_reward_threshold,
            # ZONE-FOCUSED: NO BREAKOUT, ZONES ENABLED
            breakout_lookbacks=[],  # disabled
            enable_zones=enable_zones,
            zone_tol=zone_tol,
            zone_lookback=zone_lookback,
        )
    return _init


def load_data(train_end_str, val_end_str):
    """Load and prepare training/validation dataset."""
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col="Close")
    df["Close"] = df["Close_KF"]
    
    # Ensure Time column is datetime and properly sorted
    if "Time" not in df.columns:
        raise ValueError("Column Time missing for temporal split")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    
    # Split by date
    train_end_dt = pd.to_datetime(train_end_str)
    val_end_dt = pd.to_datetime(val_end_str)
    df_train = df[df["Time"] <= train_end_dt].copy()
    df_test = df[(df["Time"] > train_end_dt) & (df["Time"] <= val_end_dt)].copy()
    
    return df_train, df_test


def main():
    print("\n" + "="*60)
    print("[SR AGENT] Support/Resistance Zone-Based RL Training")
    print("="*60)
    print(f"Seed: {seed}, Timesteps: {train_timesteps}, LR: {learning_rate}")
    print(f"Zones: enabled={enable_zones}, tol={zone_tol}, lookback={zone_lookback}")
    print(
        f"Dynamic sizing: enabled={enable_dynamic_sizing}, floor={dynamic_size_floor:.2f}, "
        f"ceiling={dynamic_size_ceiling:.2f}, gate={enable_confidence_gate}"
    )
    print(f"No breakout features (edge: S/R zones only)")
    print("="*60)

    errors = []

    # Load data
    try:
        print("\n[STEP 1/4] Loading and preparing data...")
        df_train, df_test = load_data(train_end, val_end)
        print(f"  Train rows: {len(df_train)}, Test rows: {len(df_test)}")
        print_step_summary("data_load", status="OK", 
                         info=f"train={len(df_train)}, test={len(df_test)}")
    except Exception as e:
        error_msg = log_error(e, "data_load", critical=True)
        errors.append(error_msg)
        raise

    # Create environment
    try:
        print("\n[STEP 2/4] Initializing TradingEnv (zones-focused, no breakout)...")
        env = DummyVecEnv([make_env(df_train, allow_hold_when_flat=True)])
        obs_dim = env.observation_space.shape[0]
        print(f"  Observation space: {obs_dim} dims (base 15 + zones 4 + breakout 0)")
        print_step_summary("env_init", status="OK", 
                         info=f"obs_dim={obs_dim}")
    except Exception as e:
        error_msg = log_error(e, "env_init", critical=True)
        errors.append(error_msg)
        raise

    # Train model
    try:
        print("\n[STEP 3/4] Training PPO agent on zone features...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=1,
            seed=seed,
            device="auto",
            tensorboard_log="logs_sr",
        )
        
        model.learn(
            total_timesteps=train_timesteps,
            callback=None,
            log_interval=5,
            tb_log_name=f"ppo_sr_zones_seed{seed}",
        )
        
        print_step_summary("model_training", status="OK", timesteps=train_timesteps)
    except Exception as e:
        error_msg = log_error(e, "model_training", critical=True)
        errors.append(error_msg)
        raise

    # Save model and config
    try:
        print("\n[STEP 4/4] Saving model and configuration...")
        os.makedirs("models", exist_ok=True)
        model.save(MODEL_OUTPUT)
        
        env_config = {
            "window_size": 100,
            "initial_capital": 10000.0,
            "max_drawdown": 0.30,
            "allow_hold_when_flat": True,
            "random_start_on_reset": True,
            "min_episode_steps": 1500,
            "target_active_steps_pct": target_active_steps_pct,
            "target_active_steps_tolerance": target_active_steps_tolerance,
            "active_target_reward_coef": active_target_reward_coef,
            "activity_hold_penalty_coef": activity_hold_penalty_coef,
            "hold_penalty_base": hold_penalty_base,
            "hold_penalty_growth": hold_penalty_growth,
            "hold_penalty_cap": hold_penalty_cap,
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
            "reverse_action_penalty_coef": reverse_action_penalty_coef,
            "min_trade_duration_hours": min_trade_duration_hours,
            "preferred_max_trade_duration_hours": preferred_max_trade_duration_hours,
            "hard_max_trade_duration_hours": hard_max_trade_duration_hours,
            "trade_duration_bonus_coef": trade_duration_bonus_coef,
            "overhold_penalty_coef": overhold_penalty_coef,
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
            # ZONE-SPECIFIC CONFIG
            "breakout_lookbacks": [],
            "enable_zones": enable_zones,
            "zone_tol": zone_tol,
            "zone_lookback": zone_lookback,
            # System markers
            "agent_type": "sr_zones",
            "trained_seed": seed,
            "trained_timesteps": train_timesteps,
        }
        
        with open(ENV_CONFIG_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(env_config, f, indent=2)
        
        print_step_summary("model_save", status="OK",
                         files=[MODEL_OUTPUT, ENV_CONFIG_OUTPUT])
    except Exception as e:
        error_msg = log_error(e, "model_save", critical=True)
        errors.append(error_msg)
        raise

    if not errors:
        print("\n" + "="*60)
        print("[OK] SR Agent training completed successfully")
        print(f"  Model: {MODEL_OUTPUT}")
        print(f"  Config: {ENV_CONFIG_OUTPUT}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("[ERROR] Training encountered errors:")
        for err in errors:
            print(f"  - {err}")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
