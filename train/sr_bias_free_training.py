#!/usr/bin/env python3
"""Bias-aware SR training run that uses updated rewards, large CSV coverage, and downstream validation."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, ".")

from backtest.analyze_actions_v2 import analyze_actions_v2
from backtest.out_of_sample_test import _run_slice
from env.trading_env import TradingEnv
from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter


MODEL_OUTPUT = Path("models/ppo_sr_bias_free.zip")
ENV_CONFIG_OUTPUT = Path("models/env_config_sr_bias_free.json")
OUTPUT_DIR = Path("backtest/plots_oos/sr_bias_free")

TRAIN_END = "2025-06-30 23:00:00"
VAL_END = "2025-12-31 23:00:00"
TRAIN_TIMESTEPS = 50_000
SEED = 22
SHORT_GAP_PENALTY_COEF = 0.0005
SHORT_GAP_THRESHOLD = 48
SHORT_REWARD_BONUS_COEF = 0.004
LONG_REWARD_BONUS_COEF = 0.004


def load_data(train_end: str, val_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col="Close")
    df["Close"] = df["Close_KF"]
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    train_df = df[df["Time"] <= pd.to_datetime(train_end)].copy()
    test_df = df[(df["Time"] > pd.to_datetime(train_end)) & (df["Time"] <= pd.to_datetime(val_end))].copy()
    return train_df, test_df


def make_env(
    df: pd.DataFrame,
    allow_hold_when_flat: bool,
    random_start: bool,
    seed: int,
) -> Any:
    def _init():
        return TradingEnv(
            df=df,
            window_size=150,
            initial_capital=10000.0,
            max_drawdown=0.30,
            random_start_on_reset=random_start,
            min_episode_steps=1800,
            allow_hold_when_flat=allow_hold_when_flat,
            target_active_steps_pct=0.12,
            target_active_steps_tolerance=0.06,
            active_target_reward_coef=0.01,
            activity_hold_penalty_coef=0.002,
            hold_penalty_base=0.00005,
            hold_penalty_growth=0.03,
            hold_penalty_cap=0.002,
            hold_in_position_grace_hours=12,
            hold_in_position_penalty_coef=0.0001,
            hold_in_position_penalty_growth=0.02,
            hold_in_position_penalty_cap=0.003,
            hold_missed_move_coef=0.25,
            reverse_action_penalty_coef=0.002,
            min_trade_duration_hours=6,
            preferred_max_trade_duration_hours=36,
            hard_max_trade_duration_hours=60,
            trade_duration_bonus_coef=0.001,
            overhold_penalty_coef=0.003,
            atr_stop_multiple=2.5,
            hard_time_stop_hours=48,
            reward_shaping_weight=0.08,
            turnover_penalty_coef=0.0002,
            entry_quality_coef=0.0,
            entry_quality_threshold=0.0,
            entry_quality_noise_scale=0.5,
            breakout_lookbacks=[],
            enable_zones=True,
            zone_tol=0.003,
            zone_lookback=5,
            short_gap_penalty_coef=SHORT_GAP_PENALTY_COEF,
            short_gap_threshold=SHORT_GAP_THRESHOLD,
            short_reward_bonus_coef=SHORT_REWARD_BONUS_COEF,
            long_reward_bonus_coef=LONG_REWARD_BONUS_COEF,
        )

    return _init


def build_env_config() -> dict[str, Any]:
    return {
        "window_size": 150,
        "initial_capital": 10000.0,
        "max_drawdown": 0.30,
        "target_active_steps_pct": 0.12,
        "target_active_steps_tolerance": 0.06,
        "active_target_reward_coef": 0.01,
        "activity_hold_penalty_coef": 0.0,
        "hold_penalty_base": 0.0,
        "hold_penalty_growth": 0.02,
        "hold_penalty_cap": 0.0,
        "hold_in_position_grace_hours": 12,
        "hold_in_position_penalty_coef": 0.0001,
        "hold_in_position_penalty_growth": 0.02,
        "hold_in_position_penalty_cap": 0.003,
        "hold_missed_move_coef": 0.25,
        "reverse_action_penalty_coef": 0.002,
        "min_trade_duration_hours": 6,
        "preferred_max_trade_duration_hours": 36,
        "hard_max_trade_duration_hours": 60,
        "trade_duration_bonus_coef": 0.001,
        "overhold_penalty_coef": 0.003,
        "atr_stop_multiple": 2.5,
        "hard_time_stop_hours": 48,
        "reward_shaping_weight": 0.08,
        "turnover_penalty_coef": 0.0002,
        "entry_quality_coef": 0.0,
        "entry_quality_threshold": 0.0,
        "entry_quality_noise_scale": 0.5,
        "breakout_lookbacks": [],
        "enable_zones": True,
        "zone_tol": 0.003,
        "zone_lookback": 5,
        "short_gap_penalty_coef": SHORT_GAP_PENALTY_COEF,
        "short_gap_threshold": SHORT_GAP_THRESHOLD,
        "short_reward_bonus_coef": SHORT_REWARD_BONUS_COEF,
        "long_reward_bonus_coef": LONG_REWARD_BONUS_COEF,
        "agent_type": "sr_bias_free",
        "trained_seed": SEED,
        "trained_timesteps": TRAIN_TIMESTEPS,
    }


def evaluate_model(model: PPO, df_train: pd.DataFrame, df_test: pd.DataFrame, env_config: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env_cfg = dict(env_config)
    training_actions, train_metrics = _run_slice(model, df_train, env_cfg, allow_hold_when_flat=True, label="train_bias_free")
    test_actions, test_metrics = _run_slice(model, df_test, env_cfg, allow_hold_when_flat=True, label="test_bias_free")

    train_csv = OUTPUT_DIR / "train_bias_actions.csv"
    test_csv = OUTPUT_DIR / "test_bias_actions.csv"
    training_actions.to_csv(train_csv, index=False)
    test_actions.to_csv(test_csv, index=False)

    analyze_actions_v2(actions_csv=str(train_csv), save_dir=str(OUTPUT_DIR / "train_bias_dashboard"))
    analyze_actions_v2(actions_csv=str(test_csv), save_dir=str(OUTPUT_DIR / "test_bias_dashboard"))

    report = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "train_html": str(OUTPUT_DIR / "train_bias_dashboard" / "dashboard.html"),
        "test_html": str(OUTPUT_DIR / "test_bias_dashboard" / "dashboard.html"),
        "env_config": env_config,
    }
    report_path = OUTPUT_DIR / "bias_free_validation.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nBias-aware validation saved: {report_path}")


def main() -> None:
    os.makedirs("models", exist_ok=True)
    df_train, df_test = load_data(TRAIN_END, VAL_END)
    train_env = DummyVecEnv([make_env(df_train, True, True, SEED)])
    eval_env = DummyVecEnv([make_env(df_test, True, False, SEED)])

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.0, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path="models/checkpoints_sr_bias_free",
        log_path="logs_sr_bias_free",
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=SEED,
        tensorboard_log="logs_sr_bias_free",
        device="auto",
    )

    print(f"Training bias-aware SR agent (seed {SEED}, timesteps {TRAIN_TIMESTEPS})...")
    model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=eval_callback)

    model.save(MODEL_OUTPUT)
    env_config = build_env_config()
    with ENV_CONFIG_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(env_config, f, indent=2)
    print(f"Model saved: {MODEL_OUTPUT}")
    print(f"Env config saved: {ENV_CONFIG_OUTPUT}")

    evaluate_model(model, df_train, df_test, env_config)


if __name__ == "__main__":
    main()
