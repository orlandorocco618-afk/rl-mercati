"""
Out-of-sample validation with strict temporal split.
- Train slice: <= TRAIN_END
- Test slice:  > VAL_END
Evaluates the same model on both slices with identical env config.
"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, ".")

from env.trading_env import TradingEnv
from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter
from backtest.analyze_actions_v2 import analyze_actions_v2

BEST_MAIN_MODEL = "models/ppo_trading.zip"
BEST_PHASE2_MODEL = "models/checkpoints_phase2/ppo_phase2_best.zip"
BEST_BOOTSTRAP_MODEL = "models/checkpoints_iter/ppo_bootstrap_best.zip"

VALIDATION_THRESHOLDS = {
    "return_drop_tolerance_pct": 10.0,
    "winrate_drop_tolerance_pct": 20.0,
    "hard_fail_return_collapse_pct": -50.0,
}


def get_best_model_path():
    if Path(BEST_MAIN_MODEL).exists():
        return BEST_MAIN_MODEL, "main"
    if Path(BEST_PHASE2_MODEL).exists():
        return BEST_PHASE2_MODEL, "phase2"
    if Path(BEST_BOOTSTRAP_MODEL).exists():
        return BEST_BOOTSTRAP_MODEL, "bootstrap"
    return None, None


def split_train_test_by_date(
    df: pd.DataFrame,
    train_end: str = "2022-12-31 23:00:00",
    val_end: str = "2023-12-31 23:00:00",
):
    out = df.copy()
    out["Time"] = pd.to_datetime(out["Time"], errors="coerce")
    out = out.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    if val_end_ts <= train_end_ts:
        raise ValueError("val_end must come after train_end")

    df_train = out.loc[out["Time"] <= train_end_ts].reset_index(drop=True)
    df_test = out.loc[out["Time"] > val_end_ts].reset_index(drop=True)
    return df_train, df_test


def _load_env_config(config_path: str = "models/env_config.json") -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def make_env(df, allow_hold_when_flat: bool, env_config: dict | None = None):
    def _init():
        cfg = dict(env_config or {})
        cfg.update(
            {
                "df": df,
                "max_drawdown": 2.0,
                "allow_hold_when_flat": allow_hold_when_flat,
                "regime_hard_stop": False,
                "enforce_directional_rebalance": False,
                "directional_balance_reward_coef": 0.0,
                "random_start_on_reset": False,
            }
        )
        # Filter out metadata fields that aren't TradingEnv parameters
        excluded_keys = {'agent_type', 'trained_seed', 'trained_timesteps'}
        cfg = {k: v for k, v in cfg.items() if k not in excluded_keys}
        return TradingEnv(**cfg)

    return _init


def _infer_allow_hold_from_model_path(model_path: str, model_source: str) -> bool:
    return True


def _extract_model_actions(action):
    arr = np.asarray(action)
    flat = arr.reshape(-1)
    if flat.size >= 2:
        return int(flat[0]), int(flat[1])
    if flat.size == 1:
        return int(flat[0]), -1
    try:
        return int(action), -1
    except Exception:
        return 0, -1


def _run_slice(model, df_slice: pd.DataFrame, env_cfg: dict, allow_hold_when_flat: bool, label: str):
    env = DummyVecEnv([make_env(df_slice, allow_hold_when_flat=allow_hold_when_flat, env_config=env_cfg)])
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    done = False
    steps = 0
    episode_return = 0.0
    actions_rows = []

    position = 0
    entry_price = None
    closed_trade_pnls = []
    closed_trade_durations = []  # hours
    equity_curve = []

    while not done and steps < max(1, len(df_slice) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done_raw, info = env.step(action)

        done = bool(done_raw[0]) if isinstance(done_raw, (np.ndarray, list, tuple)) else bool(done_raw)
        info_dict = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else (info if isinstance(info, dict) else {})

        model_action_val, model_breakout_idx = _extract_model_actions(action)
        action_val = int(info_dict.get("executed_action", model_action_val))
        reward_val = float(rewards[0]) if isinstance(rewards, (np.ndarray, list, tuple)) else float(rewards)

        env_obj = env.envs[0]
        executed_step = int(info_dict.get("executed_step", max(0, int(getattr(env_obj, "current_step", 0)) - 1)))
        row_idx = min(max(executed_step, 0), len(df_slice) - 1)
        row = df_slice.iloc[row_idx]

        execution_price = pd.to_numeric(info_dict.get("execution_price", np.nan), errors="coerce")
        if pd.isna(execution_price):
            execution_price = pd.to_numeric(row.get("Open", row.get("Close", np.nan)), errors="coerce")

        equity_val = float(info_dict.get("equity", getattr(env_obj, "equity", np.nan)))
        equity_curve.append(equity_val)
        episode_return += reward_val

        actions_rows.append(
            {
                "Time": row.get("Time"),
                "Open": row.get("Open", np.nan),
                "High": row.get("High", np.nan),
                "Low": row.get("Low", np.nan),
                "Close": row.get("Close", np.nan),
                "Regime": int(pd.to_numeric(row.get("REGIME", 2), errors="coerce")),
                "Action": action_val,
                "ModelAction": model_action_val,
                "ModelBreakoutIdx": model_breakout_idx,
                "SelectedBreakoutIdx": int(info_dict.get("selected_breakout_idx", model_breakout_idx)),
                "SelectedBreakoutN": int(info_dict.get("selected_breakout_n", 0)),
                "BreakoutSignal": int(info_dict.get("selected_breakout_signal", 0)),
                "BreakoutStrength": float(info_dict.get("selected_breakout_strength", 0.0)),
                "Reward": reward_val,
                "Position": int(info_dict.get("position", getattr(env_obj, "position", 0))),
                "Equity": equity_val,
                "DangerLevel": int(info_dict.get("danger_level", 0)),
                "PositionSize": float(info_dict.get("position_size", np.nan)),
                "BasePositionSizePreDynamic": float(info_dict.get("base_position_size_pre_dynamic", np.nan)),
                "DynamicSizeMultiplier": float(info_dict.get("dynamic_size_multiplier", np.nan)),
                "SetupConfidence": float(info_dict.get("setup_confidence", np.nan)),
                "VolatilityConfidence": float(info_dict.get("volatility_confidence", np.nan)),
                "SignalConfidence": float(info_dict.get("signal_confidence", np.nan)),
                "ZoneConfidence": float(info_dict.get("zone_confidence", np.nan)),
                "BreakoutConfidence": float(info_dict.get("breakout_confidence", np.nan)),
                "ConfidenceGateBlocked": bool(info_dict.get("confidence_gate_blocked", False)),
                "ForcedAction": bool(info_dict.get("forced_action", False)),
                "HardStop": bool(info_dict.get("hard_stop_applied", False)),
            }
        )

        if not pd.isna(execution_price):
            px = float(execution_price)
            if action_val == 1:
                if position <= 0:
                    if position == -1 and entry_price is not None:
                        closed_trade_pnls.append((entry_price - px) / max(1e-9, entry_price))
                        # record duration: use info if available
                        dur = info_dict.get("trade_duration_hours")
                        if dur is not None:
                            try:
                                closed_trade_durations.append(int(dur))
                            except Exception:
                                pass
                    position = 1
                    entry_price = px
            elif action_val == 2:
                if position >= 0:
                    if position == 1 and entry_price is not None:
                        closed_trade_pnls.append((px - entry_price) / max(1e-9, entry_price))
                        dur = info_dict.get("trade_duration_hours")
                        if dur is not None:
                            try:
                                closed_trade_durations.append(int(dur))
                            except Exception:
                                pass
                    position = -1
                    entry_price = px
            elif action_val == 3:
                if position == 1 and entry_price is not None:
                    closed_trade_pnls.append((px - entry_price) / max(1e-9, entry_price))
                    dur = info_dict.get("trade_duration_hours")
                    if dur is not None:
                        try:
                            closed_trade_durations.append(int(dur))
                        except Exception:
                            pass
                elif position == -1 and entry_price is not None:
                    closed_trade_pnls.append((entry_price - px) / max(1e-9, entry_price))
                    dur = info_dict.get("trade_duration_hours")
                    if dur is not None:
                        try:
                            closed_trade_durations.append(int(dur))
                        except Exception:
                            pass
                position = 0
                entry_price = None

        steps += 1
        if steps % 2000 == 0:
            print(f"  [{label}] progress: {steps}/{len(df_slice)}")

    if position != 0 and entry_price is not None and len(df_slice) > 0:
        last_px = pd.to_numeric(df_slice.iloc[-1].get("Close"), errors="coerce")
        if not pd.isna(last_px):
            lp = float(last_px)
            if position == 1:
                closed_trade_pnls.append((lp - entry_price) / max(1e-9, entry_price))
            elif position == -1:
                closed_trade_pnls.append((entry_price - lp) / max(1e-9, entry_price))

    final_capital = float(equity_curve[-1]) if equity_curve else 10000.0
    total_return_pct = 100.0 * (final_capital - 10000.0) / 10000.0
    if equity_curve:
        eq = np.array(equity_curve, dtype=float)
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / (running_max + 1e-9) * 100.0
        max_drawdown_pct = float(np.min(dd))
    else:
        max_drawdown_pct = 0.0

    total_trades = len(closed_trade_pnls)
    win_rate = (100.0 * sum(1 for x in closed_trade_pnls if x > 0) / total_trades) if total_trades > 0 else 0.0
    avg_duration = float(np.mean(closed_trade_durations)) if closed_trade_durations else 0.0

    metrics = {
        "total_return_pct": float(total_return_pct),
        "pct_profitable": float(win_rate),
        "total_trades": int(total_trades),
        "avg_trade_duration_hours": avg_duration,
        "max_drawdown_pct": float(max_drawdown_pct),
        "steps": int(steps),
        "final_capital": float(final_capital),
        "episode_return": float(episode_return),
    }
    env.close()
    return pd.DataFrame(actions_rows), metrics


def run_out_of_sample_test():
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE TEST (STRICT TEMPORAL SPLIT)")
    print("=" * 80)

    os.makedirs("backtest/plots_oos", exist_ok=True)
    os.makedirs("backtest/data", exist_ok=True)

    train_end = os.getenv("RL_TRAIN_END", "2022-12-31 23:00:00").strip()
    val_end = os.getenv("RL_VAL_END", "2023-12-31 23:00:00").strip()

    print("[INIT] Loading data...")
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col="Close")
    df["Close"] = df["Close_KF"]

    df_train, df_test = split_train_test_by_date(df, train_end=train_end, val_end=val_end)
    print(f"  Train rows: {len(df_train)} (<= {train_end})")
    print(f"  Test rows:  {len(df_test)} (> {val_end})")

    model_path, model_source = get_best_model_path()
    if model_path is None:
        print("  FAIL - No model found")
        return False, "Best model not found"

    print(f"[INIT] Loading model ({model_source}): {model_path}")
    model = PPO.load(model_path)
    allow_hold_when_flat = _infer_allow_hold_from_model_path(model_path, model_source)
    env_cfg = _load_env_config()

    print("[RUN] Inference on the train slice...")
    train_actions, train_metrics = _run_slice(model, df_train, env_cfg, allow_hold_when_flat, label="train")
    train_csv = "backtest/data/xauusd_actions_train_eval.csv"
    train_actions.to_csv(train_csv, index=False)

    print("[RUN] Inference on the test slice...")
    test_actions, test_metrics = _run_slice(model, df_test, env_cfg, allow_hold_when_flat, label="test")
    test_csv = "backtest/data/xauusd_actions_test_eval.csv"
    test_actions.to_csv(test_csv, index=False)

    analyze_actions_v2(actions_csv=train_csv, initial_capital=10000.0, save_dir="backtest/plots_oos/train_eval")
    analyze_actions_v2(actions_csv=test_csv, initial_capital=10000.0, save_dir="backtest/plots_oos/test_eval")

    ret_train = float(train_metrics.get("total_return_pct", 0.0))
    ret_test = float(test_metrics.get("total_return_pct", 0.0))
    wr_train = float(train_metrics.get("pct_profitable", 0.0))
    wr_test = float(test_metrics.get("pct_profitable", 0.0))
    dd_train = float(train_metrics.get("max_drawdown_pct", 0.0))
    dd_test = float(test_metrics.get("max_drawdown_pct", 0.0))

    ret_delta = ret_test - ret_train
    ret_delta_pct = (ret_delta / max(abs(ret_train), 1e-12) * 100.0) if abs(ret_train) > 1e-12 else (0.0 if abs(ret_test) < 1e-12 else np.sign(ret_test) * 999.0)
    wr_delta = wr_test - wr_train
    dd_delta = dd_test - dd_train
    dur_delta = test_metrics.get("avg_trade_duration_hours",0.0) - train_metrics.get("avg_trade_duration_hours",0.0)

    reasons = []
    test_worse_than_train = ret_test < ret_train
    if (not test_worse_than_train) or (abs(ret_delta_pct) <= VALIDATION_THRESHOLDS["return_drop_tolerance_pct"]):
        reasons.append(f"[OK] return delta% {ret_delta_pct:+.1f} within threshold")
    else:
        reasons.append(f"[FAIL] return delta% {ret_delta_pct:+.1f} outside threshold")

    if abs(wr_delta) <= VALIDATION_THRESHOLDS["winrate_drop_tolerance_pct"]:
        reasons.append(f"[OK] winrate delta {wr_delta:+.1f} within threshold")
    else:
        reasons.append(f"[FAIL] winrate delta {wr_delta:+.1f} outside threshold")

    if test_worse_than_train and ret_delta_pct <= VALIDATION_THRESHOLDS["hard_fail_return_collapse_pct"]:
        assessment = "LIKELY OVERFITTING"
        reasons.append("[HARD FAIL] return collapse >50% (test worse than train)")
    elif ret_train <= 0.0 and ret_test <= 0.0:
        assessment = "NOT PROFITABLE (NO EDGE)"
        reasons.append("[FAIL] train and test both unprofitable")
    else:
        ok_count = sum(1 for x in reasons if x.startswith("[OK]"))
        assessment = "REAL PATTERN" if ok_count >= 2 else "LIKELY OVERFITTING"

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"TRAIN return={ret_train:+.2f}% | win={wr_train:.1f}% | dd={dd_train:.2f}% | trades={train_metrics['total_trades']} | avg dur={train_metrics.get('avg_trade_duration_hours',0.0):.1f}h")
    print(f"TEST  return={ret_test:+.2f}% | win={wr_test:.1f}% | dd={dd_test:.2f}% | trades={test_metrics['total_trades']} | avg dur={test_metrics.get('avg_trade_duration_hours',0.0):.1f}h")
    print(f"DELTA return={ret_delta:+.2f}% ({ret_delta_pct:+.1f}%) | win={wr_delta:+.1f}% | dd={dd_delta:+.2f}% | dur diff={dur_delta:+.1f}h")
    print(f"ASSESSMENT: {assessment}")
    for r in reasons:
        print(f"  {r}")

    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_source": model_source,
        "model_path": model_path,
        "split": {"train_end": train_end, "val_end": val_end},
        "train": train_metrics,
        "test": test_metrics,
        "deltas": {
            "return_pct": ret_delta,
            "return_pct_change": ret_delta_pct,
            "winrate_pct": wr_delta,
            "max_drawdown_pct": dd_delta,
            "avg_trade_duration_hours": dur_delta,
        },
        "assessment": assessment,
        "reasons": reasons,
        "thresholds": VALIDATION_THRESHOLDS,
    }

    report_path = Path("backtest/plots_oos/validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")

    return True, assessment


if __name__ == "__main__":
    success, assessment = run_out_of_sample_test()
    print("\n" + "=" * 80)
    if success:
        print(f"OK - Out-of-sample validation completed: {assessment}")
    else:
        print(f"FAIL - {assessment}")
    print("=" * 80 + "\n")
    sys.exit(0 if success else 1)
