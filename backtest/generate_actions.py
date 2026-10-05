import os
import json
import pandas as pd
import numpy as np
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.trading_env import TradingEnv
from utils.kalman import apply_kalman_filter
from utils.data_utils import merge_h1_with_d1_regimes
from utils.error_handler import (
    log_error, print_step_summary, print_final_report,
    check_file_exists, validate_dataframe
)


def load_data(
    h1_path: str = "data/xauusd_h1_clean.csv",
    d1_path: str = "data/xauusd_d1_clean.csv"
) -> pd.DataFrame:
    """
    Loads H1 for trading and D1 for the regime computation.
    """
    df = merge_h1_with_d1_regimes(h1_path, d1_path)
    return df


def _load_env_config(config_path: str = "models/env_config.json") -> dict:
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def make_env(df, allow_hold_when_flat: bool = True, env_config: dict | None = None):
    def _init():
        # Disable drawdown-based early stop to ensure full-CSV action export.
        cfg = dict(env_config or {})
        cfg.update(
            {
                "df": df,
                "max_drawdown": 2.0,
                "allow_hold_when_flat": allow_hold_when_flat,
                "regime_hard_stop": False,
                "enforce_directional_rebalance": False,
                "directional_balance_reward_coef": 0.0,
            }
        )
        return TradingEnv(
            **cfg
        )
    return _init


def _infer_allow_hold_from_model_path(model_path: str) -> bool:
    # Legacy hook kept for compatibility: HOLD is always allowed now.
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


def generate_actions(
    model_path: str,
    output_path: str,
    allow_hold_when_flat: bool | None = None,
    df_override: pd.DataFrame | None = None,
    env_config: dict | None = None,
):
    if df_override is None:
        print("Loading data...")
        df = load_data()
        from utils.kalman import apply_kalman_filter
        df = apply_kalman_filter(df)
        df["Close"] = df["Close_KF"]
    else:
        df = df_override.reset_index(drop=True).copy()

    if allow_hold_when_flat is None:
        allow_hold_when_flat = True

    loaded_cfg = _load_env_config()
    if env_config:
        loaded_cfg.update(env_config)

    print(f"Inference config: allow_hold_when_flat={allow_hold_when_flat}")

    print("Creating the RL environment...")
    env = DummyVecEnv([make_env(df, allow_hold_when_flat=allow_hold_when_flat, env_config=loaded_cfg)])

    print("Loading the RL model...")
    model = PPO.load(model_path)

    obs = env.reset()
    done_flag = False

    actions = []
    step = 0

    print("Generating RL actions...")

    while not done_flag:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if isinstance(done, (np.ndarray, list, tuple)):
            done_flag = bool(done[0])
        else:
            done_flag = bool(done)

        if isinstance(reward, (np.ndarray, list, tuple)):
            reward_val = float(reward[0])
        else:
            reward_val = float(reward)

        if isinstance(info, (list, tuple)) and len(info) > 0:
            info_dict = info[0] if isinstance(info[0], dict) else {}
        elif isinstance(info, dict):
            info_dict = info
        else:
            info_dict = {}

        internal_env = env.envs[0]
        executed_step = int(info_dict.get("executed_step", max(0, getattr(internal_env, "current_step", 0) - 1)))
        current_idx = min(max(executed_step, 0), len(df) - 1)
        current_row = df.iloc[current_idx]
        model_action_val, model_breakout_idx = _extract_model_actions(action)
        action_val = int(info_dict.get("executed_action", model_action_val))
        selected_breakout_idx = int(info_dict.get("selected_breakout_idx", model_breakout_idx))
        position_val = int(info_dict.get("position", getattr(internal_env, "position", 0)))
        equity_val = float(info_dict.get("equity", getattr(internal_env, "equity", np.nan)))

        actions.append({
            "Time": current_row["Time"],
            "Open": current_row["Open"],
            "High": current_row["High"],
            "Low": current_row["Low"],
            "Close": current_row["Close"],
            "Regime": int(current_row.get("REGIME", 2)),
            "Action": action_val,
            "ModelAction": model_action_val,
            "ModelBreakoutIdx": model_breakout_idx,
            "SelectedBreakoutIdx": selected_breakout_idx,
            "SelectedBreakoutN": int(info_dict.get("selected_breakout_n", 0)),
            "BreakoutSignal": int(info_dict.get("selected_breakout_signal", 0)),
            "BreakoutStrength": float(info_dict.get("selected_breakout_strength", 0.0)),
            "Reward": reward_val,
            "Position": position_val,
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
        })

        step += 1

    print(f"Saving actions to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(actions).to_csv(output_path, index=False)

    print("Done.")


if __name__ == "__main__":
    errors = []
    status = "FAILURE"
    
    try:
        print("="*60)
        print("[STEP 1/6] Loading H1+D1 data...")
        print("="*60)
        
        check_file_exists("data/xauusd_h1_clean.csv", critical=True)
        check_file_exists("data/xauusd_d1_clean.csv", critical=True)
        
        try:
            df = load_data()
            import os
            quick = os.getenv("RL_QUICK_TEST") == "1"
            if quick:
                print("[WARN] Quick-test mode: skipping strict DataFrame validation")
            else:
                validate_dataframe(df, critical=True)
            print_step_summary("load_data", status="OK", rows=len(df))
        except Exception as e:
            error_msg = log_error(e, "load_data", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 2/6] Applying the Kalman filter...")
        print("="*60)
        try:
            df = apply_kalman_filter(df)
            df["Close"] = df["Close_KF"]
            print_step_summary("kalman_filter", status="OK")
        except Exception as e:
            error_msg = log_error(e, "kalman_filter", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 3/6] Creating the RL environment...")
        print("="*60)
        try:
            allow_hold = True
            env = DummyVecEnv([make_env(df, allow_hold_when_flat=allow_hold, env_config=_load_env_config())])
            print_step_summary("make_env", status="OK")
        except Exception as e:
            error_msg = log_error(e, "make_env", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 4/6] Loading the RL model...")
        print("="*60)
        try:
            check_file_exists("models/ppo_trading.zip", critical=True)
            model = PPO.load("models/ppo_trading.zip")
            print_step_summary("model_load", status="OK")
        except Exception as e:
            error_msg = log_error(e, "model_load", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 5/6] Generating RL actions...")
        print("="*60)
        
        actions = []

        try:
            obs = env.reset()
            done_flag = False
            step = 0

            while not done_flag:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                if isinstance(done, (np.ndarray, list, tuple)):
                    done_flag = bool(done[0])
                else:
                    done_flag = bool(done)

                if isinstance(reward, (np.ndarray, list, tuple)):
                    reward_val = float(reward[0])
                else:
                    reward_val = float(reward)

                if isinstance(info, (list, tuple)) and len(info) > 0:
                    info_dict = info[0] if isinstance(info[0], dict) else {}
                elif isinstance(info, dict):
                    info_dict = info
                else:
                    info_dict = {}

                try:
                    internal_env = env.envs[0]
                    executed_step = int(info_dict.get("executed_step", max(0, getattr(internal_env, "current_step", 0) - 1)))
                    current_idx = min(max(executed_step, 0), len(df) - 1)
                    current_row = df.iloc[current_idx]
                    model_action_val, model_breakout_idx = _extract_model_actions(action)
                    action_val = int(info_dict.get("executed_action", model_action_val))
                    selected_breakout_idx = int(info_dict.get("selected_breakout_idx", model_breakout_idx))
                    position_val = int(info_dict.get("position", getattr(internal_env, "position", 0)))
                    equity_val = float(info_dict.get("equity", getattr(internal_env, "equity", np.nan)))

                    actions.append({
                        "Time": current_row["Time"],
                        "Open": current_row["Open"],
                        "High": current_row["High"],
                        "Low": current_row["Low"],
                        "Close": current_row["Close"],
                        "Regime": current_row.get("REGIME", "UNKNOWN"),
                        "Action": action_val,
                        "ModelAction": model_action_val,
                        "ModelBreakoutIdx": model_breakout_idx,
                        "SelectedBreakoutIdx": selected_breakout_idx,
                        "SelectedBreakoutN": int(info_dict.get("selected_breakout_n", 0)),
                        "BreakoutSignal": int(info_dict.get("selected_breakout_signal", 0)),
                        "BreakoutStrength": float(info_dict.get("selected_breakout_strength", 0.0)),
                        "Reward": reward_val,
                        "Position": position_val,
                        "Equity": equity_val,
                        "DangerLevel": int(info_dict.get("danger_level", 0)),
                        "PositionSize": float(info_dict.get("position_size", np.nan)),
                        "ForcedAction": bool(info_dict.get("forced_action", False)),
                        "HardStop": bool(info_dict.get("hard_stop_applied", False)),
                    })
                except Exception as e:
                    error_msg = log_error(e, f"action_generation_step_{step}", critical=False)
                    errors.append(error_msg)
                    continue
                
                step += 1
            
            print_step_summary("action_generation", status="OK", actions=len(actions))
        except Exception as e:
            error_msg = log_error(e, "action_generation", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 6/6] Saving the actions CSV...")
        print("="*60)
        
        try:
            os.makedirs("backtest/data", exist_ok=True)
            output_path = "backtest/data/xauusd_actions.csv"
            pd.DataFrame(actions).to_csv(output_path, index=False)
            print_step_summary("save_actions", status="OK", 
                             file=output_path, rows=len(actions))
        except Exception as e:
            error_msg = log_error(e, "save_actions", critical=True)
            errors.append(error_msg)
            raise
        
        status = "SUCCESS"
        
    except Exception as e:
        error_msg = log_error(e, "generate_actions_main", critical=True)
        errors.append(error_msg)
        status = "FAILURE"
    
    finally:
        try:
            env.close()
        except:
            pass
    
    print_final_report(status=status, errors=errors)
