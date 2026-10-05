import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.trading_env import TradingEnv
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


# ============================================================
#  BACKTEST ENVIRONMENT CREATION
# ============================================================

def make_env(df):
    """
    Factory that creates an RL environment.
    """
    def _init():
        return TradingEnv(df=df, allow_hold_when_flat=True, regime_hard_stop=False)
    return _init


# ============================================================
#  BACKTEST RUN
# ============================================================

def run_backtest(model_path: str):
    """
    Runs a full episode with the trained model.
    Records equity, position, regime and reward.
    """
    print("Loading data...")
    df = load_data()

    print("Creating the environment...")
    env = DummyVecEnv([make_env(df)])

    print("Loading the model...")
    model = PPO.load(model_path)

    obs = env.reset()
    done = False

    equity_curve = []
    position_curve = []
    regime_curve = []
    reward_curve = []

    step = 0

    print("Starting the backtest...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Extract info from the inner environment
        internal_env = env.envs[0]

        equity_curve.append(internal_env.equity)
        position_curve.append(internal_env.position)
        regime_curve.append(internal_env.df.iloc[internal_env.current_step]["REGIME"])
        reward_curve.append(reward)

        step += 1

    print("Backtest completed.")
    print(f"Final equity: {equity_curve[-1]:.2f}")

    # ============================================================
    #  STATISTICS
    # ============================================================

    equity = np.array(equity_curve)
    returns = np.diff(equity)

    total_pnl = equity[-1] - equity[0]
    max_dd = np.max(np.maximum.accumulate(equity) - equity)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 24 * 12)

    print("\n===== BACKTEST STATISTICS =====")
    print(f"Total PNL: {total_pnl:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    # ============================================================
    #  EQUITY CURVE CHART
    # ============================================================

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Equity")
    plt.title("Equity Curve - Backtest RL")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.show()

    return {
        "equity": equity_curve,
        "position": position_curve,
        "regime": regime_curve,
        "reward": reward_curve
    }


# ============================================================
#  ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    errors = []
    status = "FAILURE"
    
    try:
        print("="*60)
        print("[STEP 1/5] Loading H1+D1 data...")
        print("="*60)
        
        check_file_exists("data/xauusd_h1_clean.csv", critical=True)
        check_file_exists("data/xauusd_d1_clean.csv", critical=True)
        
        try:
            df = load_data()
            validate_dataframe(df, critical=True)
            print_step_summary("load_data", status="OK", rows=len(df))
        except Exception as e:
            error_msg = log_error(e, "load_data", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 2/5] Creating the RL environment...")
        print("="*60)
        try:
            env = DummyVecEnv([make_env(df)])
            print_step_summary("make_env", status="OK")
        except Exception as e:
            error_msg = log_error(e, "make_env", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 3/5] Loading the RL model...")
        print("="*60)
        try:
            check_file_exists("models/ppo_trading.zip", critical=True)
            model = PPO.load("models/ppo_trading.zip")
            print_step_summary("model_load", status="OK", file="models/ppo_trading.zip")
        except Exception as e:
            error_msg = log_error(e, "model_load", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 4/5] Running the backtest...")
        print("="*60)
        
        equity_curve = []
        position_curve = []
        regime_curve = []
        reward_curve = []
        
        try:
            obs = env.reset()
            done = False
            step = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                try:
                    internal_env = env.envs[0]
                    equity_curve.append(internal_env.equity)
                    position_curve.append(internal_env.position)
                    regime_curve.append(internal_env.df.iloc[internal_env.current_step]["REGIME"])
                    reward_curve.append(reward)
                except Exception as e:
                    error_msg = log_error(e, f"backtest_step_{step}", critical=False)
                    errors.append(error_msg)
                    continue
                
                step += 1
            
            print_step_summary("backtest_execution", status="OK", steps=len(equity_curve))
        except Exception as e:
            error_msg = log_error(e, "backtest_execution", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 5/5] Computing statistics and plotting...")
        print("="*60)
        
        try:
            equity = np.array(equity_curve)
            returns = np.diff(equity)
            
            total_pnl = equity[-1] - equity[0]
            max_dd = np.max(np.maximum.accumulate(equity) - equity)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 24 * 12)
            
            print("\n" + "="*40)
            print("BACKTEST STATISTICS")
            print("="*40)
            print(f"Total PNL: {total_pnl:.2f}")
            print(f"Max Drawdown: {max_dd:.2f}")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"N° Steps: {len(equity_curve)}")
            print("="*40)
            
            print_step_summary("statistics", status="OK")
        except Exception as e:
            error_msg = log_error(e, "statistics", critical=False)
            errors.append(error_msg)
            print("[WARN] Statistics computation failed")
        
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve, label="Equity", linewidth=2)
            plt.title("Equity Curve - Backtest RL", fontsize=14)
            plt.xlabel("Step", fontsize=12)
            plt.ylabel("Equity", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig("backtest/equity_curve.png", dpi=150)
            print("[OK] Chart saved: backtest/equity_curve.png")
            plt.show()
        except Exception as e:
            error_msg = log_error(e, "plotting", critical=False)
            errors.append(error_msg)
            print("[WARN] Plot failed but the backtest completed")
        
        status = "SUCCESS"
        
    except Exception as e:
        error_msg = log_error(e, "backtest_main", critical=True)
        errors.append(error_msg)
        status = "FAILURE"
    
    finally:
        try:
            env.close()
        except:
            pass
    
    print_final_report(status=status, errors=errors)
