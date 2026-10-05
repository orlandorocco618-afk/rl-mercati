"""
GUIDE: How to add robust error handling to the modules
========================================================

This file shows exactly how to integrate the error_handler module
into the main files (train_agent.py, backtest.py, etc.)

STEP 1: Import the module
---------------------------
from utils.error_handler import (
    handle_errors, safe_execute, log_error, 
    validate_dataframe, check_file_exists, print_step_summary
)

STEP 2: Use the @handle_errors decorator
---------------------------------------
@handle_errors(error_context="train_agent", critical=True)
def train_agent(...):
    # Your code
    pass

STEP 3: Use safe_execute for critical steps
---------------------------------------------
result = safe_execute(
    func=load_csv,
    args=(csv_path,),
    error_context="load_csv",
    critical=True
)

STEP 4: Use validate_dataframe for data checks
----------------------------------------------------
validate_dataframe(
    df,
    expected_columns=["Time", "Open", "High", "Low", "Close"],
    min_rows=1000,
    context="H1 data"
)

STEP 5: Try-except with log_error
---------------------------------
try:
    df = pd.read_csv(path)
except Exception as e:
    log_error(e, context="read_csv", critical=True)
    raise

STEP 6: Print a step summary
-----------------------------
import time
start = time.time()
# ... run the operation ...
print_step_summary(
    "Train PPO",
    status="OK",
    details={"timesteps": 300_000, "loss": 0.123},
    duration_sec=time.time() - start
)
"""

# ============================================================
#  FULL EXAMPLE: train_agent.py WITH ERROR HANDLING
# ============================================================

TRAIN_AGENT_EXAMPLE = '''
import os
import pandas as pd
import time
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from env.trading_env import TradingEnv
from utils.kalman import apply_kalman_filter
from utils.data_utils import merge_h1_with_d1_regimes
from utils.error_handler import (
    handle_errors, safe_execute, log_error, 
    validate_dataframe, check_file_exists, print_step_summary
)


def load_data(
    h1_path: str = "data/xauusd_h1_clean.csv",
    d1_path: str = "data/xauusd_d1_clean.csv"
) -> pd.DataFrame:
    """Loads H1+D1 with robust error handling."""
    
    check_file_exists(h1_path, "H1 CSV")
    check_file_exists(d1_path, "D1 CSV")
    
    df = merge_h1_with_d1_regimes(h1_path, d1_path)
    
    validate_dataframe(
        df,
        expected_columns=["Time", "Close", "REGIME"],
        min_rows=1000,
        context="H1+D1 merged data"
    )
    
    return df


def make_env(df, window_size=100, initial_capital=10000.0):
    def _init():
        return TradingEnv(
            df=df,
            window_size=window_size,
            initial_capital=initial_capital,
            max_drawdown=0.30
        )
    return _init


@handle_errors(error_context="train_agent", critical=True)
def train_agent(
    model_path: str = "models/ppo_trading.zip",
    timesteps: int = 300_000,
    tensorboard_log: str = "logs/"
):
    """PPO training with full error handling."""
    
    start_time = time.time()
    
    try:
        # ====== STEP 1: Load data ======
        try:
            print("\\n" + "="*60)
            print("[STEP 1] Loading H1+D1 data...")
            print("="*60)
            df = load_data()
            print_step_summary(
                "Data Loading",
                status="OK",
                details={"rows": len(df), "columns": len(df.columns)}
            )
        except Exception as e:
            log_error(e, "load_data", critical=True)
            raise
        
        # ====== STEP 2: Apply Kalman ======
        try:
            print("\\n" + "="*60)
            print("[STEP 2] Applying the Kalman filter...")
            print("="*60)
            df = apply_kalman_filter(df, price_col="Close")
            if "Close_KF" not in df.columns:
                raise ValueError("Close_KF column not produced by the Kalman filter")
            df["Close"] = df["Close_KF"]
            print_step_summary("Kalman Filter", status="OK")
        except Exception as e:
            log_error(e, "apply_kalman_filter", critical=True)
            raise
        
        # ====== STEP 3: Create environment ======
        try:
            print("\\n" + "="*60)
            print("[STEP 3] Creating the RL environment...")
            print("="*60)
            env = DummyVecEnv([make_env(df)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)
            print_step_summary("Environment Creation", status="OK")
        except Exception as e:
            log_error(e, "create_environment", critical=True)
            raise
        
        # ====== STEP 4: Initialize model ======
        try:
            print("\\n" + "="*60)
            print("[STEP 4] Initializing the PPO model...")
            print("="*60)
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                tensorboard_log=tensorboard_log,
                learning_rate=3e-4,
                batch_size=128,
                n_steps=2048,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0
            )
            model.set_random_seed(42)
            print_step_summary("Model Initialization", status="OK")
        except Exception as e:
            log_error(e, "initialize_model", critical=True)
            raise
        
        # ====== STEP 5: Setup callbacks ======
        try:
            print("\\n" + "="*60)
            print("[STEP 5] Setup callbacks...")
            print("="*60)
            os.makedirs("models/checkpoints/", exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=50_000,
                save_path="models/checkpoints/",
                name_prefix="ppo_trading"
            )
            print_step_summary("Callbacks Setup", status="OK")
        except Exception as e:
            log_error(e, "setup_callbacks", critical=False)
            checkpoint_callback = None
        
        # ====== STEP 6: Training ======
        try:
            print("\\n" + "="*60)
            print(f"[STEP 6] Starting training ({timesteps} timesteps)...")
            print("="*60)
            model.learn(
                total_timesteps=timesteps,
                callback=checkpoint_callback if checkpoint_callback else None,
                progress_bar=True
            )
            print_step_summary(
                "Training",
                status="OK",
                details={"timesteps": timesteps}
            )
        except Exception as e:
            log_error(e, "model_training", critical=True)
            raise
        
        # ====== STEP 7: Save model ======
        try:
            print("\\n" + "="*60)
            print("[STEP 7] Saving the model...")
            print("="*60)
            os.makedirs("models/", exist_ok=True)
            model.save(model_path)
            print_step_summary(
                "Model Saving",
                status="OK",
                details={"path": model_path}
            )
        except Exception as e:
            log_error(e, "save_model", critical=True)
            raise
        
        # ====== STEP 8: Save normalization ======
        try:
            print("\\n" + "="*60)
            print("[STEP 8] Saving the normalization...")
            print("="*60)
            env.save("models/vecnormalize.pkl")
            print_step_summary(
                "Normalization Saving",
                status="OK"
            )
        except Exception as e:
            log_error(e, "save_normalization", critical=False)
        
        # ====== Cleanup ======
        try:
            env.close()
        except Exception as e:
            log_error(e, "env_close", critical=False)
        
        elapsed = time.time() - start_time
        print("\\n" + "="*60)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total time: {elapsed:.2f}s")
        print(f"Model saved: {model_path}")
        print("="*60 + "\\n")
    
    except Exception as e:
        elapsed = time.time() - start_time
        print("\\n" + "="*60)
        print("✗ TRAINING FAILED")
        print("="*60)
        print(f"Time before the failure: {elapsed:.2f}s")
        print(f"Error: {str(e)}")
        print("="*60 + "\\n")
        raise


if __name__ == "__main__":
    train_agent(
        model_path="models/ppo_trading.zip",
        timesteps=300_000
    )
'''

# ============================================================
#  WHAT TO ADD TO EACH FILE
# ============================================================

INTEGRATION_CHECKLIST = """
CHECKLIST: Add error handling to every file
================================================

[ ] train/train_agent.py
    - Import error_handler
    - @handle_errors on train_agent()
    - validate_dataframe() after load_data()
    - try-except around every main step
    - print_step_summary() after critical operations

[ ] backtest/backtest.py
    - Import error_handler
    - @handle_errors on run_backtest()
    - check_file_exists() for the model path
    - try-except around the prediction loop

[ ] backtest/generate_actions.py
    - Import error_handler
    - validate_dataframe() after load_data()
    - safe_execute() for PPO.load()
    - Log every generated action

[ ] backtest/backtest_bt.py
    - Import error_handler
    - check_file_exists() for the CSV and actions
    - try-except around cerebro.run()
    - Log the final result

[ ] utils/data_utils.py
    - Import error_handler
    - Wrap load_csv() in try-except
    - Wrap merge_h1_with_d1_regimes() in try-except
    - check_columns() after every merge

[ ] utils/regimes.py
    - Import error_handler
    - try-except around add_regimes()
    - Log the regime distribution
    - Check for NaN in the computed values

[ ] env/trading_env.py
    - Import error_handler
    - Input validation in __init__()
    - Data checks in reset()
    - Log errors in step()

PRIORITY:
1. train_agent.py (critical for training)
2. backtest.py (critical for testing)
3. generate_actions.py (critical for simulation)
4. data_utils.py (foundation of everything)
5. trading_env.py (real-time feedback)
"""

print(__doc__)
print(TRAIN_AGENT_EXAMPLE)
print(INTEGRATION_CHECKLIST)
