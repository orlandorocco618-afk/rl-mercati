"""
Iterative Bootstrap Training Loop
Runs training in chunks of 10k timesteps, analyzes after every chunk,
and stops if it detects bias/degradation in the metrics.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, ".")

from env.trading_env import TradingEnv
from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from backtest.generate_actions import generate_actions
from backtest.run_analysis import run_analysis_and_organize


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_TIMESTEPS = 10_000          # Timesteps per chunk
MAX_CHUNKS = 10                    # Max chunks (10 * 10k = 100k in total)
MODEL_PATH = "models/ppo_trading.zip"
CHECKPOINT_DIR = "models/checkpoints_iter"
ACTIONS_CSV = "backtest/data/xauusd_actions.csv"
PLOTS_DIR = "backtest/plots"

# Bootstrap exit metrics (once they are met, we can exit)
EXIT_THRESHOLD = {
    "total_return_pct": 5.0,
    "pct_profitable": 50.0,
    "total_trades": 100,
}

# Bias thresholds (degradation vs the previous checkpoint)
BIAS_THRESHOLD = {
    "return_drop": -3.0,      # If return drops by 3% = bias
    "winrate_drop": -15.0,    # If win rate drops by 15% = bias
    "max_drawdown_increase": 10.0,  # If drawdown worsens by 10% = bias
}


def make_env(df):
    """Creates an environment with HOLD allowed."""
    def _init():
        return TradingEnv(
            df=df,
            window_size=100,
            initial_capital=10000.0,
            max_drawdown=0.30,
            allow_hold_when_flat=True,
            regime_hard_stop=False,
            enforce_directional_rebalance=True,
            rebalance_after_entries=40,
            rebalance_dominance_threshold=0.70,
            directional_balance_reward_coef=0.001
        )
    return _init


def load_latest_metrics():
    """Loads the summary JSON from the ltst plots."""
    summary_path = Path(PLOTS_DIR) / "ltst" / "analysis_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def check_bias(current_metrics, previous_metrics):
    """
    Checks whether the current metrics show bias/degradation.
    Returns: (has_bias: bool, reason: str)
    """
    if previous_metrics is None:
        return False, "First chunk, no comparison"
    
    bias_reasons = []
    
    # Check return drop
    ret_drop = current_metrics.get("total_return_pct", 0) - previous_metrics.get("total_return_pct", 0)
    if ret_drop < BIAS_THRESHOLD["return_drop"]:
        bias_reasons.append(f"Return drop: {ret_drop:.2f}% < {BIAS_THRESHOLD['return_drop']:.2f}%")
    
    # Check win rate drop
    wr_drop = current_metrics.get("pct_profitable", 0) - previous_metrics.get("pct_profitable", 0)
    if wr_drop < BIAS_THRESHOLD["winrate_drop"]:
        bias_reasons.append(f"Win rate drop: {wr_drop:.2f}% < {BIAS_THRESHOLD['winrate_drop']:.2f}%")
    
    # Check max drawdown increase (degradation)
    dd_curr = current_metrics.get("max_drawdown_pct", 0)
    dd_prev = previous_metrics.get("max_drawdown_pct", 0)
    dd_change = dd_curr - dd_prev  # Negative = worse (more negative)
    if dd_change < -BIAS_THRESHOLD["max_drawdown_increase"]:
        bias_reasons.append(f"Drawdown worsened: {dd_change:.2f}% < -{BIAS_THRESHOLD['max_drawdown_increase']:.2f}%")
    
    has_bias = len(bias_reasons) > 0
    reason = " | ".join(bias_reasons) if bias_reasons else "No issues detected"
    
    return has_bias, reason


def check_exit_conditions(metrics):
    """
    Checks whether the bootstrap exit criteria have been reached.
    Returns: (can_exit: bool, missing: list)
    """
    missing = []
    
    if metrics.get("total_return_pct", 0) < EXIT_THRESHOLD["total_return_pct"]:
        missing.append(f"Return {metrics.get('total_return_pct', 0):.2f}% < {EXIT_THRESHOLD['total_return_pct']}%")
    
    if metrics.get("pct_profitable", 0) < EXIT_THRESHOLD["pct_profitable"]:
        missing.append(f"Win Rate {metrics.get('pct_profitable', 0):.2f}% < {EXIT_THRESHOLD['pct_profitable']}%")
    
    if metrics.get("total_trades", 0) < EXIT_THRESHOLD["total_trades"]:
        missing.append(f"Trades {metrics.get('total_trades', 0)} < {EXIT_THRESHOLD['total_trades']}")
    
    can_exit = len(missing) == 0
    return can_exit, missing


def run_iterative_bootstrap():
    """
    Training loop: every 10k-timestep chunk, analyze and check for bias.
    """
    print("\n" + "="*80)
    print("ITERATIVE BOOTSTRAP TRAINING LOOP")
    print("="*80)
    print(f"Chunk size: {CHUNK_TIMESTEPS} timesteps")
    print(f"Max chunks: {MAX_CHUNKS} ({MAX_CHUNKS * CHUNK_TIMESTEPS} total timesteps)")
    print(f"HOLD policy: allow_hold_when_flat=True")
    print("="*80 + "\n")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Load data
    print("[INIT] Loading data...")
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col="Close")
    df["Close"] = df["Close_KF"]
    print(f"  OK - Data: {len(df)} rows")
    
    # Create environment and model (first time)
    print("[INIT] Creating the environment and the model...")
    env = DummyVecEnv([make_env(df)])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        tensorboard_log=None,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05  # INCREASED from 0.01 to prevent policy collapse
    )
    model.set_random_seed(42)
    print("  OK - Model initialized")
    
    previous_metrics = None
    total_timesteps = 0
    
    for chunk_idx in range(1, MAX_CHUNKS + 1):
        chunk_num = chunk_idx
        cumulative_timesteps = chunk_idx * CHUNK_TIMESTEPS
        
        print(f"\n{'='*80}")
        print(f"CHUNK {chunk_num}/{MAX_CHUNKS} - Timesteps: {total_timesteps} → {cumulative_timesteps}")
        print(f"{'='*80}")
        
        # Train chunk
        print(f"[TRAIN] Esecuzione {CHUNK_TIMESTEPS} timesteps...")
        try:
            model.learn(total_timesteps=CHUNK_TIMESTEPS, progress_bar=False)
            total_timesteps = cumulative_timesteps
            print(f"  OK - Training completed")
        except Exception as e:
            print(f"  FAIL - Training failed: {e}")
            return False, f"Training error at chunk {chunk_num}: {e}"
        
        # Save model
        checkpoint_path = Path(CHECKPOINT_DIR) / f"ppo_bootstrap_chunk{chunk_num:02d}.zip"
        model.save(str(checkpoint_path))
        print(f"  OK - Checkpoint saved: {checkpoint_path.name}")
        
        # Generate actions
        print(f"[ANALYSIS] Generating actions...")
        try:
            generate_actions(
                model_path=str(checkpoint_path),
                output_path=ACTIONS_CSV,
                allow_hold_when_flat=True
            )
            print(f"  OK - Actions generated")
        except Exception as e:
            print(f"  FAIL - Action generation failed: {e}")
            return False, f"Action generation error at chunk {chunk_num}: {e}"
        
        # Run analysis and organize
        print(f"[ANALYSIS] Analysis and chart organization...")
        try:
            run_analysis_and_organize(
                actions_csv=ACTIONS_CSV,
                initial_capital=10000.0,
                save_dir=PLOTS_DIR
            )
            print(f"  OK - Analysis completed")
        except Exception as e:
            print(f"  FAIL - Analysis failed: {e}")
            return False, f"Analysis error at chunk {chunk_num}: {e}"
        
        # Load metrics
        current_metrics = load_latest_metrics()
        if current_metrics is None:
            print(f"  FAIL - Could not load metrics")
            return False, f"Metrics not found at chunk {chunk_num}"
        
        # Print metrics
        print(f"\n[METRICS] Chunk {chunk_num}:")
        print(f"  Return:      {current_metrics.get('total_return_pct', 0):+.2f}%")
        print(f"  Win Rate:    {current_metrics.get('pct_profitable', 0):.1f}%")
        print(f"  Trades:      {current_metrics.get('total_trades', 0)}")
        print(f"  Drawdown:    {current_metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Steps:       {current_metrics.get('steps', 0)}")
        
        # Check bias
        has_bias, bias_reason = check_bias(current_metrics, previous_metrics)
        if has_bias:
            print(f"\nWARN - BIAS DETECTED: {bias_reason}")
            print(f"Bootstrap stopped to avoid degradation")
            return True, f"Bootstrap stopped at chunk {chunk_num} due to bias"
        
        # Check exit conditions
        can_exit, missing = check_exit_conditions(current_metrics)
        if can_exit:
            print(f"\nOK - EXIT CONDITIONS MET!")
            print(f"  Bootstrap phase completed successfully")
            print(f"  Ready for Phase 2 (fine-tuning)")
            return True, f"Bootstrap completed successfully at chunk {chunk_num}"
        else:
            print(f"\nINFO - EXIT CONDITIONS not met yet:")
            for m in missing:
                print(f"   - {m}")
        
        previous_metrics = current_metrics
        
        # STRATEGY: After chunk 1, if performance was good, saving best model
        if chunk_num == 1 and can_exit == False:
            print(f"\nINFO - Chunk 1 completed. Saving best model...")
            best_checkpoint = Path(CHECKPOINT_DIR) / "ppo_bootstrap_best.zip"
            model.save(str(best_checkpoint))
            print(f"  OK - Best model saved for recovery")
        
        # Continue?
        if chunk_idx < MAX_CHUNKS:
            print(f"\n-> Moving on to the next chunk...")
    
    print(f"\n{'='*80}")
    print(f"OK - CHUNK LIMIT REACHED ({MAX_CHUNKS})")
    print(f"  Training completed: {total_timesteps} timesteps in total")
    print(f"  Latest results available in: backtest/plots/ltst/")
    print(f"{'='*80}\n")
    
    return True, f"Bootstrap completed with {MAX_CHUNKS} chunks ({total_timesteps} timesteps)"


if __name__ == "__main__":
    success, message = run_iterative_bootstrap()
    print(f"\n{'='*80}")
    if success:
        print(f"OK - {message}")
    else:
        print(f"FAIL - {message}")
    print(f"{'='*80}\n")
    sys.exit(0 if success else 1)
