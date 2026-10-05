"""
PHASE 2: Fine-tuning without Forcing
Loads the best model from the bootstrap, disables forcing,
and keeps training for 100k timesteps with an iterative analysis every 10k.
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

CHUNK_TIMESTEPS = 10_000
MAX_CHUNKS = 10  # 100k total per phase 2
BEST_BOOTSTRAP_MODEL = "models/checkpoints_iter/ppo_bootstrap_best.zip"
CHECKPOINT_DIR = "models/checkpoints_phase2"
ACTIONS_CSV = "backtest/data/xauusd_actions_phase2.csv"
PLOTS_DIR = "backtest/plots_phase2"

# Phase 2 exit metrics (target: better performance)
EXIT_THRESHOLD = {
    "total_return_pct": 15.0,     # Conservative for phase 2
    "pct_profitable": 55.0,        # A bit more conservative
    "total_trades": 80,            # A bit less than bootstrap
}

# Bias thresholds
BIAS_THRESHOLD = {
    "return_drop": -5.0,           # More tolerant than bootstrap
    "winrate_drop": -20.0,
    "max_drawdown_increase": 15.0,
}


def make_env(df):
    """Creates an environment with free hold + light directional rebalancing."""
    def _init():
        return TradingEnv(
            df=df,
            window_size=100,
            initial_capital=10000.0,
            max_drawdown=0.30,
            allow_hold_when_flat=True,  # Hold allowed when flat
            regime_hard_stop=False,
            enforce_directional_rebalance=True,
            rebalance_after_entries=30,
            rebalance_dominance_threshold=0.70,
            directional_balance_reward_coef=0.001
        )
    return _init


def load_latest_metrics():
    """Loads the summary JSON from the ltst plots."""
    # Try ltst first, then root
    summary_path = Path(PLOTS_DIR) / "ltst" / "analysis_summary.json"
    if not summary_path.exists():
        summary_path = Path(PLOTS_DIR) / "analysis_summary.json"
    
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def check_bias(current_metrics, previous_metrics):
    """Checks for degradation."""
    if previous_metrics is None:
        return False, "First chunk, no comparison"
    
    bias_reasons = []
    
    ret_drop = current_metrics.get("total_return_pct", 0) - previous_metrics.get("total_return_pct", 0)
    if ret_drop < BIAS_THRESHOLD["return_drop"]:
        bias_reasons.append(f"Return drop: {ret_drop:.2f}% < {BIAS_THRESHOLD['return_drop']:.2f}%")
    
    wr_drop = current_metrics.get("pct_profitable", 0) - previous_metrics.get("pct_profitable", 0)
    if wr_drop < BIAS_THRESHOLD["winrate_drop"]:
        bias_reasons.append(f"Win rate drop: {wr_drop:.2f}% < {BIAS_THRESHOLD['winrate_drop']:.2f}%")
    
    dd_curr = current_metrics.get("max_drawdown_pct", 0)
    dd_prev = previous_metrics.get("max_drawdown_pct", 0)
    dd_change = dd_curr - dd_prev
    if dd_change < -BIAS_THRESHOLD["max_drawdown_increase"]:
        bias_reasons.append(f"Drawdown worsened: {dd_change:.2f}% < -{BIAS_THRESHOLD['max_drawdown_increase']:.2f}%")
    
    has_bias = len(bias_reasons) > 0
    reason = " | ".join(bias_reasons) if bias_reasons else "No issues detected"
    
    return has_bias, reason


def check_exit_conditions(metrics):
    """Checks the exit criteria."""
    missing = []
    
    if metrics.get("total_return_pct", 0) < EXIT_THRESHOLD["total_return_pct"]:
        missing.append(f"Return {metrics.get('total_return_pct', 0):.2f}% < {EXIT_THRESHOLD['total_return_pct']}%")
    
    if metrics.get("pct_profitable", 0) < EXIT_THRESHOLD["pct_profitable"]:
        missing.append(f"Win Rate {metrics.get('pct_profitable', 0):.2f}% < {EXIT_THRESHOLD['pct_profitable']}%")
    
    if metrics.get("total_trades", 0) < EXIT_THRESHOLD["total_trades"]:
        missing.append(f"Trades {metrics.get('total_trades', 0)} < {EXIT_THRESHOLD['total_trades']}")
    
    can_exit = len(missing) == 0
    return can_exit, missing


def run_phase2_finetuning():
    """
    Fine-tuning phase 2: loads the best model from the bootstrap, disables forcing,
    keeps training for 100k timesteps with iterative analysis.
    """
    print("\n" + "="*80)
    print("PHASE 2: FINE-TUNING (allow_hold_when_flat=True + directional rebalance)")
    print("="*80)
    print(f"Chunk size: {CHUNK_TIMESTEPS} timesteps")
    print(f"Max chunks: {MAX_CHUNKS} ({MAX_CHUNKS * CHUNK_TIMESTEPS} total timesteps)")
    print(f"Forcing exploration: DISABLED")
    print(f"Model: Loading from {BEST_BOOTSTRAP_MODEL}")
    print("="*80 + "\n")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load data
    print("[INIT] Loading data...")
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col="Close")
    df["Close"] = df["Close_KF"]
    print(f"  OK - Data: {len(df)} rows")
    
    # Load best model from bootstrap
    print(f"[INIT] Loading the best model from the bootstrap...")
    if not Path(BEST_BOOTSTRAP_MODEL).exists():
        print(f"  FAIL - Model not found: {BEST_BOOTSTRAP_MODEL}")
        return False, f"Best bootstrap model not found"
    
    env = DummyVecEnv([make_env(df)])
    model = PPO.load(BEST_BOOTSTRAP_MODEL, env=env)
    print(f"  OK - Model loaded")
    
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
        
        # Save checkpoint
        checkpoint_path = Path(CHECKPOINT_DIR) / f"ppo_phase2_chunk{chunk_num:02d}.zip"
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
            print(f"Fine-tuning stopped to avoid degradation")
            return True, f"Phase 2 stopped at chunk {chunk_num} due to bias"
        
        # Check exit conditions
        can_exit, missing = check_exit_conditions(current_metrics)
        if can_exit:
            print(f"\nOK - EXIT CONDITIONS MET!")
            print(f"  Fine-tuning completed successfully")
            print(f"  Ready for out-of-sample testing")
            return True, f"Phase 2 completed successfully at chunk {chunk_num}"
        else:
            print(f"\nINFO - EXIT CONDITIONS not met yet:")
            for m in missing:
                print(f"   - {m}")
        
        previous_metrics = current_metrics
        
        # Save best if good
        if chunk_num == 1 or (current_metrics.get('total_return_pct', 0) > 10):
            best_path = Path(CHECKPOINT_DIR) / "ppo_phase2_best.zip"
            model.save(str(best_path))
            print(f"\nINFO - Best model updated")
        
        if chunk_idx < MAX_CHUNKS:
            print(f"\n-> Moving on to the next chunk...")
    
    print(f"\n{'='*80}")
    print(f"OK - CHUNK LIMIT REACHED ({MAX_CHUNKS})")
    print(f"  Fine-tuning completed: {total_timesteps} timesteps in total")
    print(f"  Latest results available in: {PLOTS_DIR}/ltst/")
    print(f"{'='*80}\n")
    
    return True, f"Phase 2 completed with {MAX_CHUNKS} chunks ({total_timesteps} timesteps)"


if __name__ == "__main__":
    success, message = run_phase2_finetuning()
    print(f"\n{'='*80}")
    if success:
        print(f"OK - {message}")
    else:
        print(f"FAIL - {message}")
    print(f"{'='*80}\n")
    sys.exit(0 if success else 1)
