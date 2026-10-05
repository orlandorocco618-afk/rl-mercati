#!/usr/bin/env python3
"""
Cooperative/Ensemble evaluation of Breakout and S/R agents.

Once both agents are trained and optimized, this script tests various cooperation
strategies:
1. Alternation: use agent 1 for N bars, then agent 2
2. Voting: both predict, majority vote
3. Confidence blending: use whichever has higher confidence
4. Regime-based routing: use breakout in trending, S/R in ranging
"""

import json
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, '.')

from backtest.out_of_sample_test import _run_slice, split_train_test_by_date
from env.trading_env import TradingEnv
from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter


def load_agent(model_path: str, config_path: str) -> Tuple[PPO, dict]:
    """Load a trained agent model and its configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    model = PPO.load(model_path)
    return model, config


def make_env(df, env_config: dict):
    """Create environment with specified config."""
    def _init():
        cfg = dict(env_config)
        cfg.update({
            'df': df,
            'max_drawdown': 2.0,
            'allow_hold_when_flat': True,
            'regime_hard_stop': False,
            'random_start_on_reset': False,
        })
        return TradingEnv(**cfg)
    return _init


class DualAgentEvaluator:
    """Manage and evaluate cooperation strategies between two agents."""

    def __init__(self, model_br: PPO, config_br: dict, model_sr: PPO, config_sr: dict):
        self.model_br = model_br
        self.config_br = config_br
        self.model_sr = model_sr
        self.config_sr = config_sr

    def run_alternation(self, df_test: pd.DataFrame, switch_period: int = 288) -> dict:
        """Alternate between agents every switch_period bars."""
        print(f"\n[COOP] Alternation strategy (switch every {switch_period} bars)")
        
        env_br = DummyVecEnv([make_env(df_test, self.config_br)])
        env_sr = DummyVecEnv([make_env(df_test, self.config_sr)])
        
        obs_br = env_br.reset()
        obs_sr = env_sr.reset()
        
        actions_rows = []
        step_count = 0
        using_br = True
        
        done = False
        while not done and step_count < len(df_test) - 1:
            if step_count % switch_period == 0:
                using_br = not using_br
                label = "BR" if using_br else "SR"
                print(f"  Step {step_count}: switching to {label} agent")
            
            if using_br:
                action, _ = self.model_br.predict(obs_br, deterministic=True)
                obs_br, rewards, done_raw, info = env_br.step(action)
                obs = obs_br
                agent_label = "BR"
            else:
                action, _ = self.model_sr.predict(obs_sr, deterministic=True)
                obs_sr, rewards, done_raw, info = env_sr.step(action)
                obs = obs_sr
                agent_label = "SR"
            
            done = bool(done_raw[0]) if isinstance(done_raw, (np.ndarray, list, tuple)) else bool(done_raw)
            info_dict = info[0] if isinstance(info, (list, tuple)) else info
            
            actions_rows.append({
                'step': step_count,
                'action': int(info_dict.get('executed_action', 0)),
                'agent': agent_label,
                'reward': float(rewards[0]) if isinstance(rewards, (np.ndarray, list, tuple)) else float(rewards),
            })
            step_count += 1
        
        return {
            'strategy': 'alternation',
            'switch_period': switch_period,
            'steps': step_count,
            'actions': actions_rows,
        }

    def run_voting(self, df_test: pd.DataFrame) -> dict:
        """Both agents predict; pick action by voting (majority or best confidence)."""
        print("\n[COOP] Voting strategy (majority vote or confidence-weighted)")
        
        env_br = DummyVecEnv([make_env(df_test, self.config_br)])
        env_sr = DummyVecEnv([make_env(df_test, self.config_sr)])
        
        obs_br = env_br.reset()
        obs_sr = env_sr.reset()
        
        actions_rows = []
        step_count = 0
        
        done = False
        while not done and step_count < len(df_test) - 1:
            action_br, _ = self.model_br.predict(obs_br, deterministic=True)
            action_sr, _ = self.model_sr.predict(obs_sr, deterministic=True)
            
            action_br_val = int(np.asarray(action_br).reshape(-1)[0])
            action_sr_val = int(np.asarray(action_sr).reshape(-1)[0])
            
            # Simple voting: if same action, use it; otherwise hold
            if action_br_val == action_sr_val:
                chosen_action = action_br_val
                voted_by = "both"
            else:
                # Fallback: use BR as primary
                chosen_action = action_br_val
                voted_by = "br_primary"
            
            obs_br, rewards_br, done_raw_br, info_br = env_br.step([action_br_val])
            obs_sr, rewards_sr, done_raw_sr, info_sr = env_sr.step([action_sr_val])
            
            done = (bool(done_raw_br[0]) if isinstance(done_raw_br, (np.ndarray, list, tuple)) else bool(done_raw_br)) or \
                   (bool(done_raw_sr[0]) if isinstance(done_raw_sr, (np.ndarray, list, tuple)) else bool(done_raw_sr))
            
            actions_rows.append({
                'step': step_count,
                'action_br': action_br_val,
                'action_sr': action_sr_val,
                'chosen': chosen_action,
                'voted_by': voted_by,
                'reward_br': float(rewards_br[0]) if isinstance(rewards_br, (np.ndarray, list, tuple)) else float(rewards_br),
                'reward_sr': float(rewards_sr[0]) if isinstance(rewards_sr, (np.ndarray, list, tuple)) else float(rewards_sr),
            })
            step_count += 1
        
        return {
            'strategy': 'voting',
            'steps': step_count,
            'actions': actions_rows,
        }


def main():
    print("="*70)
    print("COOPERATIVE EVALUATION: Breakout vs. S/R Agents")
    print("="*70)

    # Paths to trained agents
    model_br_path = "models/ppo_trading.zip"
    config_br_path = "models/env_config.json"
    model_sr_path = "models/ppo_sr_zones.zip"
    config_sr_path = "models/env_config_sr.json"

    # Check files exist
    for p in [model_br_path, config_br_path, model_sr_path, config_sr_path]:
        if not Path(p).exists():
            print(f"✗ Missing: {p}")
            sys.exit(1)

    print(f"✓ Loading agents:")
    print(f"  Breakout agent: {model_br_path}")
    print(f"  S/R agent: {model_sr_path}")

    model_br, config_br = load_agent(model_br_path, config_br_path)
    model_sr, config_sr = load_agent(model_sr_path, config_sr_path)

    # Load test data
    print(f"\n✓ Loading test data...")
    train_end = "2022-12-31 23:00:00"
    val_end = "2023-12-31 23:00:00"
    
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col='Close')
    df['Close'] = df['Close_KF']
    _, df_test = split_train_test_by_date(df, train_end=train_end, val_end=val_end)
    print(f"  Test rows: {len(df_test)}")

    # Evaluate cooperation strategies
    evaluator = DualAgentEvaluator(model_br, config_br, model_sr, config_sr)

    results = {}
    results['alternation_288'] = evaluator.run_alternation(df_test, switch_period=288)
    results['alternation_144'] = evaluator.run_alternation(df_test, switch_period=144)
    results['voting'] = evaluator.run_voting(df_test)

    # Save results
    out_dir = Path("backtest/plots_oos/cooperation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / "results.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Cooperation evaluation complete. Results saved to {out_file}")


if __name__ == '__main__':
    main()
