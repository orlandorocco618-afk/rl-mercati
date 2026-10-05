#!/usr/bin/env python3
"""
Quick launcher for dual-system training.

This script orchestrates parallel training of:
1. Breakout agent (BR) - multi-seed tuning
2. Support/Resistance agent (SR) - multi-seed tuning

Both run independently; later they are evaluated for cooperation.
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(".")


def run_command(cmd, description, background=False):
    """Run a command with status reporting."""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    if background:
        print("[Launching in background...]")
        proc = subprocess.Popen(cmd, cwd=str(ROOT))
        return proc
    else:
        result = subprocess.run(cmd, cwd=str(ROOT))
        return result.returncode


def main():
    print("="*70)
    print("DUAL-SYSTEM RL TRAINING LAUNCHER")
    print("="*70)
    print("\nThis script coordinates training of two independent agents:")
    print("  1. Breakout agent (BR) - edge: multi-timeframe breakout patterns")
    print("  2. Support/Resistance agent (SR) - edge: zone-based mean reversion")
    print("\nBoth agents share the same dataset and analysis pipeline.")
    print("Cooperation strategies are evaluated once both are optimized.")
    print("="*70)

    mode = input("\nChoose mode:\n  1 = Train SR agent single-seed (seed 11)\n  2 = Run SR multi-seed grid (5 seeds)\n  3 = Check BR grid progress\n  4 = Check SR grid progress\n  5 = Prepare cooperation evaluation\nEnter choice (1-5): ").strip()

    if mode == "1":
        print("\n▶ Training Support/Resistance agent (single seed)...")
        env = os.environ.copy()
        env.update({
            'RL_SEED': '11',
            'RL_TRAIN_TIMESTEPS': '250000',
            'RL_ENABLE_ZONES': '1',
            'RL_ZONE_TOL': '0.002',
            'RL_ZONE_LOOKBACK': '5',
        })
        cmd = [sys.executable, 'train/train_agent_sr.py']
        result = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if result.returncode == 0:
            print("\n✓ SR Agent training completed successfully")
            print("  Models saved to:")
            print("    - models/ppo_sr_zones.zip")
            print("    - models/env_config_sr.json")
        else:
            print("\n✗ SR Agent training failed")
            sys.exit(1)

    elif mode == "2":
        print("\n▶ Launching SR multi-seed feature grid...")
        cmd = [sys.executable, 'tools/multi_seed_feature_grid_sr.py']
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode == 0:
            print("\n✓ SR grid completed")
            print("  Results saved to: backtest/plots_oos/feature_grid_multi_seed_sr/results.csv")
        else:
            print("\n✗ SR grid failed")
            sys.exit(1)

    elif mode == "3":
        print("\n▶ Checking BR grid progress...")
        try:
            import pandas as pd
            csv_path = Path("backtest/plots_oos/feature_grid_multi_seed/results.csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                print(f"\n✓ BR Grid Status:")
                print(f"  Rows completed: {len(df)}/25")
                print(f"  Seeds: {sorted(df['seed'].unique())}")
                print(f"  Best return: {df['total_return_pct'].max():.2f}%")
                print(f"\n  Sample results:")
                print(df[['seed', 'lookbacks', 'quantile', 'total_return_pct']].head(10).to_string(index=False))
            else:
                print("✗ BR grid results not found yet")
        except Exception as e:
            print(f"✗ Error: {e}")

    elif mode == "4":
        print("\n▶ Checking SR grid progress...")
        try:
            import pandas as pd
            csv_path = Path("backtest/plots_oos/feature_grid_multi_seed_sr/results.csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                print(f"\n✓ SR Grid Status:")
                print(f"  Rows completed: {len(df)}/25")
                print(f"  Seeds: {sorted(df['seed'].unique())}")
                print(f"  Best return: {df['total_return_pct'].max():.2f}%")
                print(f"\n  Sample results:")
                print(df[['seed', 'zone_tol', 'zone_lookback', 'total_return_pct']].head(10).to_string(index=False))
            else:
                print("✗ SR grid results not found yet")
        except Exception as e:
            print(f"✗ Error: {e}")

    elif mode == "5":
        print("\n▶ Preparing cooperation evaluation...")
        # Check that both agents exist
        br_model = Path("models/ppo_trading.zip")
        br_config = Path("models/env_config.json")
        sr_model = Path("models/ppo_sr_zones.zip")
        sr_config = Path("models/env_config_sr.json")

        missing = []
        if not br_model.exists():
            missing.append(f"BR model: {br_model}")
        if not br_config.exists():
            missing.append(f"BR config: {br_config}")
        if not sr_model.exists():
            missing.append(f"SR model: {sr_model}")
        if not sr_config.exists():
            missing.append(f"SR config: {sr_config}")

        if missing:
            print("\n✗ Missing files for cooperation evaluation:")
            for m in missing:
                print(f"    {m}")
            print("\nPlease train both agents first:")
            print("  1. Let current BR grid complete (or select mode 3 to check)")
            print("  2. Train SR agent (select mode 1)")
            print("  3. Run SR grid (select mode 2)")
        else:
            print("\n✓ Both agents found. Running cooperation evaluation...")
            cmd = [sys.executable, 'tools/dual_agent_cooperate.py']
            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode == 0:
                print("\n✓ Cooperation evaluation completed")
                print("  Results saved to: backtest/plots_oos/cooperation/results.json")
            else:
                print("\n✗ Cooperation evaluation failed")
                sys.exit(1)

    else:
        print("✗ Invalid choice")
        sys.exit(1)


if __name__ == "__main__":
    main()
