import pandas as pd
import numpy as np
from pathlib import Path

# Load the top models
br = pd.read_csv('backtest/plots_oos/feature_grid_multi_seed/results.csv')
sr = pd.read_csv('backtest/plots_oos/feature_grid_multi_seed_sr/results.csv')

br_top3 = br.sort_values('total_return_pct', ascending=False).head(3)
sr_top3 = sr.sort_values('total_return_pct', ascending=False).head(3)

print("=== TOP 3 BR MODELS ===")
for i, row in br_top3.iterrows():
    print(f"{i+1}. Seed {int(row.seed)}, Lookbacks {row.lookbacks}, Quantile {row.quantile}")
    print(f"   Return: {row.total_return_pct:.2f}%, Drawdown: {row.max_drawdown_pct:.2f}%")

print("\n=== TOP 3 SR MODELS ===")
for i, row in sr_top3.iterrows():
    print(f"{i+1}. Seed {int(row.seed)}, Tol {row.zone_tol}, Lookback {int(row.zone_lookback)}")
    print(f"   Return: {row.total_return_pct:.2f}%, Drawdown: {row.max_drawdown_pct:.2f}%")

# Merge scheme with weights based on normalized total_return_pct
def normalize_weights(returns):
    min_ret = returns.min()
    max_ret = returns.max()
    if max_ret == min_ret:
        return np.ones(len(returns)) / len(returns)
    weights = (returns - min_ret) / (max_ret - min_ret)
    return weights / weights.sum()

br_weights = normalize_weights(br_top3.total_return_pct.values)
sr_weights = normalize_weights(sr_top3.total_return_pct.values)

print("\n=== WEIGHTS FOR SIGNAL MERGING ===")
print("BR weights:", [f"{w:.3f}" for w in br_weights])
print("SR weights:", [f"{w:.3f}" for w in sr_weights])

# Save the configuration for the backtest
config = {
    'br_models': br_top3[['model_path', 'env_config_path']].to_dict('records'),
    'sr_models': sr_top3[['model_path', 'env_config_path']].to_dict('records'),
    'br_weights': br_weights.tolist(),
    'sr_weights': sr_weights.tolist()
}

import json
Path('best_models/combined_config.json').write_text(json.dumps(config, indent=2))
print("\nConfig saved to best_models/combined_config.json")