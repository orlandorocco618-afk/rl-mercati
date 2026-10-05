# Dual-System RL Architecture: Breakout vs. Support/Resistance

This document describes the parallel training and evaluation of two distinct RL agents, each specialized on a different market microstructure **edge**:

- **Agent 1 (Breakout/BR):** exploits directional breakout patterns via multi-timeframe analysis
- **Agent 2 (S/R/Support-Resistance):** exploits mean-reversion at support/resistance zones

Both agents are **independently trained, separately optimized, and later cooperatively evaluated**.

---

## Directory & File Structure

### Shared Resources
```
data/                          # Identical dataset (H1 + D1 merged, Kalman-filtered)
utils/
  - data_utils.py            # merge_h1_with_d1_regimes() shared by both
  - kalman.py                # Filter shared by both
  - support_zones.py         # Zone detection (used by SR agent)
  - indicators.py            # Common technical indicators
backtest/
  - out_of_sample_test.py    # Generic OOS evaluation (parametrizable by config)
  - generate_actions.py      # Action-generation pipeline
  - run_analysis.py          # Backtesting & metrics calculation
```

### Agent 1: Breakout (BR)
```
train/train_agent.py                      # Main training script
  - breakout_enabled=True, enable_zones=False
  - Multi-seed support via RL_* env vars
  - Output: models/ppo_trading.zip, models/env_config.json

models/
  - ppo_trading.zip                       # Trained breakout agent
  - env_config.json                       # BR configuration
  - checkpoints/                          # BR intermediate checkpoints
  - multi_seed_runs/seed_*/               # BR multi-seed models

tools/
  - feature_grid.py                       # Fast inference grid (BR)
  - multi_seed_feature_grid.py            # BR multi-seed grid tuning
  - turnover_strength_grid.py             # BR parameter scan
  - multi_seed_duration_grid.py           # BR duration tuning

backtest/plots_oos/
  - feature_grid_results.csv              # BR grid results
  - feature_grid_multi_seed/results.csv   # BR multi-seed results
```

### Agent 2: Support/Resistance (SR)
```
train/train_agent_sr.py                   # SR training script (NEW)
  - breakout_enabled=False, enable_zones=True
  - Multi-seed support via RL_* env vars
  - Output: models/ppo_sr_zones.zip, models/env_config_sr.json

models/
  - ppo_sr_zones.zip                      # Trained SR agent
  - env_config_sr.json                    # SR configuration
  - checkpoints_sr/                       # SR intermediate checkpoints
  - multi_seed_runs_sr/seed_*/            # SR multi-seed models

tools/
  - multi_seed_feature_grid_sr.py         # SR multi-seed grid tuning (NEW)
    - Tuning: zone_tol (cluster tolerance), zone_lookback (extrema window)

backtest/plots_oos/
  - feature_grid_multi_seed_sr/results.csv # SR multi-seed results
```

### Cooperation & Ensemble
```
tools/
  - dual_agent_cooperate.py               # Load both agents, run cooperation strategies (NEW)
    - Alternation (switch every N bars)
    - Voting (both predict, majority vote)
    - Confidence blending (choose best)
    - Regime-based routing (BR in trends, SR in ranges)

backtest/plots_oos/
  - cooperation/results.json              # Cooperation evaluation results
```

---

## Training Workflow

### Phase 1: BR Agent (Existing)
```bash
# Train breakout agent (already running)
RL_SEED=11 RL_TRAIN_TIMESTEPS=250000 python train/train_agent.py
# Output: models/ppo_trading.zip, models/env_config.json
```

### Phase 2: SR Agent (New)
```bash
# Train support/resistance agent
RL_SEED=11 RL_TRAIN_TIMESTEPS=250000 python train/train_agent_sr.py
# Output: models/ppo_sr_zones.zip, models/env_config_sr.json
```

### Phase 3: Multi-Seed BR Grid (Running Now)
```bash
# Tune breakout parameters across multiple seeds
python tools/multi_seed_feature_grid.py
# Output: backtest/plots_oos/feature_grid_multi_seed/results.csv (25 rows, 5 seeds × 5 configs)
```

### Phase 4: Multi-Seed SR Grid
```bash
# Tune zone parameters across multiple seeds
python tools/multi_seed_feature_grid_sr.py
# Output: backtest/plots_oos/feature_grid_multi_seed_sr/results.csv (25 rows, 5 seeds × 5 configs)
```

### Phase 5: Agent Cooperation
```bash
# Once both agents are trained and optimized, evaluate cooperation
python tools/dual_agent_cooperate.py
# Output: backtest/plots_oos/cooperation/results.json
```

---

## Key Configuration Parameters

### BR Agent (Breakout)
```json
{
  "breakout_lookbacks": [72, 96, 144],
  "breakout_entry_strength_quantile": 0.65,
  "enable_zones": false,
  "atr_stop_multiple": 2.5
}
```

### SR Agent (Support/Resistance)
```json
{
  "breakout_lookbacks": [],
  "enable_zones": true,
  "zone_tol": 0.002,
  "zone_lookback": 5
}
```

Both agents share:
- Dataset (H1+D1, Kalman-filtered)
- Duration constraints (min/max trade duration)
- Activity targets (hold penalty, active steps target)
- Basic reward structure (entry quality, close quality, turnover)

---

## Analysis & Metrics

### Individual Agent Analysis
For each agent (BR and SR), generate:
1. **In-sample performance** (training period): Sharpe, return, drawdown, win rate
2. **Out-of-sample performance** (test period): evaluate robustness and generalization
3. **Regime-specific performance**: analyze performance by REGIME (trending vs. ranging)
4. **Edge characteristics**: 
   - BR: breakout frequency, average breakout strength, direction balance
   - SR: zone touch frequency, distance-to-zone distribution, zone efficacy

### Cooperation Analysis
Once both agents are deployed:
1. **Alternation**: compare holding period returns when alternating every N bars
2. **Voting**: measure improvement in win rate when both agents agree
3. **Blending**: measure out-of-sample stability when using confidence-weighted decisions
4. **Drawdown reduction**: test if alternating reduces max drawdown during crisis periods

---

## Environment Variable Guide

### Shared (Both Agents)
```bash
RL_SEED                    # Random seed for reproducibility (default: 11)
RL_TRAIN_END               # Training split cutoff (default: 2022-12-31 23:00:00)
RL_VAL_END                 # Validation split cutoff (default: 2023-12-31 23:00:00)
RL_TRAIN_TIMESTEPS        # PPO training timesteps (default: 250000)
RL_LEARNING_RATE          # Optimizer learning rate (default: 0.0001)
RL_BATCH_SIZE             # PPO batch size (default: 256)
RL_N_STEPS                # Rollout length (default: 2048)
```

### BR-Specific
```bash
RL_BREAKOUT_LOOKBACKS              # Lookback windows (default: 72,96,144)
RL_BREAKOUT_ENTRY_STRENGTH_QUANTILE # Threshold for entry (default: 0.65)
```

### SR-Specific
```bash
RL_ENABLE_ZONES            # Use zone features (always 1 for SR)
RL_ZONE_TOL                # Zone cluster tolerance (default: 0.002)
RL_ZONE_LOOKBACK           # Extrema detection window (default: 5)
```

---

## Example: Running Both Systems in Parallel

### Terminal 1: BR Multi-Seed Grid (Currently Running)
```bash
python tools/multi_seed_feature_grid.py
# Grid over: (lookbacks, quantile, enable_zones=True, tol, lookback) combinations
```

### Terminal 2: SR Multi-Seed Grid (After SR Training)
```bash
python tools/multi_seed_feature_grid_sr.py
# Grid over: (zone_tol, zone_lookback) combinations
```

### Terminal 3: Monitoring / Analysis
```bash
# Check BR progress
python -c "import pandas as pd; df=pd.read_csv('backtest/plots_oos/feature_grid_multi_seed/results.csv'); print(len(df), df['seed'].unique())"

# Check SR progress
python -c "import pandas as pd; df=pd.read_csv('backtest/plots_oos/feature_grid_multi_seed_sr/results.csv'); print(len(df), df['seed'].unique())"
```

---

## Cooperation Strategies (Detailed)

### 1. Alternation with Switch Period
- Use BR for 288 bars (1 day @ H1), then SR for 288 bars, repeat
- Rationale: allow each agent to develop conviction in its edge without interference

### 2. Voting / Consensus
- If BR and SR predict same action → execute
- If they disagree → hold (conservative) or use one as tiebreaker
- Measure: win rate improvement, drawdown reduction

### 3. Confidence Blending
- Extract action logits/probabilities from policy networks
- Weight actions by confidence: α×BR_action + (1-α)×SR_action
- Adaptive α based on recent Sharpe ratio

### 4. Regime-Based Routing
- Use BR in trending regimes (REGIME ∈ {0, 1})
- Use SR in ranging/crisis regimes (REGIME ∈ {2, 3})
- Test: can specialization reduce drawdown??

---

## Next Steps

1. **Complete current BR multi-seed grid** (25 rows)
2. **Train SR agent** with same data pipeline
3. **Complete SR multi-seed grid** (25 rows, tuning zone parameters)
4. **Analyze best configurations** for each agent
5. **Run cooperation evaluation** with alternation, voting, blending strategies
6. **Compare final metrics**: individual Sharpe vs. ensemble Sharpe
7. **(Optional) Meta-RL:** Train a "switcher" network to learn when to use BR vs. SR dynamically

---

## Files Reference

| File | Purpose |
|------|---------|
| `train/train_agent.py` | Main BR training |
| `train/train_agent_sr.py` | Main SR training (NEW) |
| `tools/multi_seed_feature_grid.py` | BR multi-seed tuning |
| `tools/multi_seed_feature_grid_sr.py` | SR multi-seed tuning (NEW) |
| `tools/dual_agent_cooperate.py` | Cooperation evaluation (NEW) |
| `backtest/out_of_sample_test.py` | Generic OOS evaluation (used by both) |
| `env/trading_env.py` | Environment (supports both BR and SR via config) |
| `utils/support_zones.py` | Zone detection utility |

---

**Last Updated:** March 4, 2026  
**Status:** Dual-system architecture implemented; ready for parallel training and evaluation

