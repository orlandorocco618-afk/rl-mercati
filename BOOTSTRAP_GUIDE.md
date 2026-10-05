# 📋 PHASE 1 BOOTSTRAP STRATEGY - GUIDE TO EXITING THE FORCING

## Context
Training currently uses `allow_hold_when_flat=False` to force the model to explore.
This is a **temporary tool** to keep the policy from getting stuck in HOLD.

---

## Bootstrap EXIT Metrics

After every training phase, check the following KPIs (available in `backtest/plots/ltst/analysis_summary.json`):

| Metric | EXIT threshold | Meaning |
|---------|-------------|------------|
| **Total Return** | > +5.0% | Credible profits (vs. initial arbitrage) |
| **Win Rate** | > 50% | At least half of the trades are profitable |
| **Total Trades** | > 100 | Enough training data |
| **Max Drawdown** | < -10% | Maximum loss under control |
| **No-Action Steps** | 30-50% | The policy still recognizes when HOLD would be appropriate |

### Simultaneous EXIT conditions:
- ✅ Return > +5% **AND**
- ✅ Win Rate > 50% **AND**
- ✅ Trades > 100

### If they are NOT met:
→ Keep training with `allow_hold_when_flat=False` for another 50-100k timesteps
→ Check the intermediate checkpoints every 10-20k timesteps

---

## Suggested Training Cycles

### Phase 1: Bootstrap (CURRENT)
```
- allow_hold_when_flat = False
- timesteps: 50-100k
- Checkpoint every 10-20k
- Exit: when Return > 5% AND Win Rate > 50% AND Trades > 100
```

### Phase 2: Fine-tuning (NEXT)
```
- allow_hold_when_flat = True  (re-enables HOLD)
- Load the model from Phase 1
- timesteps: 100-200k more
- Checkpoint every 20-50k
- Goal: keep the profits and learn to wait (appropriate HOLD)
```

### Phase 3: Production (FINAL)
```
- allow_hold_when_flat = True
- Load the best model from Phase 2
- Validate on the test set (out-of-sample)
- Deploy
```

---

## How to Check Progress

### 1. **Automatic** - Post-Training:
After every training run, the script automatically generates:
```
backtest/plots/ltst/
├── 01_equity_curve_trades.png    → Equity + trade markers
├── 02_action_distribution.png    → Action counts (HOLD/LONG/SHORT/CLOSE)
├── 03_pnl_per_type.png           → P/L per trade type
├── 04_trade_pnl_histogram.png    → P/L distribution
├── 05_position_over_time.png     → Positions over time
├── analysis_summary.json         → JSON metrics
└── dashboard.html                → Visual dashboard
```

### 2. **Manual** - Inspect the JSON:
```python
import json
with open("backtest/plots/ltst/analysis_summary.json") as f:
    stats = json.load(f)
    print(f"Return: {stats['total_return_pct']}%")
    print(f"Win Rate: {stats['pct_profitable']}%")
    print(f"Trades: {stats['total_trades']}")
```

---

## Checkpoint Archive

Every chart generation is archived automatically:
```
backtest/plots/chkpt/
├── 20260209_230008_01_equity_curve_trades.png
├── 20260209_230008_02_action_distribution.png
├── ...
├── TIMESTAMP_*.png  (all previous generations, dated)
```

You can compare performance across runs by reviewing the timestamped checkpoints.

---

## Warning Signs

🚨 If you see these patterns, **keep forcing**:

1. **Action Distribution ALL HOLD**: the policy isn't exploring
   → Increase entropy (`ent_coef` in PPO)
   → Keep forcing

2. **Win Rate < 30%**: random / unprofitable trades
   → The model is oscillating
   → Option: reset the model and start over with different hyperparameters

3. **Negative return (< -5%)**: significant losses
   → Possibly overfitting to the specific training window
   → Increase exploration vs. exploitation

---

## Decision: When to Disable Forcing

**Once you reach**:
- Total Return > 5%
- Win Rate > 50%
- Total Trades > 100

**Do this**:
1. Change `train/train_agent.py` → `allow_hold_when_flat=True`
2. Increase timesteps (e.g. 150-200k more) to fine-tune
3. Relaunch training from Phase 2
4. Review the results after Phase 2 to confirm the policy has learned to wait

---

## Checklist for Exiting the Bootstrap

- [ ] Total Return > +5.0%
- [ ] Win Rate > 50%
- [ ] Total Trades >= 100
- [ ] Max Drawdown < -10%
- [ ] No-Action Steps ratio in the 30-50% range
- [ ] Visual inspection of the equity curve (consistent trend)
- [ ] Change `allow_hold_when_flat=False` → `allow_hold_when_flat=True`
- [ ] Relaunch Phase 2 training with more timesteps
- [ ] Document the Phase 1 results before moving on

---

**Guide created on**: 2026-02-09  
**Current phase**: Bootstrap (Phase 1)  
**Last check**: [UPDATE MANUALLY]
