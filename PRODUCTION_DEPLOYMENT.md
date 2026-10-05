# 🚀 PRODUCTION DEPLOYMENT GUIDE - RL-Mercati
## Robust Error Handling for Live Trading

---

## ✅ What Has Been Implemented

### 1. **Complete Error Handler Module** (`utils/error_handler.py`)
- ✅ Detailed logging with traceback
- ✅ Automatic data validation
- ✅ `@handle_errors` decorator for functions
- ✅ `safe_execute()` function with fallback
- ✅ Formatted report at the end of the run
- ✅ Console colors for readability

### 2. **ROBUST Versions of All the Main Modules**
Files with the `_robust.py` suffix implement full error handling:

| Original file | Robust version | Description |
|---|---|---|
| `train/train_agent.py` | `train/train_agent_robust.py` | PPO training with 8 monitored steps |
| `backtest/backtest.py` | `backtest/backtest_robust.py` | RL backtest with statistics |
| `backtest/generate_actions.py` | `backtest/generate_actions_robust.py` | Action generation from the model |
| `backtest/backtest_bt.py` | `backtest/backtest_bt_robust.py` | Backtest with Backtrader |

### 3. **Every Step Reports:**
- **The exact name of the step** where it fails
- **The error type** (FileNotFoundError, TypeError, etc.)
- **A detailed message** about what went wrong
- **The full traceback** for debugging
- **The recovery action** (continue or stop)

---

## 🎯 How to Use

### Step 1: Install the Dependencies
```powershell
cd rl-mercati
pip install -r requirements.txt
```

### Step 2: Choose Which Script to Run

#### **TRAINING (first run)**
```powershell
python train/train_agent_robust.py
```
**Expected output:**
```
============================================================
[STEP 1] Loading H1+D1 data...
============================================================
[LOAD CSV] Reading: data/xauusd_h1_clean.csv
[LOAD CSV] ✓ CSV loaded correctly: 8760 rows

[STEP 2] Applying the Kalman filter...
[Kalman] ✓ Done

[STEP 3] Creating the RL environment...
...
```

If it fails, you'll see:
```
============================================================
[CRITICAL ERROR] in load_data
============================================================
Type: FileNotFoundError
Message: [CRITICAL ERROR] CSV file not found: data/xauusd_h1_clean.csv
   Current folder: rl-mercati
   Files in data/: ['xauusd_h1_clean.csv', 'xauusd_d1_clean.csv']
...
```

#### **GENERATE ACTIONS (after training)**
```powershell
python backtest/generate_actions_robust.py
```

#### **BACKTEST (test the model)**
```powershell
python backtest/backtest_robust.py
```

#### **BACKTRADER BACKTEST (alternative)**
```powershell
python backtest/backtest_bt_robust.py
```

---

## 🔍 Reading the Errors

### Example 1: Missing File
```
❌ [CRITICAL ERROR] CSV file not found: data/xauusd_d1_clean.csv
   Current folder: rl-mercati
   Files in data/: ['xauusd_h1_clean.csv']
```
**Fix:** the file `xauusd_d1_clean.csv` doesn't exist in `data/`. Copy it there.

### Example 2: Error During Training
```
============================================================
[CRITICAL ERROR] in model_training
============================================================
Type: RuntimeError
Message: ...CUDA out of memory...
```
**Fix:** lower `batch_size` in `train_agent_robust.py` from 128 to 64.

### Example 3: Actions Not Generated
```
❌ [ERROR] action_generation(step 15000)
Message: ...ValueError...
```
**Fix:** if it fails at step 15000 of 20000, the model may be incomplete. Check `models/ppo_trading.zip`.

---

## 📊 Final Report

Every script ends with a structured report:

**Success:**
```
======================================================================
✓ SUCCESS
======================================================================

Statistics:
   Timesteps: 300000
   Model: models/ppo_trading.zip
   Status: ✓ READY FOR DEPLOYMENT

Total time: 1234.56s
```

**Failure:**
```
======================================================================
✗ FAILURE
======================================================================

Errors found:
   1. [load_data] FileNotFoundError: CSV file not found
   2. [merge_h1_d1] ValueError: Merge failed

Total time: 45.23s
```

---

## 🛡️ Recovery Strategy

### For NON-Critical Errors
- ⚠️  Print a warning
- ✓ Continue with a fallback/default values
- 📝 Record the error for review

**Examples:**
- Callback setup fails → continue training without checkpoints
- Plot generation fails → continue anyway
- Normalization save fails → continue with the model

### For CRITICAL Errors
- ❌ Print the full error
- 🛑 Stop the program
- 📋 Show exactly what didn't work

**Examples:**
- CSV not found
- PPO model can't be loaded
- Environment creation fails
- The training loop goes wrong

---

## 🔄 Full Deployment Cycle

### Day 1: First Training
```bash
# Prepare the environment and the data
pip install -r requirements.txt

# Run the robust training
python train/train_agent_robust.py
# ✓ Produces: models/ppo_trading.zip, models/vecnormalize.pkl

# Quick test
python backtest/backtest_robust.py
# ✓ Prints statistics
```

### Day 2: Generate Actions for the Backtest
```bash
# Generate actions from the model
python backtest/generate_actions_robust.py
# ✓ Produces: backtest/data/xauusd_actions.csv

# Backtest with Backtrader
python backtest/backtest_bt_robust.py
# ✓ Shows P&L, charts, metrics
```

### Day 3+: Live Deployment (when ready)
- Use `models/ppo_trading.zip` for real-time predictions
- Monitor with the same error handlers
- Log every trade in real time
- Retrain periodically

---

## ⚙️ Advanced Configuration

### Change the Training Hyperparameters
File: `train/train_agent_robust.py`, line ~163

```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,      # Lower for slower convergence
    batch_size=128,           # Lower if out of memory
    n_steps=2048,            # Raise for more stability
    ...
)
```

### Change the Training Timesteps
```powershell
# Reduced (quick test)
python -c "from train.train_agent_robust import train_agent; train_agent(timesteps=10_000)"

# Standard
python train/train_agent_robust.py
# Uses the default 300_000

# Extended (for better convergence)
python -c "from train.train_agent_robust import train_agent; train_agent(timesteps=1_000_000)"
```

### Debugging a Single Step
```powershell
# Test data loading only
python -c "from backtest.backtest_robust import load_data; df=load_data(); print(df.shape)"

# Test the environment only
python -c "from env.trading_env import TradingEnv; print('✓ Env OK')"
```

---

## 📝 Pre-Live-Deployment Checklist

- [ ] Training completed with `train_agent_robust.py` without errors
- [ ] Positive backtest with `backtest_robust.py`
- [ ] Actions generated with `generate_actions_robust.py`
- [ ] Model saved: `models/ppo_trading.zip`
- [ ] Normalization saved: `models/vecnormalize.pkl`
- [ ] H1 and D1 CSVs checked and present
- [ ] PC with a GPU available (or CPU if acceptable)
- [ ] Stable connection to the broker/API
- [ ] Stop-loss and risk management configured
- [ ] Paper trading completed

---

## 🆘 Quick Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: statsmodels` | `pip install statsmodels==0.13.5` |
| `CUDA out of memory` | Lower `batch_size` from 128 to 64 |
| `CSV file not found` | Check the path with `ls data/` |
| Training is very slow | Raise `batch_size` or lower `timesteps` |
| Backtest doesn't produce an equity curve | Check that `min_rows` isn't too high |
| Actions not saved | Check the permissions on `backtest/data/` |
| Chart doesn't show up | Normal in headless mode, check the generated PNG file |

---

## 📞 Contact & Logging

Every error found is printed to the console with:
- **TIMESTAMP** (when it failed)
- **LOCATION** (file and function)
- **ERROR TYPE** (FileNotFoundError, ValueError, etc.)
- **FULL TRACEBACK** (for debugging)

To save the log to a file:
```powershell
python train/train_agent_robust.py > training_log_$(Get-Date -Format "yyyy-MM-dd_HHmmss").txt 2>&1
```

---

**Version:** 1.0  
**Date:** 2026-02-08  
**Status:** ✅ PRODUCTION READY
