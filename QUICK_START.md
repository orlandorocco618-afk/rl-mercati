# 🚀 QUICK START - Robust Error Handling

## 30 Seconds to Get Started

### 1. Install the Dependencies
```powershell
cd rl-mercati
pip install -r requirements.txt
pip install statsmodels==0.13.5
```

### 2. Check the Data
```powershell
ls data/
# Must show:
# - xauusd_h1_clean.csv
# - xauusd_d1_clean.csv
```

### 3. Run Training
```powershell
python train/train_agent.py
```

You'll see output with "[STEP X/8]", and if it fails it tells you EXACTLY where.

---

## 📝 What Happens Behind the Scenes

### Training (train_agent.py)
```
STEP 1: Load the H1 + D1 CSVs
STEP 2: Apply the Kalman filter
STEP 3: Create the Gymnasium environment
STEP 4: Normalize observations
STEP 5: Initialize the PPO model
STEP 6: Set up the checkpoint callback
STEP 7: TRAINING (300k timesteps)
STEP 8: Save the model
```

**If Step N fails:** you see an exact message like:
```
============================================================
[CRITICAL ERROR] in load_data
============================================================
Type: FileNotFoundError
Message: CSV file not found: data/xauusd_d1_clean.csv
```

### Backtest (backtest.py)
```
STEP 1: Load data
STEP 2: Create the environment
STEP 3: Load the trained model
STEP 4: Run the backtest
STEP 5: Compute statistics + plot
```

### Generate Actions (generate_actions.py)
```
STEP 1: Load data
STEP 2: Kalman filter
STEP 3: Create the environment
STEP 4: Load the model
STEP 5: Generate actions
STEP 6: Save the CSV
```

---

## 🎯 Full Workflow

### Day 1: Preparation
```powershell
# 1. Check the environment
python -c "from utils.data_utils import merge_h1_with_d1_regimes; df=merge_h1_with_d1_regimes(); print(f'✓ Data OK: {df.shape}')"

# 2. Run training (it takes a few hours)
python train/train_agent.py
# Wait until you see "✓ SUCCESS"
```

### Day 2: Test the Model
```powershell
# 3. Generate actions
python backtest/generate_actions.py

# 4. Backtest
python backtest/backtest.py

# 5. Backtrader backtest
python backtest/backtest_bt.py
```

---

## ⚠️ Common Errors and Fixes

### ❌ "CSV file not found"
```
Fix: 
  cd rl-mercati\data
  ls
  # Must show xauusd_h1_clean.csv and xauusd_d1_clean.csv
```

### ❌ "ModuleNotFoundError: statsmodels"
```
Fix:
  pip install statsmodels==0.13.5
```

### ❌ "CUDA out of memory"
```
Fix:
  Edit train_agent.py, line ~70:
  batch_size=64  # instead of 128
```

### ❌ "ValueError in vecnormalize"
```
Fix:
  Corrupted DataFrame. Check the CSV:
  python -c "import pandas as pd; df=pd.read_csv('data/xauusd_h1_clean.csv'); print(df.head())"
```

---

## 🎓 Expected Output

### ✅ Training Completed Successfully
```
============================================================
[STEP 7/8] Training PPO (300k timesteps)...
============================================================
Timesteps: 300000/300000 [████████████████████] 100%

============================================================
[STEP 8/8] Saving the model and the normalization...
============================================================
[OK] Model saved: models/ppo_trading.zip
[OK] Normalization saved: models/vecnormalize.pkl

======================================================================
✓ SUCCESS
======================================================================
Total time: 1234.56s
Status: ✓ SUCCESS
Model: ✓ READY
```

### ❌ Training Failed (Example)
```
============================================================
[STEP 1/8] Loading H1+D1 data...
============================================================
[CHECK FILE] data/xauusd_h1_clean.csv... ✓ FOUND
[CHECK FILE] data/xauusd_d1_clean.csv... ✗ NOT FOUND

============================================================
[CRITICAL ERROR] in file_check
============================================================
Type: FileNotFoundError
Message: [CRITICAL ERROR] CSV file not found: data/xauusd_d1_clean.csv

Traceback (most recent call last):
  ...

======================================================================
✗ FAILURE
======================================================================
Critical errors: 1
Status: ✗ FAILURE
```

---

## 🔄 If You Want to Change Parameters

### Slower Learning Rate
Edit [train/train_agent.py](train/train_agent.py#L70) line ~70:
```python
model = PPO(
    learning_rate=1e-4,  # ← Lower for slower convergence
    ...
)
```

### Fewer Timesteps (Quick Test)
Edit [train/train_agent.py](train/train_agent.py#L105) line ~105:
```python
model.learn(
    total_timesteps=10_000,  # ← Instead of 300_000, for testing
    ...
)
```

### Batch Size for a Limited GPU
Edit [train/train_agent.py](train/train_agent.py#L75) line ~75:
```python
model = PPO(
    batch_size=64,  # ← Lower it if CUDA runs out of memory
    ...
)
```

---

## 📊 Generated Output Files

### After Training
```
models/
├── ppo_trading.zip                 ← RL model
├── vecnormalize.pkl                ← Normalization
└── checkpoints/
    ├── ppo_trading_50000_steps.zip
    ├── ppo_trading_100000_steps.zip
    └── ...
```

### After the Backtest
```
backtest/
├── data/
│   └── xauusd_actions.csv          ← Generated actions
└── equity_curve.png                ← Chart
```

---

## ✨ Key Features

✅ **Never Crash**: if it fails, it prints exactly where and continues (when possible)  
✅ **Detailed Errors**: error type + message + full traceback  
✅ **Step-by-Step**: you see exactly where progress is ("[STEP X/8]")  
✅ **Final Report**: summary with success/failure and the accumulated errors  
✅ **Smart Recovery**: non-critical errors don't stop the program  
✅ **Production Ready**: ready for live deployment and automated trading  

---

## 🆘 If It Still Doesn't Work

Collect diagnostics:
```powershell
# 1. Test the imports
python -c "from utils.error_handler import *; print('✓ error_handler OK')"

# 2. Test the data
python -c "from utils.data_utils import merge_h1_with_d1_regimes; df=merge_h1_with_d1_regimes(); print(f'✓ Data: {df.shape}')"

# 3. Test the environment
python -c "from env.trading_env import TradingEnv; print('✓ TradingEnv OK')"

# 4. Test the model
python -c "from stable_baselines3 import PPO; print('✓ PPO OK')"

# 5. Save the log
python train/train_agent.py 2>&1 | tee training_log.txt
```

Then share the `training_log.txt` file.

---

**Version**: 1.0 - Error Handling Complete  
**Status**: ✅ Production Ready  
**Last Updated**: 2025-02-08
