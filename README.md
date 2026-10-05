# rl-mercati — a foundation for building RL trading systems on gold (XAUUSD)

A solid foundation for building a profitable trading system: a **causal RL
environment** (decision on the closed candle, execution at the next open, real
commission/spread/slippage), **regime detection**, **support/resistance zones**,
and the **validation tools** (walk-forward, multi-seed, cost stress tests) you need
before trusting any edge. Plug in your own features or strategy and test them properly.

The included agent is a PPO model (stable-baselines3) trained on hourly XAUUSD
candles, with Kalman-filtered prices and regime-aware position sizing.

> ⚠️ This is a learning/research project, **not** financial advice and not a
> production trading system. Backtests are easy to fool yourself with — a big part
> of this repo is exactly about catching that.

## What's inside

| Folder | What it does |
|---|---|
| `env/trading_env.py` | Gymnasium environment: XAUUSD H1, actions LONG / SHORT / CLOSE / HOLD, t+1 execution, costs, drawdown stop, regime and dynamic sizing |
| `utils/` | indicators (z-score, RSI, ATR), 1-D Kalman filter, regime classification, support/resistance zones, exposure scaler, error handling |
| `train/` | PPO training (`train_agent.py`, `train_agent_sr.py`), iterative bootstrap, phase-2 fine-tuning, time-chunked and roll-forward validation, multi-seed OOS |
| `backtest/` | backtests (custom + Backtrader), action analysis, crisis-period analysis, regime-system comparison, walk-forward evaluation, cost stress tests, HTML report generators |
| `tools/` | debugging helpers |

The `*.md` files in the root are my working notes: roadmap, regime-detection
analysis, deployment checklist, fix reports.

## Pipeline

1. Load H1 + D1 candles and apply a Kalman filter to the close.
2. Compute features: returns, z-scores, RSI, ATR, regime label, distance to
   support/resistance zones.
3. Train PPO (300k timesteps by default) with checkpoints, observation normalization
   and multiple seeds.
4. Backtest on held-out periods and stress the result: higher costs, delayed
   execution, different seeds, time chunks.

## Findings so far

- Early runs looked great (+41.9% return, 54.6% win rate, 465 trades), but that
  test window (2006–2008) overlapped the training window (2004–2008), so it was
  **not** a real out-of-sample result. That's why most of the later work is about
  validation.
- The 2008 crisis is a regime shift: average ATR goes from ~8.7 (2004–2008) to
  ~21–22.5 afterwards, and agents trained on the calm period diverged numerically
  (NaN logits) on post-2008 data. Normalization and regime features were added to
  deal with it — see `REGIME_DETECTION_ANALYSIS.md` and
  `TIME_CHUNKED_ANALYSIS_REPORT.md`.
- Cost stress tests and multi-seed runs are the reality check: an edge that
  disappears with 2× spread or with a different seed isn't an edge.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# put your candles in data/ (see "Data" below), then:
python train/train_agent.py     # train
python backtest/backtest.py     # backtest + plots
```

### Data

Price data is not included in the repo. The code expects two CSV files in `data/`:
`xauusd_h1_clean.csv` (hourly) and `xauusd_d1_clean.csv` (daily), with at least
`Date/Time, Open, High, Low, Close` columns. Any XAUUSD export from your broker or
a free historical-data site works after cleaning duplicates and gaps.

## Tech

Python · Gymnasium · stable-baselines3 (PPO) · pandas · NumPy · scikit-learn ·
statsmodels · ta · matplotlib · Backtrader

## License

MIT — see [LICENSE](LICENSE).
