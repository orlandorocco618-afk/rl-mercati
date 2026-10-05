import os
import json
import pandas as pd
import numpy as np

def analyze_actions(actions_csv: str = "backtest/data/xauusd_actions.csv",
                    initial_capital: float = 10000.0,
                    save_dir: str = "backtest/plots"):
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(actions_csv):
        raise FileNotFoundError(f"Actions CSV not found: {actions_csv}")

    df = pd.read_csv(actions_csv)

    # Ensure Time is datetime and prices are numeric
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
    for col in ["Open","High","Low","Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Simulate trades to compute per-trade PnL similar to TradingEnv
    position = 0
    entry_price = None
    capital = initial_capital
    trade_results = []
    per_step_equity = []

    for idx, row in df.iterrows():
        action = int(row.get("Action", 0))
        price = float(row.get("Close", np.nan))

        realized_pnl = 0.0

        # Safe: treat missing price as skip
        if np.isnan(price):
            per_step_equity.append(capital)
            continue

        if action == 1:  # LONG
            if position <= 0:
                if position == -1 and entry_price is not None:
                    realized_pnl += (entry_price - price) / entry_price
                    trade_results.append({"type":"cover_short","pnl":realized_pnl})
                position = 1
                entry_price = price

        elif action == 2:  # SHORT
            if position >= 0:
                if position == 1 and entry_price is not None:
                    realized_pnl += (price - entry_price) / entry_price
                    trade_results.append({"type":"close_long","pnl":realized_pnl})
                position = -1
                entry_price = price

        elif action == 3:  # CLOSE
            if position == 1 and entry_price is not None:
                realized_pnl += (price - entry_price) / entry_price
                trade_results.append({"type":"close_long","pnl":realized_pnl})
            elif position == -1 and entry_price is not None:
                realized_pnl += (entry_price - price) / entry_price
                trade_results.append({"type":"cover_short","pnl":realized_pnl})
            position = 0
            entry_price = None

        # Update capital by realized pnl (as percent)
        capital += realized_pnl * capital

        per_step_equity.append(capital)

    # If position left open at end, compute unrealized PnL (approx)
    # We'll not force-close; the equity series reflects realized PnL only.

    # Compute summary stats
    trades_df = pd.DataFrame(trade_results)
    total_trades = len(trades_df)
    profitable_trades = trades_df[trades_df["pnl"] > 0].shape[0] if total_trades>0 else 0
    pct_profitable = 100.0 * profitable_trades / total_trades if total_trades>0 else 0.0
    total_return_pct = 100.0 * (capital - initial_capital) / initial_capital

    # Per-action aggregates (sum PnL by type)
    per_type = trades_df.groupby("type")["pnl"].agg(["count","sum","mean"]).to_dict() if total_trades>0 else {}

    stats = {
        "initial_capital": initial_capital,
        "final_capital": capital,
        "total_return_pct": total_return_pct,
        "total_trades": total_trades,
        "profitable_trades": profitable_trades,
        "pct_profitable": pct_profitable,
        "per_type": per_type,
        "steps": len(df),
        "no_action_steps": int((df.get("Action",0)==0).sum())
    }

    # Save a JSON summary
    with open(os.path.join(save_dir, "analysis_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Try to create plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Equity curve
        plt.figure(figsize=(10, 4))
        plt.plot(df["Time"], per_step_equity)
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "equity_curve.png"))
        plt.close()

        # Action distribution
        plt.figure(figsize=(6,4))
        df["Action"].value_counts().sort_index().plot(kind="bar")
        plt.title("Action Distribution")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "action_distribution.png"))
        plt.close()

        # Per-trade PnL histogram
        if total_trades>0:
            plt.figure(figsize=(6,4))
            plt.hist(trades_df["pnl"].dropna(), bins=50)
            plt.title("Per-Trade PnL Distribution")
            plt.xlabel("PnL (fraction)")
            plt.ylabel("Trades")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "trade_pnl_hist.png"))
            plt.close()

            # Bar of sum PnL per type
            agg = trades_df.groupby("type")["pnl"].sum()
            plt.figure(figsize=(6,4))
            agg.plot(kind="bar")
            plt.title("Sum PnL per Trade Type")
            plt.ylabel("Sum PnL (fraction)")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "pnl_per_type.png"))
            plt.close()

    except Exception as e:
        print("Matplotlib not available or plotting failed:", e)

    print(f"Analysis complete. Summary saved to: {os.path.join(save_dir, 'analysis_summary.json')}")
    return stats


if __name__ == "__main__":
    try:
        stats = analyze_actions()
        print("Analysis stats:")
        print(stats)
    except Exception as e:
        print("Analysis failed:", e)
