import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from bt_strategy import RLActionStrategy
from utils.error_handler import (
    log_error, print_step_summary, print_final_report,
    check_file_exists, validate_dataframe
)


def run_backtest_bt(
    data_path="data/xauusd_h1_clean.csv",
    actions_path="backtest/data/xauusd_actions.csv"
):
    print("Loading MT5 data...")
    df = pd.read_csv(data_path)
    df.columns = [c.capitalize() for c in df.columns]
    
    if "Tickvolume" in df.columns:
        df.rename(columns={"Tickvolume": "TickVolume"}, inplace=True)
    
    df["Time"] = pd.to_datetime(df["Time"])
    df.set_index("Time", inplace=True)

    print("Loading RL actions...")
    actions_df = pd.read_csv(actions_path)

    # Backtrader feed
    data = bt.feeds.PandasData(
        dataname=df,
        timeframe=bt.TimeFrame.Minutes,
        compression=5
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(data)

    cerebro.addstrategy(
        RLActionStrategy,
        actions_df=actions_df
    )

    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0002)

    print("Running the backtest...")
    result = cerebro.run()

    print("Backtest completed.")
    print(f"Final equity: {cerebro.broker.getvalue():.2f}")

    cerebro.plot(style="candlestick")


if __name__ == "__main__":
    errors = []
    status = "FAILURE"
    
    try:
        print("="*60)
        print("[STEP 1/7] Checking files...")
        print("="*60)
        
        try:
            check_file_exists("data/xauusd_h1_clean.csv", critical=True)
            check_file_exists("backtest/data/xauusd_actions.csv", critical=True)
            print_step_summary("file_check", status="OK")
        except Exception as e:
            error_msg = log_error(e, "file_check", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 2/7] Loading H1 data...")
        print("="*60)
        
        df = None
        try:
            df = pd.read_csv("data/xauusd_h1_clean.csv")
            df.columns = [c.capitalize() for c in df.columns]
            
            if "Tickvolume" in df.columns:
                df.rename(columns={"Tickvolume": "TickVolume"}, inplace=True)
            
            df["Time"] = pd.to_datetime(df["Time"])
            df.set_index("Time", inplace=True)
            
            validate_dataframe(df, critical=True)
            print_step_summary("load_h1_data", status="OK", rows=len(df))
        except Exception as e:
            error_msg = log_error(e, "load_h1_data", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 3/7] Loading RL actions...")
        print("="*60)
        
        actions_df = None
        try:
            actions_df = pd.read_csv("backtest/data/xauusd_actions.csv")
            validate_dataframe(actions_df, critical=True)
            print_step_summary("load_actions", status="OK", rows=len(actions_df))
        except Exception as e:
            error_msg = log_error(e, "load_actions", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 4/7] Setup Backtrader feed...")
        print("="*60)
        
        data = None
        try:
            data = bt.feeds.PandasData(
                dataname=df,
                timeframe=bt.TimeFrame.Minutes,
                compression=60
            )
            print_step_summary("backtrader_feed", status="OK")
        except Exception as e:
            error_msg = log_error(e, "backtrader_feed", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 5/7] Setup Cerebro...")
        print("="*60)
        
        cerebro = None
        try:
            cerebro = bt.Cerebro()
            cerebro.adddata(data)
            
            cerebro.addstrategy(
                RLActionStrategy,
                actions_df=actions_df
            )
            
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.0002)
            
            print_step_summary("cerebro_setup", status="OK")
        except Exception as e:
            error_msg = log_error(e, "cerebro_setup", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 6/7] Running the Backtrader backtest...")
        print("="*60)
        
        initial_cash = 10000.0
        final_cash = None
        
        try:
            result = cerebro.run()
            final_cash = cerebro.broker.getvalue()
            pnl = final_cash - initial_cash
            pnl_pct = (pnl / initial_cash) * 100
            
            print(f"Starting cash: {initial_cash:.2f}")
            print(f"Final cash: {final_cash:.2f}")
            print(f"PNL: {pnl:.2f} ({pnl_pct:.2f}%)")
            
            print_step_summary("backtest_execution", status="OK", 
                             pnl=pnl, pnl_pct=pnl_pct)
        except Exception as e:
            error_msg = log_error(e, "backtest_execution", critical=True)
            errors.append(error_msg)
            raise
        
        print("\n" + "="*60)
        print("[STEP 7/7] Plot equity curve...")
        print("="*60)
        
        try:
            cerebro.plot(style="candlestick")
            print("[OK] Plot completed")
            print_step_summary("plotting", status="OK")
        except Exception as e:
            error_msg = log_error(e, "plotting", critical=False)
            errors.append(error_msg)
            print("[WARN] Plot failed but the backtest completed")
        
        status = "SUCCESS"
        
    except Exception as e:
        error_msg = log_error(e, "backtest_bt_main", critical=True)
        errors.append(error_msg)
        status = "FAILURE"
    
    print_final_report(status=status, errors=errors)
