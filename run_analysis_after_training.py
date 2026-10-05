#!/usr/bin/env python3
"""
AUTO-ANALYSIS SCRIPT
Run after training to automatically generate:
1. Actions CSV
2. Plots and HTML dashboard
3. Email notification (optional)
"""
import os
import time
import subprocess
import sys
from pathlib import Path

def wait_for_model(model_path="models/ppo_trading.zip", timeout_sec=18000):  # 5-hour timeout
    """Waits for the model to be saved"""
    print(f"\n⏳ Waiting for the model to be saved: {model_path}")
    start = time.time()
    while True:
        if os.path.exists(model_path):
            # Make sure the file is no longer being written (stable size)
            size1 = os.path.getsize(model_path)
            time.sleep(2)
            size2 = os.path.getsize(model_path)
            if size1 == size2:
                print(f"✅ Model saved!")
                return True
        
        elapsed = time.time() - start
        if elapsed > timeout_sec:
            print(f"❌ Timeout after {timeout_sec}s")
            return False
        
        print(f"  ... {int(elapsed)}s", end="\r")
        time.sleep(5)

def generate_actions():
    """Generates the actions CSV from the model"""
    print("\n📊 Generating the actions CSV...")
    os.environ['RL_QUICK_TEST'] = '1'  # Quick mode for speed
    try:
        from backtest.generate_actions import generate_actions
        generate_actions(model_path='models/ppo_trading.zip', 
                        output_path='backtest/data/xauusd_actions.csv')
        print("✅ Actions CSV generated")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def analyze_actions():
    """Generates plots and the HTML dashboard"""
    print("\n📈 Generating analysis and charts...")
    try:
        from backtest.analyze_actions_v2 import analyze_actions_v2
        stats = analyze_actions_v2(
            actions_csv='backtest/data/xauusd_actions.csv',
            initial_capital=10000.0,
            save_dir='backtest/plots'
        )
        print("✅ Analysis completed")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def open_dashboard():
    """Opens the HTML dashboard in the browser"""
    dashboard_path = os.path.abspath("backtest/plots/dashboard.html")
    print(f"\n🎯 Dashboard: {dashboard_path}")
    try:
        if sys.platform == "win32":
            os.startfile(dashboard_path)
        else:
            subprocess.Popen(['open', dashboard_path])
        print("✅ Browser opened")
    except Exception as e:
        print(f"⚠️  Error opening the browser: {e}")

if __name__ == "__main__":
    print("="*70)
    print("🚀 AUTO-ANALYSIS PIPELINE")
    print("="*70)
    
    # Step 1: Wait for the model
    if not wait_for_model():
        print("❌ Training not completed (timeout)")
        sys.exit(1)
    
    # Step 2: Generate actions
    if not generate_actions():
        print("❌ Failed: generate_actions")
        sys.exit(1)
    
    # Step 3: Analyze
    if not analyze_actions():
        print("❌ Failed: analyze_actions")
        sys.exit(1)
    
    # Step 4: Open the browser
    open_dashboard()
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED!")
    print("="*70)
    print(f"\n📂 Results in: backtest/plots/")
    print(f"📊 Dashboard: backtest/plots/dashboard.html")
