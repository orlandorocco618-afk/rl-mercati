import os
import subprocess
import sys

if __name__ == "__main__":
    # Force CPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("Starting RL training on CPU...")
    subprocess.run([sys.executable, "train/train_agent.py"])
