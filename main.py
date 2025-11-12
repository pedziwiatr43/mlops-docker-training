# main.py
import argparse
from train import run_training

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--checkpoint_dir", type=str, default="models/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="my-project")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_training(vars(args))
