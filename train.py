import argparse
import os
import time
import torch
from tqdm import tqdm
import wandb
from datetime import datetime

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# DEBUG: print current working dir
print("🚨 train.py working dir:", os.getcwd())
print("🚨 Contents before anything:", os.listdir(os.getcwd()))

# Compute the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=None,
                        help='Override number of DataLoader workers')
    cli_args, _ = parser.parse_known_args()

    # Parse training options
    opt = TrainOptions().parse()

    run_name = None
    if getattr(opt, 'use_wandb', False):
        # log in
        api_key = os.getenv('WANDB_API_KEY')
        if api_key:
            wandb.login(key=api_key)
        else:
            wandb.login()

        # create a human-readable name
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M,%S')
        run_name = f"{opt.name}-{timestamp}"

        # initialize W&B run
        wandb.init(
            entity="jackiechanchunki2852002-king-s-college-london",
            project=opt.wandb_project_name,
            name=run_name,
            config=vars(opt),
            mode="online"
        )

        # ✅ Write run ID & name into the repo root directory
        if wandb.run is not None:
            try:
                BASE_DIR = os.getcwd()  # current working directory
                run_id_path = os.path.join(BASE_DIR, "wandb_run_id.txt")
                run_name_path = os.path.join(BASE_DIR, "wandb_run_name.txt")

                with open(run_id_path, "w") as f_id:
                    f_id.write(wandb.run.id)
                with o
