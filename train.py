import argparse
import os
import time
from datetime import datetime

import torch
from tqdm import tqdm
import wandb

import psutil  # ✅ Resource monitoring
import GPUtil  # ✅ GPU monitoring

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# ✅ Resource monitoring function
def monitor_resources():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"🔍 GPU {gpu.id} ({gpu.name}) — Load: {gpu.load*100:.1f}%, Mem: {gpu.memoryUtil*100:.1f}%, Temp: {gpu.temperature}°C")
    except Exception as e:
        print("GPU info error:", e)
    print(f"🧠 CPU: {cpu:.1f}% | RAM: {mem:.1f}%")

def main():
    # Parse CLI and training options
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=None, help='Override DataLoader workers')
    cli_args, _ = parser.parse_known_args()

    opt = TrainOptions().parse()
    if cli_args.threads is not None:
        opt.num_threads = cli_args.threads
    elif 'COLAB_GPU' in os.environ and opt.num_threads == 4:
        opt.num_threads = 2

    print(f"Using num_threads = {opt.num_threads}")

    run_name = None
    if getattr(opt, 'use_wandb', False):
        # Authenticate and initialize wandb
        api_key = os.getenv('WANDB_API_KEY')
        wandb.login(key=api_key) if api_key else wandb.login()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        run_name = f"{opt.name}-{timestamp}"
        wandb.init(
            entity="jackiechanchunki2852002-king-s-college-london",
            project=opt.wandb_project_name,
            name=run_name,
            config=vars(opt),
            mode="online"
        )
        print(f"✅ W&B run started: {run_name}")

    # Create dataset and model
    dataset = create_dataset(opt)
    print(f"The number of training images = {len(dataset)}")
    model = create_model(opt)
    model.setup(opt)

    total_iters = 0
    max_epochs = opt.n_epochs + opt.n_epochs_decay

    for epoch in range(opt.epoch_count, max_epochs + 1):
        pbar = tqdm(dataset, desc=f"Epoch {epoch}/{max_epochs}")

        for data in pbar:
            total_iters += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # Monitor system resources
            if total_iters % 100 == 0:
                monitor_resources()

            # Log visuals to WandB
            if run_name and total_iters % opt.display_freq == 0:
                model.compute_visuals()
                visuals = model.get_current_visuals()
                img_logs = [wandb.Image(img, caption=label) for label, img in visuals.items()]
                wandb.log({"sample_images": img_logs}, step=total_iters)

            # Log losses to WandB
            if run_name and total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                wandb.log({f"loss/{k}": float(v) for k, v in losses.items()}, step=total_iters)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

        model.update_learning_rate()

    print("✅ Training complete.")

if __name__ == '__main__':
    main()
