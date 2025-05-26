#!/usr/bin/env python3
"""Enhanced CycleGAN training script with:
1. WandB logging of losses, images, and artifacts
2. GPU/CPU resource monitoring every 100 iterations
3. Automatic tracking of *best* model (lowest epoch‑average generator loss)
   – Saved locally as checkpoints/<exp>/best_*  and uploaded as a WandB artifact
4. Upload of the *latest* checkpoint folder at the end of training

Assumes CycleGAN/Pix2Pix codebase from https://github.com/CKJ-Chan/HnEcycleGAN
"""

import argparse
import os
from datetime import datetime
import shutil

import torch
from tqdm import tqdm
import wandb

import psutil         # Resource monitoring
import GPUtil         # GPU monitoring

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# ---------------------------------------------------------------------------
# Helper: print CPU / GPU utilisation every N iterations
# ---------------------------------------------------------------------------

def monitor_resources() -> None:
    """Print a one‑line snapshot of CPU, RAM, and each GPU."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(
                f"🔍 GPU {gpu.id} ({gpu.name}) — "
                f"Load: {gpu.load * 100:.1f}%, "
                f"Mem: {gpu.memoryUtil * 100:.1f}%, "
                f"Temp: {gpu.temperature}°C"
            )
    except Exception as e:
        print("GPU info error:", e)
    print(f"🧠 CPU: {cpu:.1f}% | RAM: {mem:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1) CLI + default options
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=None, help="Override DataLoader workers")
    cli_args, _ = parser.parse_known_args()

    opt = TrainOptions().parse()  # loads default + command‑line opts

    # Adjust num_threads for Colab stability if user didn't override
    if cli_args.threads is not None:
        opt.num_threads = cli_args.threads
    elif "COLAB_GPU" in os.environ and opt.num_threads == 4:
        opt.num_threads = 2
    print(f"Using num_threads = {opt.num_threads}")

    # ------------------------------------------------------------------
    # 2) WandB initialisation (optional)
    # ------------------------------------------------------------------
    wandb_run = None
    if getattr(opt, "use_wandb", False):
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key) if api_key else wandb.login()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        run_name = f"{opt.name}-{timestamp}"
        wandb_run = wandb.init(
            entity="jackiechanchunki2852002-king-s-college-london",
            project=opt.wandb_project_name,
            name=run_name,
            config=vars(opt),
            mode="online",
        )
        print(f"✅ W&B run started: {run_name}")

    # ------------------------------------------------------------------
    # 3) Data + Model
    # ------------------------------------------------------------------
    dataset = create_dataset(opt)
    print(f"The number of training images = {len(dataset)}")
    model = create_model(opt)
    model.setup(opt)

    # ------------------------------------------------------------------
    # 4) Training loop
    # ------------------------------------------------------------------
    total_iters = 0
    max_epochs = opt.n_epochs + opt.n_epochs_decay

    # Trackers for best model (lowest generator loss per epoch)
    best_score = float("inf")
    best_epoch = None

    for epoch in range(opt.epoch_count, max_epochs + 1):
        pbar = tqdm(dataset, desc=f"Epoch {epoch}/{max_epochs}")

        epoch_loss_sum = 0.0
        epoch_steps = 0

        for data in pbar:
            total_iters += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # Grab losses for logging + best‑model metric
            losses = model.get_current_losses()  # dict of floats / tensors
            loss_val = float(sum(v for v in losses.values()))  # aggregate generator + discriminator
            epoch_loss_sum += loss_val
            epoch_steps += 1

            # Resource monitor every 100 iterations
            if total_iters % 100 == 0:
                monitor_resources()

            # ---- WandB: images -------------------------------------------------
            if wandb_run and total_iters % opt.display_freq == 0:
                model.compute_visuals()
                visuals = model.get_current_visuals()
                img_logs = [wandb.Image(img, caption=label) for label, img in visuals.items()]
                wandb.log({"sample_images": img_logs}, step=total_iters)

            # ---- WandB: losses ------------------------------------------------
            if wandb_run and total_iters % opt.print_freq == 0:
                wandb.log({f"loss/{k}": float(v) for k, v in losses.items()}, step=total_iters)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

        # -------- Epoch ends -------------------------------------------------
        model.update_learning_rate()

        epoch_loss_avg = epoch_loss_sum / max(1, epoch_steps)
        if wandb_run:
            wandb.log({"val/epoch_total_loss": epoch_loss_avg}, step=total_iters)

        # --- Check for new best -------------------------------------------
        if epoch_loss_avg < best_score:
            best_score, best_epoch = epoch_loss_avg, epoch
            print(f"🎯 New best model at epoch {epoch}: avg total loss {best_score:.4f}")

            # 1) Save local checkpoint with tag 'best'
            model.save_networks("best")

            # 2) Upload checkpoint folder as WandB artifact
            if wandb_run:
                art = wandb.Artifact(
                    name=f"cycleGAN-{opt.name}-best",
                    type="model",
                    metadata={"epoch": epoch, "avg_total_loss": best_score},
                )
                art.add_dir(f"./checkpoints/{opt.name}")
                wandb.log_artifact(art)

    # ------------------------------------------------------------------
    # 5) Save & log the latest checkpoint after all epochs
    # ------------------------------------------------------------------
    model.save_networks("latest")
    if wandb_run:
        latest_art = wandb.Artifact(
            name=f"cycleGAN-{opt.name}-latest",
            type="model",
        )
        latest_art.add_dir(f"./checkpoints/{opt.name}")
        wandb.log_artifact(latest_art)

    print("✅ Training complete.")


if __name__ == "__main__":
    main()
