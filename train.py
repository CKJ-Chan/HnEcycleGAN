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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=None,
                        help='Override number of DataLoader workers')
    cli_args, _ = parser.parse_known_args()

    # Parse training options
    opt = TrainOptions().parse()

    # Optional Weights & Biases setup
    run_name = None
    if getattr(opt, 'use_wandb', False):
        api_key = os.getenv('WANDB_API_KEY')
        if api_key:
            wandb.login(key=api_key)
        else:
            wandb.login()
        # Create a timestamp for readability (e.g. 2025-04-27_15:33,18)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M,%S')
        run_name = f"{opt.name}-{timestamp}"
        # Initialize W&B
        wandb.init(
            entity="jackiechanchunki2852002-king-s-college-london",
            project=opt.wandb_project_name,
            name=run_name,
            config=vars(opt),
            mode="online"
        )
        # Immediately write run identifiers so files exist upfront
        run_id = wandb.run.id
        id_path = os.path.join(os.getcwd(), "wandb_run_id.txt")
        name_path = os.path.join(os.getcwd(), "wandb_run_name.txt")
        with open(id_path, "w") as f_id:
            f_id.write(run_id)
        with open(name_path, "w") as f_name:
            f_name.write(run_name)
        print(f"✔️ W&B run files written to {id_path} and {name_path}")
        # Print W&B run URL for easy access
        print(f"🚀 W&B run: {run_name} -> {wandb.run.url}")
        # System & Hardware Metrics: watch model for gradients and weights will be set after model setup

    # Override num_threads if provided via CLI or Colab
    if cli_args.threads is not None:

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    # Model setup
    model = create_model(opt)
    model.setup(opt)

    # If using W&B, watch the model and save run identifiers
    if run_name is not None:
        wandb.watch(model, log="all", log_freq=opt.print_freq)
        # Save run identifiers for later lookup
        with open("wandb_run_id.txt", "w") as f_id:
            f_id.write(wandb.run.id)
        with open("wandb_run_name.txt", "w") as f_name:
            f_name.write(run_name)

    # Track iterations and best loss
    total_iters = 0
    best_total_loss = float('inf')

    # Training loop
    max_epochs = opt.n_epochs + opt.n_epochs_decay
    for epoch in range(opt.epoch_count, max_epochs + 1):
        pbar = tqdm(dataset, desc=f"Epoch {epoch}/{max_epochs}")

        for data in pbar:
            total_iters += opt.batch_size

            # Forward and backward
            model.set_input(data)
            model.optimize_parameters()

            # Log sample images to W&B
            if run_name and total_iters % opt.display_freq == 0:
                model.compute_visuals()
                visuals = model.get_current_visuals()
                img_logs = [wandb.Image(img, caption=label) for label, img in visuals.items()]
                wandb.log({"sample_images": img_logs}, step=total_iters)

            # Log losses to W&B and console
            if run_name and total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                wandb.log({f"loss/{k}": float(v) for k, v in losses.items()}, step=total_iters)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

                # Best-model checkpoint
                total_loss = sum(losses.values())
                if total_loss < best_total_loss:
                    best_total_loss = total_loss
                    print(f"🏆 New best model at iter {total_iters} (loss={total_loss:.4f}). Saving...")
                    model.save_networks('best')
                    # Log best-model checkpoint as W&B artifact
                    artifact = wandb.Artifact(f"{run_name}-best", type="model")
                    artifact.add_dir(os.path.join("checkpoints", opt.name))
                    wandb.log_artifact(artifact)

            # Save latest model checkpoint
            if total_iters % opt.save_latest_freq == 0:
                print(f"💾 Saving latest model at iter {total_iters}")
                suffix = f"iter_{total_iters}" if opt.save_by_iter else 'latest'
                model.save_networks(suffix)
                # Log latest checkpoint as W&B artifact
                artifact = wandb.Artifact(f"{run_name}-latest-{suffix}", type="model")
                artifact.add_dir(os.path.join("checkpoints", opt.name))
                wandb.log_artifact(artifact)

        # End of epoch: update LR and save epoch checkpoints
        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:
            print(f"💾 Saving model at epoch {epoch}")
            model.save_networks('latest')
            model.save_networks(str(epoch))
            # Log epoch checkpoint as W&B artifact
            artifact = wandb.Artifact(f"{run_name}-epoch_{epoch}", type="model")
            artifact.add_dir(os.path.join("checkpoints", opt.name))
            wandb.log_artifact(artifact)

    print("Training complete.")


if __name__ == '__main__':
    main()
