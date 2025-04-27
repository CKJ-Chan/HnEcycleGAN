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
    if getattr(opt, 'use_wandb', False):
        api_key = os.getenv('WANDB_API_KEY')
        if api_key:
            wandb.login(key=api_key)
        else:
            wandb.login()
        # Create a timestamp for readability (e.g. 2025-04-27_15:33,18)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M,%S')
        run_name = f"{opt.name}-{timestamp}"
        wandb.init(
            entity="jackiechanchunki2852002-king-s-college-london",
            project=opt.wandb_project_name,
            name=run_name,
            config=vars(opt),
            mode="online"
        )
        # Save run identifiers for later lookup
        with open("wandb_run_id.txt", "w") as f_id:
            f_id.write(wandb.run.id)
        with open("wandb_run_name.txt", "w") as f_name:
            f_name.write(run_name)

    # Override num_threads if provided via CLI or Colab
    if cli_args.threads is not None:
        opt.num_threads = cli_args.threads
        print(f"Overriding num_threads from CLI: {opt.num_threads}")
    elif 'COLAB_GPU' in os.environ and opt.num_threads == 4:
        opt.num_threads = 2
        print("Colab detected: setting DataLoader num_threads to 2")
    else:
        print(f"Using user-defined num_threads: {opt.num_threads}")

    # Create dataset (handles resizing/cropping internally)
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    # Model setup
    model = create_model(opt)
    model.setup(opt)

    # Iteration tracking
    total_iters = 0
    best_total_loss = float('inf')

    # Training loop over epochs
    max_epochs = opt.n_epochs + opt.n_epochs_decay
    for epoch in range(opt.epoch_count, max_epochs + 1):
        pbar = tqdm(dataset, desc=f"Epoch {epoch}/{max_epochs}")

        for data in pbar:
            total_iters += opt.batch_size

            # Forward/backward
            model.set_input(data)
            model.optimize_parameters()

            # Log sample images to W&B
            if getattr(opt, 'use_wandb', False) and total_iters % opt.display_freq == 0:
                model.compute_visuals()
                visuals = model.get_current_visuals()
                img_logs = [wandb.Image(img, caption=label) for label, img in visuals.items()]
                wandb.log({"sample_images": img_logs}, step=total_iters)

            # Log losses to W&B and console
            if getattr(opt, 'use_wandb', False) and total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                wandb.log({f"loss/{k}": float(v) for k, v in losses.items()}, step=total_iters)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

                # Best-model checkpoint
                total_loss = sum(losses.values())
                if total_loss < best_total_loss:
                    best_total_loss = total_loss
                    print(f"🏆 New best model at iter {total_iters} (loss={total_loss:.4f}). Saving...")
                    model.save_networks('best')

            # Save latest model checkpoint
            if total_iters % opt.save_latest_freq == 0:
                print(f"💾 Saving latest model at iter {total_iters}")
                suffix = f"iter_{total_iters}" if opt.save_by_iter else 'latest'
                model.save_networks(suffix)

        # End of epoch: update LR and save epoch checkpoints
        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:
            print(f"💾 Saving model at epoch {epoch}")
            model.save_networks('latest')
            model.save_networks(str(epoch))

    print("Training complete.")


if __name__ == '__main__':
    main()
