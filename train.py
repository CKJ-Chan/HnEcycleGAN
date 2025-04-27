import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

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
        # Load API key from environment to skip interactive prompt in Colab
        api_key = os.getenv('WANDB_API_KEY')
        if api_key:
            wandb.login(key=api_key)
        else:
            wandb.login()
        # Initialize W&B run
        run_name = f"{opt.name}_{int(time.time())}"
        wandb.init(
            entity="jackiechanchunki2852002-king-s-college-london",
            project=opt.wandb_project_name,
            name=run_name,
            config=vars(opt),
            mode="online"
        )
        # Save WandB run ID
        with open("wandb_run_id.txt", "w") as f:
            f.write(wandb.run.id)

    # Override num_threads if provided via CLI or Colab
    if cli_args.threads is not None:
        opt.num_threads = cli_args.threads
        print(f"Overriding num_threads from CLI: {opt.num_threads}")
    elif 'COLAB_GPU' in os.environ and opt.num_threads == 4:
        opt.num_threads = 2
        print("Colab detected: setting DataLoader num_threads to 2")
    else:
        print(f"Using user-defined num_threads: {opt.num_threads}")

    # Dataset and DataLoader
    dataset = create_dataset(opt)
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_threads,
        pin_memory=True,
        prefetch_factor=2,
    )
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    # Model setup
    model = create_model(opt)
    model.setup(opt)

    # Track iterations and best loss
    total_iters = 0
    best_total_loss = float('inf')

    # Training loop
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_iter = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}")

        for data in pbar:
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # Forward and backward
            model.set_input(data)
            model.optimize_parameters()

            # Log sample images to W&B
            if getattr(opt, 'use_wandb', False) and total_iters % opt.display_freq == 0:
                model.compute_visuals()
                visuals = model.get_current_visuals()
                img_logs = [wandb.Image(img, caption=label) for label, img in visuals.items()]
                wandb.log({"sample_images": img_logs}, step=total_iters)

            # Logging losses
            if getattr(opt, 'use_wandb', False) and total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                wandb.log({f"loss/{k}": float(v) for k, v in losses.items()}, step=total_iters)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

                # Check for best model
                total_loss = sum(losses.values())
                if total_loss < best_total_loss:
                    best_total_loss = total_loss
                    print(f"🏆 New best model at iter {total_iters} (loss={total_loss:.4f}). Saving...")
                    model.save_networks('best')

            # Save latest checkpoint
            if total_iters % opt.save_latest_freq == 0:
                print(f"💾 Saving latest model at iter {total_iters}")
                suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                model.save_networks(suffix)

        # Update learning rate and save per-epoch checkpoints
        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:
            print(f"💾 Saving model at epoch {epoch}")
            model.save_networks('latest')
            model.save_networks(str(epoch))

    print("Training complete.")


if __name__ == '__main__':
    main()
