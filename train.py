import time
import datetime
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=None, help='Override number of DataLoader workers')
    cli_args, _ = parser.parse_known_args()

    opt = TrainOptions().parse()

    # DataLoader threads override for Colab
    import os
    if cli_args.threads is not None:
        opt.num_threads = cli_args.threads
        print(f"Overriding num_threads from CLI: {opt.num_threads}")
    elif 'COLAB_GPU' in os.environ and opt.num_threads == 4:
        opt.num_threads = 2
        print("Colab detected: setting DataLoader num_threads to 2")
    else:
        print(f"Using user-defined num_threads: {opt.num_threads}")

    # Auto-name WandB run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"breastUV2HE_{timestamp}"

    import wandb
    if opt.use_wandb:
        wandb.init(
            entity="jackiechanchunki2852002-king-s-college-london",
            project=opt.wandb_project_name,
            name=run_name,
            config=vars(opt),
            mode="online"
        )
        # Save WandB run ID to file for later retrieval
        with open("wandb_run_id.txt", "w") as f:
            f.write(wandb.run.id)

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    # Visual check of one batch
    import torchvision.utils as vutils
    import torchvision.transforms as transforms
    import numpy as np

    sample_batch = next(iter(dataset))
    if isinstance(sample_batch, dict):
        if 'A' in sample_batch:
            grid_a = vutils.make_grid(sample_batch['A'], nrow=4, normalize=True)
            plt.figure(figsize=(10, 10))
            plt.title("Sample Batch from trainA")
            plt.axis("off")
            plt.imshow(np.transpose(grid_a.cpu(), (1, 2, 0)))
            plt.show()
        if 'B' in sample_batch:
            grid_b = vutils.make_grid(sample_batch['B'], nrow=4, normalize=True)
            plt.figure(figsize=(10, 10))
            plt.title("Sample Batch from trainB")
            plt.axis("off")
            plt.imshow(np.transpose(grid_b.cpu(), (1, 2, 0)))
            plt.show()

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                # Manual WandB logging of scalar losses
                if opt.use_wandb:
                    log_data = {f"loss_{k}": float(v) for k, v in losses.items()}
                    log_data["epoch"] = epoch
                    log_data["iters"] = epoch_iter
                    wandb.log(log_data)

                    # New: log current visuals to WandB media
                    visuals = model.get_current_visuals()
                    img_logs = [wandb.Image(v, caption=k) for k, v in visuals.items()]
                    wandb.log({"generated_images": img_logs}, step=total_iters)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    # Fetch and plot WandB losses
    if opt.use_wandb:
        try:
            from wandb import Api
            api = Api()
            runs = api.runs(f"{opt.entity}/{opt.wandb_project_name}")
            run = next((r for r in runs if r.name == run_name), None)
            if run:
                history = run.history(samples=10000)
                loss_cols = [col for col in history.columns if col.startswith("loss_")]
                plt.figure(figsize=(15, 6))
                for col in loss_cols:
                    plt.plot(history[col].rolling(window=10).mean(), label=col)
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.title("Smoothed Training Losses Over Time")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("wandb_loss_plot.png")
                plt.show()
        except Exception as e:
            print(f"⚠️ Could not fetch WandB history: {e}")
