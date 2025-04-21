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

    opt = TrainOptions().parse 

    # ✅ Allow user to control number of DataLoader workers for optimal Colab performance
    import os
    if cli_args.threads is not None:
        opt.num_threads = cli_args.threads
        print(f"Overriding num_threads from CLI: {opt.num_threads}")
    elif 'COLAB_GPU' in os.environ and opt.num_threads == 4:
        opt.num_threads = 2  # Safe default for Colab if user didn't override
        print("Colab detected: setting DataLoader num_threads to 2")
    else:
        print(f"Using user-defined num_threads: {opt.num_threads}")

    # 🏢 Auto-name the WandB run with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"breastUV2HE_{timestamp}"

    import wandb
    if opt.use_wandb:
        wandb.init(
            entity="jackiechanchunki2852002-king-s-college-london",  # ✅ Make sure this matches your WandB username
            project=opt.wandb_project_name,
            name=run_name,
            config=vars(opt),  # logs all training args
            mode="online"
        )

    dataset = create_dataset(opt)  # create dataset
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    # ✅ Visualize 1 batch to verify DataLoader
    import torchvision.utils as vutils
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
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

    model = create_model(opt)      # create model
    model.setup(opt)               # setup model
    visualizer = Visualizer(opt)  # visualizer
    total_iters = 0                # training iteration counter

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

                # ✅ Manual wandb logging to ensure it works
                if opt.use_wandb:
                    log_data = {f"loss_{k}": float(v) for k, v in losses.items()}
                    log_data["epoch"] = epoch
                    log_data["iters"] = epoch_iter
                    wandb.log(log_data)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # ✅ Update learning rate AFTER training loop (PyTorch recommendation)
        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
