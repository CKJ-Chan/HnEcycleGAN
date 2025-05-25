#!/usr/bin/env python3
"""
Enhanced test.py with WandB monitoring and optional resource tracking.
This version mirrors the logging conveniences you added to train.py so that
inference runs ("test" phase) are also tracked, visualised and reproducible.

Key additions
-------------
1. **WandB login / run initialisation** with timestamp‑based run names.
2. **Image logging** — every few test samples are pushed as `wandb.Image` arrays.
3. **Resource monitoring** (CPU/GPU/RAM) at configurable intervals.
4. **Automatic HTML + artefact upload** so results survive beyond the Colab VM.

Usage example
-------------
```bash
python enhanced_test.py \
  --dataroot ./datasets/Breast_UV2HE/testA \
  --name breastUV2HE \
  --model test \
  --use_wandb \
  --wandb_project_name CycleGAN-BreastUV2HE
```
You can still pass all the usual `TestOptions` arguments.
"""
import os
import time
from datetime import datetime

import psutil  # ✔️ CPU/RAM monitoring
import GPUtil  # ✔️ GPU monitoring

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:  # graceful degradation if wandb isn’t installed
    wandb = None
    print("⚠️  wandb package not found — disabling WandB logging.")


# ---------------------------------------------------------------
# Helper: resource‑usage print‑out every N iterations
# ---------------------------------------------------------------

def monitor_resources():
    """Print a one‑line snapshot of CPU, RAM and every visible GPU."""
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


# ---------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------

def main():
    opt = TestOptions().parse()  # ↳ inherits --use_wandb, --wandb_project_name, etc.

    # Hard‑coded test‑time tweaks (same as the original script)
    opt.num_threads = 0           # deterministic order
    opt.batch_size = 1            # test code only supports BS=1
    opt.serial_batches = True     # no shuffling
    opt.no_flip = True            # no random flip
    opt.display_id = -1           # disable visdom

    # ----------------------------------
    # WandB initialisation (if enabled)
    # ----------------------------------
    run_name = None
    if getattr(opt, "use_wandb", False) and wandb is not None:
        # Attempt login (API key can live in env or be interactive)
        api_key = os.getenv("WANDB_API_KEY")
        try:
            wandb.login(key=api_key) if api_key else wandb.login()
        except wandb.UsageError:
            print("⚠️  WandB login failed — continuing without WandB.")
            opt.use_wandb = False
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            run_name = f"{opt.name}-test-{timestamp}"
            wandb.init(
                entity=getattr(opt, "wandb_entity", None),
                project=getattr(opt, "wandb_project_name", "CycleGAN-Test"),
                name=run_name,
                config=vars(opt),
                mode="online",
            )
            print(f"✅ W&B test run started: {run_name}")

    # ----------------------------------
    # Data + model setup
    # ----------------------------------
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()  # ensure eval mode for BN / Dropout layers

    # ----------------------------------
    # Result directory (HTML website)
    # ----------------------------------
    web_dir = os.path.join(
        opt.results_dir,
        opt.name,
        f"{opt.phase}_{opt.epoch if opt.epoch else 'latest'}",
    )
    if opt.load_iter > 0:
        web_dir = f"{web_dir}_iter{opt.load_iter}"

    print("Creating web directory:", web_dir)
    webpage = html.HTML(
        web_dir,
        f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}",
    )

    # ----------------------------------
    # Inference loop
    # ----------------------------------
    LOG_EVERY = 5        # → push an image + resource stats every N samples
    RES_MON_EVERY = 20   # → print CPU/GPU stats every M samples

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if i % LOG_EVERY == 0:
            print(f"Processing ({i:04d})‑th image … {img_path}")

        # Save to HTML (and WandB inside util if opt.use_wandb=True)
        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
            use_wandb=opt.use_wandb,
        )

        # ✚ Explicit WandB logging (similar to train.py)
        if run_name and i % LOG_EVERY == 0:
            wandb.log(
                {
                    "test_images": [wandb.Image(img, caption=lbl) for lbl, img in visuals.items()],
                    "sample_index": i,
                }
            )

        # ↳ Occasional resource read‑out
        if i % RES_MON_EVERY == 0:
            monitor_resources()

    # ----------------------------------
    # Finalise
    # ----------------------------------
    webpage.save()

    if run_name:
        # Archive the HTML folder as an artefact so it’s always retrievable.
        artefact = wandb.Artifact("cycleGAN-test-results", type="inference")
        artefact.add_dir(web_dir)
        wandb.log_artifact(artefact)
        print(f"🗂️  Results uploaded to W&B artefact: {web_dir}")
        wandb.finish()

    print("✅ Testing complete.")


if __name__ == "__main__":
    main()
