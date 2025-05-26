#!/usr/bin/env python3
"""Enhanced test.py for CycleGAN
-------------------------------------------------
Adds:
• WandB logging (images + artefact)
• CPU/GPU resource monitoring
• Quantitative metrics (SSIM & FID)
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

# ── Metrics
from torchmetrics.image import StructuralSimilarityIndexMeasure, FrechetInceptionDistance
import torchvision.transforms as T

try:
    import wandb
except ImportError:
    wandb = None
    print("⚠️  wandb package not found — disabling WandB logging.")

# ---------------------------------------------------------------
# Helper: resource‑usage print‑out every N iterations
# ---------------------------------------------------------------

def monitor_resources():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(
                f"🔍 GPU {gpu.id} ({gpu.name}) — Load: {gpu.load*100:.1f}%, "
                f"Mem: {gpu.memoryUtil*100:.1f}%, Temp: {gpu.temperature}°C"
            )
    except Exception as e:
        print("GPU info error:", e)
    print(f"🧠 CPU: {cpu:.1f}% | RAM: {mem:.1f}%")

# ---------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------

def main():
    opt = TestOptions().parse()

    # Hard‑coded test‑time tweaks
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # ── WandB init ──────────────────────────────────────────────
    run_name = None
    if getattr(opt, "use_wandb", False) and wandb is not None:
        api_key = os.getenv("WANDB_API_KEY")
        try:
            wandb.login(key=api_key) if api_key else wandb.login()
        except Exception:
            print("⚠️  WandB login failed — disabling WandB.")
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

    # ── Data + model ────────────────────────────────────────────
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    # ── Metric trackers ─────────────────────────────────────────
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(model.device)
    fid_metric = FrechetInceptionDistance(feature=64).to(model.device)
    to01 = T.Compose([T.ToPILImage(), T.Resize(opt.crop_size), T.ToTensor()])

    # ── HTML result dir ─────────────────────────────────────────
    web_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch or 'latest'}")
    if opt.load_iter > 0:
        web_dir += f"_iter{opt.load_iter}"
    print("Creating web directory:", web_dir)
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")

    LOG_EVERY = 5
    RES_MON_EVERY = 20

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if i % LOG_EVERY == 0:
            print(f"Processing ({i:04d})‑th image … {img_path}")

        # ---------- Metric update ----------
        fake_he = visuals["fake_B"]   # UV → H&E
        real_he = visuals["real_B"]
        fake_t = to01(fake_he).unsqueeze(0).to(model.device)
        real_t = to01(real_he).unsqueeze(0).to(model.device)
        ssim_metric.update(fake_t, real_t)
        fid_metric.update(real_t * 255, real=True)
        fid_metric.update(fake_t * 255, real=False)

        # ---------- Save visuals ----------
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

        if run_name and i % LOG_EVERY == 0:
            wandb.log({
                "test_images": [wandb.Image(img, caption=lbl) for lbl, img in visuals.items()],
                "sample_index": i,
            })

        if i % RES_MON_EVERY == 0:
            monitor_resources()

    # ── Compute final metrics ───────────────────────────────────
    ssim_val = ssim_metric.compute().item()
    fid_val = fid_metric.compute().item()
    print(f"SSIM: {ssim_val:.4f} | FID: {fid_val:.2f}")

    if run_name:
        wandb.log({"metric/SSIM": ssim_val, "metric/FID": fid_val})

    webpage.save()

    if run_name:
        art = wandb.Artifact("cycleGAN-test-results", type="inference")
        art.add_dir(web_dir)
        wandb.log_artifact(art)
        print(f"🗂️  Results uploaded to W&B artefact: {web_dir}")
        wandb.finish()

    print("✅ Testing complete.")


if __name__ == "__main__":
    main()
