#!/usr/bin/env python3
"""Enhanced test.py for CycleGAN (unpaired setting)
---------------------------------------------------
* Logs qualitative results to HTML + WandB
* Quantitative metrics designed for **unpaired** UV → H&E translation:
  ─ **FID** between generated‑H&E and real‑H&E (distribution realism)
  ─ **PSNR** between cycle‑reconstructed UV and original UV (content preservation)
* Optional CPU / GPU resource snapshots during inference
"""

import os
from datetime import datetime

import psutil                       # CPU / RAM
import GPUtil                       # GPU load / memory

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

# ── Metrics ──────────────────────────────────────────────────────
from torchmetrics.image import FrechetInceptionDistance, PeakSignalNoiseRatio
import torchvision.transforms as T

try:
    import wandb
except ImportError:
    wandb = None
    print("⚠️  wandb package not found — disabling WandB logging.")

# ---------------------------------------------------------------
# Helper: resource‑usage print‑out
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

    # Deterministic test‑time settings
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # ── WandB initialisation ────────────────────────────────────
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

    # ── Data + model setup ─────────────────────────────────────
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    device = model.device

    # ── Metric trackers ─────────────────────────────────────────
    fid_metric = FrechetInceptionDistance(feature=64).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    # tensor converter 0‑1, resize to crop_size to align shapes
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

        # ---------- Metric update ----------------------------------------
        # FID: compare generated H&E (fake_B) with real H&E (real_B)
        fake_he = visuals["fake_B"]
        real_he = visuals["real_B"]
        fake_t = to01(fake_he).unsqueeze(0).to(device)
        real_t = to01(real_he).unsqueeze(0).to(device)
        fid_metric.update(real_t * 255, real=True)
        fid_metric.update(fake_t * 255, real=False)

        # PSNR: cycle‑reconstructed UV (rec_A) vs original UV (real_A)
        rec_uv = visuals["rec_A"]
        real_uv = visuals["real_A"]
        rec_t = to01(rec_uv).unsqueeze(0).to(device)
        realuv_t = to01(real_uv).unsqueeze(0).to(device)
        psnr_metric.update(rec_t, realuv_t)

        # ---------- Save visuals ----------------------------------------
        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
            use_wandb=opt.use_wandb,
        )

        if run_name and i % LOG_EVERY == 0:
            wandb.log({
                "test_images": [wandb.Image(img, caption=lbl) for lbl, img in visuals.items()],
                "sample_index": i,
            })

        if i % RES_MON_EVERY == 0:
            monitor_resources()

    # ── Compute final metrics ───────────────────────────────────
    fid_val = fid_metric.compute().item()
    psnr_val = psnr_metric.compute().item()
    print(f"FID: {fid_val:.2f} | PSNR_cycle: {psnr_val:.2f} dB")

    if run_name:
        wandb.log({"metric/FID": fid_val, "metric/PSNR_cycle": psnr_val})

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

