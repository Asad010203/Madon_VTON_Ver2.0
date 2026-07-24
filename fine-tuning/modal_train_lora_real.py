"""
Real LoRA Training for IDM-VTON on Shorts Dataset

Trains LoRA adapters on the IDM-VTON UNet cross-attention layers (to_q, to_k,
to_v, to_out.0). Only LoRA params are trainable — base weights are frozen so
the model preserves its original knowledge.

Uses the same Modal image as modal_idm_inference.py (SDXL / diffusers 0.25.0)
so the produced checkpoint plugs directly into that inference pipeline.

Usage:
    modal run modal_train_lora_real.py::train_lora
"""

from __future__ import annotations

from pathlib import Path
import modal


IDM_VOLUME    = "idm-vton-weights"
IDM_MOUNT     = "/idm-weights"
MODEL_PATH    = "/idm-weights/IDM-VTON"

app                 = modal.App("idm-vton-lora-train")
idm_volume          = modal.Volume.from_name(IDM_VOLUME)
datasets_volume     = modal.Volume.from_name("idm-vton-datasets", create_if_missing=True)
checkpoints_volume  = modal.Volume.from_name("idm-vton-checkpoints", create_if_missing=True)

# Match modal_idm_inference.py so the LoRA checkpoint is drop-in compatible.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "git-lfs",
        "libgl1", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev",
        "build-essential", "ninja-build",
    )
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install([
        "diffusers==0.25.0",
        "huggingface_hub==0.25.0",
        "transformers==4.36.2",
        "accelerate==0.26.1",
        "peft==0.11.1",
        "numpy>=1.26,<3",
        "scipy>=1.10",
        "scikit-image>=0.22",
        "opencv-python==4.7.0.72",
        "pillow>=9.4,<11",
        "einops==0.7.0",
        "matplotlib==3.7.4",
        "onnxruntime>=1.19",
        "omegaconf",
        "safetensors",
        "tqdm==4.64.1",
    ])
    .run_commands(
        "git clone --depth 1 https://github.com/yisol/IDM-VTON /app/IDM-VTON",
    )
    .env({"PYTHONPATH": "/app/IDM-VTON:/app/IDM-VTON/gradio_demo"})
)


@app.function(
    image=image,
    gpu="A100",  # need memory headroom for SDXL UNet + LoRA training
    volumes={
        IDM_MOUNT: idm_volume,
        "/data": datasets_volume,
        "/checkpoints": checkpoints_volume,
    },
    timeout=7200,
)
def train_lora(
    epochs: int = 20,
    batch_size: int = 1,
    lr: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    save_every: int = 5,
):
    """Train LoRA on IDM-VTON UNet cross-attention using shorts dataset."""
    import os, sys, json, time
    from datetime import datetime

    os.chdir("/app/IDM-VTON")
    for p in ("/app/IDM-VTON", "/app/IDM-VTON/gradio_demo"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from PIL import Image
    import numpy as np
    from torchvision import transforms

    from transformers import (
        AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )
    from diffusers import AutoencoderKL, DDPMScheduler
    from src.unet_hacked_tryon   import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from peft import LoraConfig, get_peft_model

    print("=" * 80)
    print("IDM-VTON Real LoRA Training")
    print("=" * 80)
    print(f"Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"LoRA rank: {lora_rank}  |  alpha: {lora_alpha}")

    device = torch.device("cuda")
    dtype  = torch.bfloat16   # bf16: training-stable + saves VRAM vs fp32

    # ── 1. Load frozen base pipeline components ──────────────────────────────
    print("\n[1/5] Loading base IDM-VTON components (frozen)...")
    t0 = time.time()

    unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet", torch_dtype=dtype)
    vae  = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.float32)  # VAE needs fp32 for encode stability
    text_encoder_one = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder="text_encoder_2", torch_dtype=dtype)
    image_encoder    = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder="image_encoder", torch_dtype=dtype)
    unet_encoder     = UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder="unet_encoder", torch_dtype=dtype)
    tokenizer_one    = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer",   use_fast=False)
    tokenizer_two    = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2", use_fast=False)
    scheduler        = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    for m in (vae, text_encoder_one, text_encoder_two, image_encoder, unet_encoder):
        m.requires_grad_(False)
        m.to(device)
    unet.requires_grad_(False)

    # Disable IP-adapter branch during training (LoRA lives on cross-attn, which
    # doesn't depend on ip_image_proj). Config is restored implicitly at
    # inference time since we save only LoRA weights, not the config.
    unet.config.encoder_hid_dim_type = None
    print("  Disabled ip_image_proj for training (base config unaffected at inference)")

    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── 2. Inject LoRA into UNet cross-attention ─────────────────────────────
    print("\n[2/5] Injecting LoRA adapters into UNet cross-attention...")
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    # Keep LoRA params in same dtype as base (bf16) so linear ops match
    for n, p in unet.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(dtype)
    unet.to(device)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in unet.parameters())
    print(f"  Trainable LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ── 3. Dataset ───────────────────────────────────────────────────────────
    print("\n[3/5] Loading dataset...")
    manifest_path = Path("/data/shorts_dataset/manifest.jsonl")
    triplets = [json.loads(l) for l in manifest_path.open()]
    print(f"  {len(triplets)} triplets")

    IMG_H, IMG_W = 1024, 768
    tensor_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class ShortsTripletDataset(Dataset):
        def __init__(self, triplets, root):
            self.triplets, self.root = triplets, Path(root)
        def __len__(self):
            return len(self.triplets)
        def __getitem__(self, idx):
            t = self.triplets[idx]
            person  = Image.open(self.root / t["person"]).convert("RGB").resize((IMG_W, IMG_H))
            garment = Image.open(self.root / t["garment"]).convert("RGB").resize((IMG_W, IMG_H))
            target  = Image.open(self.root / t["target"]).convert("RGB").resize((IMG_W, IMG_H))
            pose    = Image.open(self.root / t["pose"]).convert("RGB").resize((IMG_W, IMG_H))
            mask    = Image.open(self.root / t["mask"]).convert("L").resize((IMG_W, IMG_H))
            return {
                "person":  tensor_tfm(person),
                "garment": tensor_tfm(garment),
                "target":  tensor_tfm(target),
                "pose":    tensor_tfm(pose),
                "mask":    transforms.ToTensor()(mask),
                "prompt":  f"model is wearing {t.get('garment_type','a garment').replace('_',' ')}",
            }

    dataset = ShortsTripletDataset(triplets, "/data/shorts_dataset")
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ── 4. Optimizer + prompt encoding helper ────────────────────────────────
    print("\n[4/5] Setting up optimizer...")
    optimizer = AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )

    def encode_prompt(prompt: str):
        """Encode text prompt via both SDXL tokenizers/encoders."""
        tokens_1 = tokenizer_one(prompt, padding="max_length", max_length=tokenizer_one.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(device)
        tokens_2 = tokenizer_two(prompt, padding="max_length", max_length=tokenizer_two.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            emb_1  = text_encoder_one(tokens_1, output_hidden_states=True).hidden_states[-2]
            out_2  = text_encoder_two(tokens_2, output_hidden_states=True)
            emb_2  = out_2.hidden_states[-2]
            pooled = out_2[0]
        return torch.cat([emb_1, emb_2], dim=-1), pooled

    # ── 5. Training loop ─────────────────────────────────────────────────────
    print("\n[5/5] Training...\n")
    losses_per_epoch = []
    best_loss = float("inf")
    checkpoint_dir = Path("/checkpoints")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = None

    for epoch in range(1, epochs + 1):
        unet.train()
        epoch_losses = []
        t_epoch = time.time()

        for step, batch in enumerate(loader, 1):
            person  = batch["person"].to(device, dtype)
            garment = batch["garment"].to(device, dtype)
            target  = batch["target"].to(device, dtype)
            pose    = batch["pose"].to(device, dtype)
            mask    = batch["mask"].to(device, dtype)

            # Encode target + masked-person + pose to latents (VAE fp32 for stability)
            with torch.no_grad():
                target_lat = vae.encode(target.float()).latent_dist.sample() * vae.config.scaling_factor

                # Masked person: zero out the garment region so the model learns to inpaint it
                mask_3ch = mask.expand(-1, 3, -1, -1)   # (B,3,H,W) in [0,1]
                masked_person = person * (1.0 - mask_3ch)
                masked_lat = vae.encode(masked_person.float()).latent_dist.sample() * vae.config.scaling_factor

                pose_lat = vae.encode(pose.float()).latent_dist.sample() * vae.config.scaling_factor

                target_lat = target_lat.to(dtype)
                masked_lat = masked_lat.to(dtype)
                pose_lat   = pose_lat.to(dtype)

                # Downsample mask to latent resolution
                mask_lat = torch.nn.functional.interpolate(mask.to(dtype), size=target_lat.shape[-2:], mode="nearest")

            # Sample noise + timestep
            noise     = torch.randn_like(target_lat)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (target_lat.shape[0],), device=device).long()
            noisy = scheduler.add_noise(target_lat, noise, timesteps)

            # IDM-VTON UNet expects 13 channels: [noisy(4), masked(4), mask(1), pose(4)]
            unet_input = torch.cat([noisy, masked_lat, mask_lat, pose_lat], dim=1)

            # Encode prompt
            prompt_embeds, pooled = encode_prompt(batch["prompt"][0])

            add_time_ids = torch.tensor([[IMG_H, IMG_W, 0, 0, IMG_H, IMG_W]],
                                        device=device, dtype=dtype).repeat(target_lat.shape[0], 1)
            added_cond = {"text_embeds": pooled.to(dtype), "time_ids": add_time_ids}

            model_pred = unet(
                unet_input,
                timesteps,
                encoder_hidden_states=prompt_embeds.to(dtype),
                added_cond_kwargs=added_cond,
            ).sample

            loss = F.mse_loss(model_pred.float(), noise.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in unet.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            print(f"  epoch {epoch}/{epochs}  step {step}/{len(loader)}  "
                  f"loss={loss.item():.4f}", flush=True)

        avg = float(np.mean(epoch_losses))
        losses_per_epoch.append(avg)
        print(f"  → epoch {epoch} done in {time.time()-t_epoch:.1f}s  avg_loss={avg:.4f}\n")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = checkpoint_dir / f"idmvton_lora_shorts_{stamp}_e{epoch}.pt"
            state = {k: v.cpu() for k, v in unet.state_dict().items() if "lora_" in k}
            torch.save({
                "lora_state_dict": state,
                "lora_config":     {"r": lora_rank, "alpha": lora_alpha,
                                    "target_modules": ["to_q","to_k","to_v","to_out.0"]},
                "epoch":           epoch,
                "avg_loss":        avg,
                "losses":          losses_per_epoch,
            }, ckpt_path)
            print(f"  💾 Saved: {ckpt_path.name}  ({len(state)} LoRA tensors)\n", flush=True)
            checkpoints_volume.commit()
            final_path = ckpt_path
            if avg < best_loss:
                best_loss = avg

    print("=" * 80)
    print(f"✅ Training complete. Best loss: {best_loss:.4f}")
    print(f"Final checkpoint: {final_path}")
    print("=" * 80)
    return str(final_path) if final_path else ""


if __name__ == "__main__":
    train_lora()
