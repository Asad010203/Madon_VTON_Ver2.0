"""
LoRA training on IDM-VTON's `unet_encoder` (the reference / garment-encoder UNet).

Unlike the hacked main UNet, `unet_encoder` is a straightforward SDXL UNet used
to extract garment features that get injected into the try-on UNet. Fine-tuning
it via LoRA makes those features shorts-specific, which propagates through to
inference output.

Training task: standard diffusion noise-prediction on garment images alone
("a photo of shorts, {desc}"). Only cross-attention LoRA weights are updated.
Base weights stay frozen so nothing else in the model is disturbed.

Usage:
    modal run modal_train_lora_encoder.py::train_encoder_lora
"""

from __future__ import annotations

from pathlib import Path
import modal


IDM_VOLUME    = "idm-vton-weights"
IDM_MOUNT     = "/idm-weights"
MODEL_PATH    = "/idm-weights/IDM-VTON"

app                 = modal.App("idm-vton-encoder-lora")
idm_volume          = modal.Volume.from_name(IDM_VOLUME)
datasets_volume     = modal.Volume.from_name("idm-vton-datasets", create_if_missing=True)
checkpoints_volume  = modal.Volume.from_name("idm-vton-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "libgl1", "libglib2.0-0",
                 "libsm6", "libxext6", "libxrender-dev",
                 "build-essential", "ninja-build")
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install([
        "diffusers==0.25.0", "huggingface_hub==0.25.0",
        "transformers==4.36.2", "accelerate==0.26.1", "peft==0.11.1",
        "numpy>=1.26,<3", "pillow>=9.4,<11", "einops==0.7.0",
        "safetensors", "tqdm==4.64.1", "omegaconf",
    ])
    .run_commands("git clone --depth 1 https://github.com/yisol/IDM-VTON /app/IDM-VTON")
    .env({"PYTHONPATH": "/app/IDM-VTON:/app/IDM-VTON/gradio_demo"})
)


@app.function(
    image=image,
    gpu="A100",
    volumes={IDM_MOUNT: idm_volume, "/data": datasets_volume, "/checkpoints": checkpoints_volume},
    timeout=7200,
)
def train_encoder_lora(
    epochs: int = 30,
    batch_size: int = 1,
    lr: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    save_every: int = 10,
):
    """Train LoRA on IDM-VTON's unet_encoder (garment reference UNet)."""
    import os, sys, json, time
    from datetime import datetime

    os.chdir("/app/IDM-VTON")
    for p in ("/app/IDM-VTON", "/app/IDM-VTON/gradio_demo"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from PIL import Image
    from torchvision import transforms

    from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
    from diffusers import AutoencoderKL, DDPMScheduler
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm
    from peft import LoraConfig, get_peft_model

    print("=" * 80)
    print("IDM-VTON Encoder-UNet LoRA Training (Garment Branch)")
    print("=" * 80)
    print(f"Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"LoRA rank: {lora_rank}  |  alpha: {lora_alpha}")

    device = torch.device("cuda")
    dtype  = torch.bfloat16

    # ── Load frozen components ────────────────────────────────────────────────
    print("\n[1/4] Loading base components (frozen)...")
    t0 = time.time()

    unet = UNetGarm.from_pretrained(MODEL_PATH, subfolder="unet_encoder", torch_dtype=dtype)
    vae  = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.float32)
    text_encoder_one = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder="text_encoder_2", torch_dtype=dtype)
    tokenizer_one    = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer",   use_fast=False)
    tokenizer_two    = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2", use_fast=False)
    scheduler        = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    for m in (vae, text_encoder_one, text_encoder_two):
        m.requires_grad_(False); m.to(device)
    unet.requires_grad_(False)

    # Disable ip_image_proj branch — the encoder UNet's config may or may not have it,
    # but disabling ensures a clean cross-attention forward for training.
    if getattr(unet.config, "encoder_hid_dim_type", None) == "ip_image_proj":
        unet.config.encoder_hid_dim_type = None
        print("  Disabled ip_image_proj on encoder UNet")

    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── LoRA injection ────────────────────────────────────────────────────────
    print("\n[2/4] Injecting LoRA on cross-attention (to_q/k/v/out.0)...")
    lora_cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha,
                          target_modules=["to_q","to_k","to_v","to_out.0"],
                          lora_dropout=0.0, bias="none")
    unet = get_peft_model(unet, lora_cfg)
    for _, p in unet.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(dtype)
    unet.to(device)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in unet.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ── Dataset (garment images only — encoder learns garment features) ──────
    print("\n[3/4] Loading dataset...")
    manifest_path = Path("/data/shorts_dataset/manifest.jsonl")
    triplets = [json.loads(l) for l in manifest_path.open()]
    print(f"  {len(triplets)} triplets → training on {len(triplets)} garment images")

    IMG_H, IMG_W = 1024, 768
    tensor_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    class GarmentDataset(Dataset):
        def __init__(self, triplets, root):
            self.triplets, self.root = triplets, Path(root)
        def __len__(self):
            return len(self.triplets)
        def __getitem__(self, idx):
            t = self.triplets[idx]
            g = Image.open(self.root / t["garment"]).convert("RGB").resize((IMG_W, IMG_H))
            gtype = t.get("garment_type", "garment").replace("_", " ")
            sku   = t.get("_sku_garment", "").split("/")[-1].lower()
            return {
                "garment": tensor_tfm(g),
                "prompt":  f"a photo of {gtype} shorts {sku}".strip(),
            }

    loader = DataLoader(GarmentDataset(triplets, "/data/shorts_dataset"),
                        batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW([p for p in unet.parameters() if p.requires_grad],
                      lr=lr, weight_decay=1e-4)

    def encode_prompt(prompt: str):
        t1 = tokenizer_one(prompt, padding="max_length", max_length=tokenizer_one.model_max_length,
                           truncation=True, return_tensors="pt").input_ids.to(device)
        t2 = tokenizer_two(prompt, padding="max_length", max_length=tokenizer_two.model_max_length,
                           truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            emb1  = text_encoder_one(t1, output_hidden_states=True).hidden_states[-2]
            out2  = text_encoder_two(t2, output_hidden_states=True)
            emb2  = out2.hidden_states[-2]
            pool  = out2[0]
        return torch.cat([emb1, emb2], dim=-1), pool

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[4/4] Training...\n")
    losses_per_epoch = []
    best_loss = float("inf")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = None

    for epoch in range(1, epochs + 1):
        unet.train()
        epoch_losses = []
        t_epoch = time.time()

        for step, batch in enumerate(loader, 1):
            garm = batch["garment"].to(device, dtype)

            with torch.no_grad():
                lat = vae.encode(garm.float()).latent_dist.sample() * vae.config.scaling_factor
                lat = lat.to(dtype)

            noise     = torch.randn_like(lat)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (lat.shape[0],), device=device).long()
            noisy = scheduler.add_noise(lat, noise, timesteps)

            prompt_embeds, pooled = encode_prompt(batch["prompt"][0])
            add_time_ids = torch.tensor([[IMG_H, IMG_W, 0, 0, IMG_H, IMG_W]],
                                        device=device, dtype=dtype).repeat(lat.shape[0], 1)
            added_cond = {"text_embeds": pooled.to(dtype), "time_ids": add_time_ids}

            # Hacked garmnet UNet returns (UNet2DConditionOutput, garment_features)
            unet_out = unet(noisy, timesteps,
                            encoder_hidden_states=prompt_embeds.to(dtype),
                            added_cond_kwargs=added_cond)
            model_pred = (unet_out[0].sample if isinstance(unet_out, tuple)
                          else unet_out.sample)

            loss = F.mse_loss(model_pred.float(), noise.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in unet.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            if step % 5 == 0 or step == len(loader):
                print(f"  epoch {epoch}/{epochs}  step {step}/{len(loader)}  "
                      f"loss={loss.item():.4f}", flush=True)

        avg = float(np.mean(epoch_losses))
        losses_per_epoch.append(avg)
        print(f"  → epoch {epoch} done in {time.time()-t_epoch:.1f}s  avg_loss={avg:.4f}\n",
              flush=True)

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = Path("/checkpoints") / f"idmvton_encoder_lora_shorts_{stamp}_e{epoch}.pt"
            state = {k: v.cpu() for k, v in unet.state_dict().items() if "lora_" in k}
            torch.save({
                "lora_state_dict": state,
                "lora_config":     {"r": lora_rank, "alpha": lora_alpha,
                                    "target_modules": ["to_q","to_k","to_v","to_out.0"]},
                "target":          "unet_encoder",
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
    train_encoder_lora()
