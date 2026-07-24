"""
LoRA training on IDM-VTON — follows the official train_xl.py forward pass.

Key recipe from yisol/IDM-VTON/train_xl.py (lines 615-712):
  latent_model_input = cat([noisy_latents(4), mask(1), masked_latents(4), pose_map(4)])
  down, reference_features = unet_encoder(cloth_latents, timesteps, text_cloth, return_dict=False)
  ip_tokens = unet.encoder_hid_proj(image_encoder(garment).hidden_states[-2])
  noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states,
                    added_cond_kwargs={..., "image_embeds": ip_tokens},
                    garment_features=list(reference_features)).sample
  loss = MSE(noise_pred, noise)

Difference: we freeze everything and only train LoRA on the main UNet's
cross-attention layers. Base weights stay untouched.

Usage:
    modal run modal_train_lora_official.py::train
"""

from __future__ import annotations

from pathlib import Path
import modal


IDM_VOLUME    = "idm-vton-weights"
IDM_MOUNT     = "/idm-weights"
MODEL_PATH    = "/idm-weights/IDM-VTON"

app                 = modal.App("idm-vton-lora-official")
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
def train(
    epochs: int = 30,
    batch_size: int = 1,
    lr: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    save_every: int = 10,
):
    """Train LoRA on IDM-VTON following the official forward-pass recipe."""
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

    from transformers import (AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection,
                              CLIPVisionModelWithProjection, CLIPImageProcessor)
    from diffusers import AutoencoderKL, DDPMScheduler
    from src.unet_hacked_tryon   import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm
    from peft import LoraConfig, get_peft_model

    print("=" * 80)
    print("IDM-VTON LoRA Training (Official Forward Pass)")
    print("=" * 80)
    print(f"Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"LoRA rank: {lora_rank}  |  alpha: {lora_alpha}")

    device = torch.device("cuda")
    dtype  = torch.bfloat16

    # ── Load base pipeline (all frozen) ──────────────────────────────────────
    print("\n[1/4] Loading base components...")
    t0 = time.time()

    unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet", torch_dtype=dtype)
    unet_encoder = UNetGarm.from_pretrained(MODEL_PATH, subfolder="unet_encoder", torch_dtype=dtype)
    vae  = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.float32)
    text_encoder_one = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder="text_encoder_2", torch_dtype=dtype)
    image_encoder    = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder="image_encoder", torch_dtype=dtype)
    tokenizer_one    = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer",   use_fast=False)
    tokenizer_two    = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2", use_fast=False)
    scheduler        = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    for m in (vae, text_encoder_one, text_encoder_two, image_encoder, unet_encoder):
        m.requires_grad_(False); m.to(device)
    unet.requires_grad_(False)

    clip_processor = CLIPImageProcessor()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Inject LoRA on main UNet cross-attention ─────────────────────────────
    print("\n[2/4] Injecting LoRA on cross-attention...")
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

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n[3/4] Loading dataset...")
    manifest_path = Path("/data/shorts_dataset/manifest.jsonl")
    all_triplets = [json.loads(l) for l in manifest_path.open()]
    root = Path("/data/shorts_dataset")
    triplets = []
    skipped = 0
    for t in all_triplets:
        if all((root / t[k]).exists() for k in ("person", "garment", "target", "pose", "mask")):
            triplets.append(t)
        else:
            skipped += 1
    print(f"  {len(triplets)} usable triplets ({skipped} skipped for missing files)")

    IMG_H, IMG_W = 1024, 768
    tensor_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    class ShortsDataset(Dataset):
        def __init__(self, triplets, root):
            self.triplets, self.root = triplets, Path(root)
        def __len__(self): return len(self.triplets)
        def __getitem__(self, idx):
            t = self.triplets[idx]
            person  = Image.open(self.root / t["person"]).convert("RGB").resize((IMG_W, IMG_H))
            garment = Image.open(self.root / t["garment"]).convert("RGB").resize((IMG_W, IMG_H))
            target  = Image.open(self.root / t["target"]).convert("RGB").resize((IMG_W, IMG_H))
            pose    = Image.open(self.root / t["pose"]).convert("RGB").resize((IMG_W, IMG_H))
            mask    = Image.open(self.root / t["mask"]).convert("L").resize((IMG_W, IMG_H))
            garment_clip = clip_processor(images=garment, return_tensors="pt").pixel_values[0]
            return {
                "person":       tensor_tfm(person),
                "garment":      tensor_tfm(garment),
                "target":       tensor_tfm(target),
                "pose":         tensor_tfm(pose),
                "mask":         transforms.ToTensor()(mask),
                "garment_clip": garment_clip,
                "prompt":       f"model is wearing shorts",
                "prompt_cloth": f"a photo of shorts",
            }

    loader = DataLoader(ShortsDataset(triplets, "/data/shorts_dataset"),
                        batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW([p for p in unet.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    def enc_prompt(prompt: str):
        t1 = tokenizer_one(prompt, padding="max_length", max_length=tokenizer_one.model_max_length,
                           truncation=True, return_tensors="pt").input_ids.to(device)
        t2 = tokenizer_two(prompt, padding="max_length", max_length=tokenizer_two.model_max_length,
                           truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            e1 = text_encoder_one(t1, output_hidden_states=True).hidden_states[-2]
            o2 = text_encoder_two(t2, output_hidden_states=True)
            e2 = o2.hidden_states[-2]
            pooled = o2[0]
        return torch.cat([e1, e2], dim=-1), pooled

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
            person  = batch["person"].to(device, dtype)
            garment = batch["garment"].to(device, dtype)
            target  = batch["target"].to(device, dtype)
            pose    = batch["pose"].to(device, dtype)
            mask    = batch["mask"].to(device, dtype)
            garment_clip = batch["garment_clip"].to(device, dtype)

            with torch.no_grad():
                # VAE encodes (fp32) → cast to bf16
                target_lat  = (vae.encode(target.float()).latent_dist.sample() * vae.config.scaling_factor).to(dtype)
                masked_lat  = (vae.encode((person * (1 - mask.expand(-1, 3, -1, -1))).float()).latent_dist.sample() * vae.config.scaling_factor).to(dtype)
                pose_lat    = (vae.encode(pose.float()).latent_dist.sample() * vae.config.scaling_factor).to(dtype)
                cloth_lat   = (vae.encode(garment.float()).latent_dist.sample() * vae.config.scaling_factor).to(dtype)
                mask_lat    = F.interpolate(mask.to(dtype), size=target_lat.shape[-2:], mode="nearest")

                # IP-adapter tokens: hidden_states[-2] → encoder_hid_proj → cross-attn dim
                clip_hidden = image_encoder(garment_clip, output_hidden_states=True).hidden_states[-2]
                # get_base_model needed because unet is wrapped in PeftModel
                ip_tokens = unet.get_base_model().encoder_hid_proj(clip_hidden)

            # Sample noise + timestep
            noise     = torch.randn_like(target_lat)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (target_lat.shape[0],), device=device).long()
            noisy = scheduler.add_noise(target_lat, noise, timesteps)

            # Official channel order: [noisy(4), mask(1), masked(4), pose(4)]
            latent_model_input = torch.cat([noisy, mask_lat, masked_lat, pose_lat], dim=1)

            # Text encodings
            enc_hidden, pooled = enc_prompt(batch["prompt"][0])
            text_cloth, _      = enc_prompt(batch["prompt_cloth"][0])

            add_time_ids = torch.tensor([[IMG_H, IMG_W, 0, 0, IMG_H, IMG_W]],
                                        device=device, dtype=dtype).repeat(target_lat.shape[0], 1)
            added_cond = {"text_embeds": pooled.to(dtype),
                          "time_ids":    add_time_ids,
                          "image_embeds": ip_tokens.to(dtype)}

            # Garment features from encoder UNet
            with torch.no_grad():
                _, ref_features = unet_encoder(cloth_lat, timesteps, text_cloth.to(dtype),
                                               return_dict=False)
            ref_features = list(ref_features)

            # Main UNet forward
            noise_pred = unet(latent_model_input, timesteps,
                              encoder_hidden_states=enc_hidden.to(dtype),
                              added_cond_kwargs=added_cond,
                              garment_features=ref_features).sample

            loss = F.mse_loss(noise_pred.float(), noise.float())

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
            ckpt_path = Path("/checkpoints") / f"idmvton_lora_official_{stamp}_e{epoch}.pt"
            state = {k: v.cpu() for k, v in unet.state_dict().items() if "lora_" in k}
            torch.save({
                "lora_state_dict": state,
                "lora_config":     {"r": lora_rank, "alpha": lora_alpha,
                                    "target_modules": ["to_q","to_k","to_v","to_out.0"]},
                "target":          "unet_main",
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
    train()
