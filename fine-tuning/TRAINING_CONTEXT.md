# IDM-VTON LoRA Fine-Tuning — Process Log

Complete record of the shorts LoRA fine-tuning effort on Modal.com. Kept for context so we don't re-litigate the same debugging next session.

## Goal

Fine-tune IDM-VTON with a shorts-specific LoRA that:
- **Preserves** all original base weights (no catastrophic forgetting)
- Trains only on user-provided shorts dataset
- Runs training + inference entirely on Modal.com (fitcheckml workspace)
- Produces a checkpoint loadable at inference time to bias output toward shorts

## Dataset

User-prepared, uploaded to Modal volume `idm-vton-datasets`:

```
/data/shorts_dataset/
  manifest.jsonl           # 17 training triplets (JSON per line)
  manifest_heldout.jsonl   # 2 validation triplets
  person/   (19 .jpg)      # subject photos
  garment/  (18 .jpg)      # flat garment photos
  target/   (19 .jpg)      # ground-truth person-wearing-garment
  mask/     (18 .png)      # binary agnostic masks (garment region)
  pose/     (18 .png)      # DensePose visualization maps
```

Manifest entry format:
```json
{"person": "person/...jpg", "garment": "garment/...jpg", "target": "target/...jpg",
 "mask": "mask/...png", "pose": "pose/...png",
 "garment_type": "lower_body", "_kind": "self_recon", ...}
```

One pose file is missing (`short_ms2529_w30_lbu_f4a58300.png`) — training script filters it out (16 usable triplets).

## Modal Volumes Used

| Volume | Mount | Contents |
|---|---|---|
| `idm-vton-weights` | `/idm-weights` | Base IDM-VTON SDXL checkpoint |
| `leffa-weights` | `/weights` | DensePose model files (reused) |
| `idm-vton-datasets` | `/data` | Shorts dataset |
| `idm-vton-checkpoints` | `/checkpoints` | Trained LoRA `.pt` files |
| `idm-vton-outputs` | `/outputs` | Generated try-on images |

## Training Attempts — What Failed and Why

Six failed iterations before the seventh worked. Each cost ~$0.30-0.50 on A100.

| # | Script | Failure | Root cause |
|---|---|---|---|
| 1 | `modal_train_simple.py` | (worked, but useless) | Trained a placeholder Conv2d, not a real LoRA. Checkpoint incompatible with IDM-VTON pipeline. |
| 2 | `modal_train_lora_real.py` | `dtype mismatch: Float vs Half` | Base UNet in fp16, LoRA weights forced to fp32 → linear ops between them fail. Fix: use bf16 throughout. |
| 3 | `modal_train_lora_real.py` | `ip_image_proj requires image_embeds` | IDM-VTON hacked UNet needs garment IP-adapter tokens as `added_cond_kwargs["image_embeds"]`. |
| 4 | `modal_train_lora_real.py` | `Expected 2048 but got 1024 for tensor number 1` | Passed raw image_encoder features (1024-dim) — needed to project through `unet.encoder_hid_proj` to 2048-dim first. |
| 5 | `modal_train_lora_real.py` | `expected input to have 13 channels, but got 4` | Hacked UNet is an inpaint UNet: expects `[noisy_latents(4), mask(1), masked_latents(4), pose_map(4)]` = 13 channels. |
| 6 | `modal_train_lora_real.py` | `'NoneType' object is not subscriptable` at `garment_features[curr_garment_feat_idx]` | Hacked UNet also needs `garment_features` — intermediate feature list from a separate reference UNet (`unet_encoder`) run on the garment image. Not optional. |
| 6b | `modal_train_lora_encoder.py` (pivot) | Output was 640-ch feature map, not 4-ch noise | Encoder UNet is a pure feature extractor — cannot be trained with standard diffusion loss. Dead end. |
| 7 | `modal_train_lora_official.py` | ✅ **SUCCESS** | Followed upstream `yisol/IDM-VTON/train_xl.py` forward pass exactly. |

## The Working Recipe

From upstream `train_xl.py` lines 615-712. All four inputs must be provided together:

```python
# 1. Build 13-channel inpaint input (order matters!)
latent_model_input = cat([noisy_latents(4), mask(1), masked_latents(4), pose_map(4)])

# 2. IP-adapter tokens via encoder_hid_proj (already baked into IDM-VTON checkpoint)
clip_hidden = image_encoder(garment_clip, output_hidden_states=True).hidden_states[-2]
ip_tokens = unet.encoder_hid_proj(clip_hidden)  # 1024 → 2048

# 3. Garment features from reference UNet
_, ref_features = unet_encoder(cloth_latents, timesteps, text_cloth, return_dict=False)

# 4. Main UNet forward with all conditioning
noise_pred = unet(
    latent_model_input, timesteps, encoder_hidden_states,
    added_cond_kwargs={"text_embeds": pooled, "time_ids": ids, "image_embeds": ip_tokens},
    garment_features=list(ref_features),
).sample

loss = MSE(noise_pred, noise)
```

**LoRA config:** rank 16, alpha 32, target `["to_q","to_k","to_v","to_out.0"]` on main UNet cross-attention.
**Base weights:** all frozen. Only LoRA (~23M params, 0.78% of total) is trainable.
**Training:** 30 epochs × 16 samples, A100 GPU, ~15 min, ~$1.

## Final Result

- Checkpoint: `/checkpoints/idmvton_lora_official_20260724_164142_e30.pt` (~90MB, 1128 LoRA tensors)
- Best loss: **0.0176**
- Base model on `idm-vton-weights` volume: **untouched**

## Inference

**Modal side:** `modal_infer_lora.py` — deployed app `idm-vton-lora-infer`. Function `run_idm_lora`:
- Loads full IDM-VTON pipeline (L4 GPU)
- If `lora_checkpoint` arg is provided: wraps UNet in PEFT, loads LoRA state, `merge_and_unload()` to bake weights into base
- Runs standard IDM-VTON inference

Also exposes `list_checkpoints()` → returns all `idmvton_lora_*.pt` on the volume.

**Local side:** `test_lora.py` — interactive CLI:
```powershell
python test_lora.py                # list checkpoints, pick, run
python test_lora.py --no-lora      # baseline (no LoRA), for A/B comparison
python test_lora.py --checkpoint <name>.pt
```

Flow: file-browser picks person image → file-browser picks garment → choose garment type → Modal L4 GPU → saves result + a composite (Person | Garment | Result side-by-side) → auto-opens the composite in default viewer.

## Behavior — Preservation vs. Specialization

The base UNet weights are frozen during training and stay 100% intact on the volume. The LoRA is a **separate, removable adapter**.

| Mode | What runs | Behavior |
|---|---|---|
| `test_lora.py` (checkpoint picked) | Base + LoRA merged | Base knowledge + shorts bias. Always-on right now. |
| `test_lora.py --no-lora` | Pure base | Identical to original IDM-VTON. |

Current behavior is **Option 3 (always-on LoRA)** — LoRA merged for every request regardless of garment type. Since LoRA is prompt-sensitive, non-shorts prompts barely activate it, but a small drift is possible. Cleaner alternative (**Option 1**) would be to auto-skip LoRA when garment_type ≠ lower_body — one-line change, not yet applied.

## Files in `fine-tuning/`

Kept:
- `modal_train_lora_official.py` — **the working training script**
- `modal_infer_lora.py` — deployed inference app
- `test_lora.py` — local interactive CLI
- `data set/artifacts_shorts/dataset/` — user's dataset (mirrored to Modal volume)
- `TRAINING_CONTEXT.md` — this file

Deprecated (kept for reference but not used):
- `modal_train_simple.py` — placeholder training, wrong approach
- `modal_train_lora_real.py` — attempts 2-6, hit architectural blockers
- `modal_train_lora_encoder.py` — encoder-UNet pivot, dead end
- `modal_test_shorts.py` — early testing on placeholder checkpoint
- `modal_inference_shorts.py` — early inference attempt, not connected to real pipeline

## Known Limitations

- **17 training samples is very small.** Even a correct LoRA barely shifts output visibly. IDM-VTON was originally trained on VITON-HD (~11k pairs). Expect subtle, not dramatic, shorts specialization. Need ≥200 samples for pronounced effect.
- **Always-on LoRA** may cause minor drift on shirts/dresses. Switch to auto-mode when ready.
- **Container cold start ~90s** (L4 image load). Warm inference ~30s per request.

## Cost Summary

- Failed training iterations 1-6: ~$3-5 total
- Successful training: ~$1
- Each inference call: ~$0.10 (L4, 30 steps)
- Modal image build (one-time): ~10 min including detectron2 compile

## Next Steps (Deferred)

1. Wire up auto-switch: LoRA only applied when garment_type == "lower_body"
2. Expand dataset to 200+ shorts triplets for a more visible effect
3. Try LoRA scale < 1.0 for softer blending
4. Consider training LoRA on unet_encoder too (garment feature extractor) for stronger garment specialization
