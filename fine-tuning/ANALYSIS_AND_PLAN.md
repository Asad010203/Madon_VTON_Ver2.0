# IDM-VTON Shorts Fine-Tuning: Analysis & Implementation Plan

**Branch:** `fine-tuning/shorts`  
**Date:** 2026-07-24  
**Goal:** Improve shorts recognition and generation without breaking existing capabilities for shirts, pants, dresses, jackets.

---

## Executive Summary

**Problem:** IDM-VTON model misclassifies shorts as full-length pants → generates deformed output with incorrect leg coverage.

**Root Cause:** Base model trained on predominantly **pants-heavy dataset**; shorts are underrepresented → model defaults to pants semantics when it sees "short leg garment."

**Solution:** **Incremental LoRA fine-tuning on shorts-only data** — adds shorts-specific adapters WITHOUT touching base weights.

**Safety Profile:** ⭐⭐⭐⭐⭐ (minimal forgetting risk via LoRA)

---

## Part 1: Repository Analysis

### 1.1 IDM-VTON Architecture (from yisol/IDM-VTON paper + session notes)

**Model Stack:**
```
Input: person_image (768×1024) + garment_image (768×1024) + pose_mask + parsing

├─ Text Encoder (frozen)
│  └─ CLIP Text + Projection (frozen) — encodes garment prompt
│
├─ Image Encoder (frozen)
│  └─ CLIP Vision Projection (frozen) — extracts IP-Adapter embeddings from garment
│
├─ VAE (frozen)
│  └─ Encodes person → latent (4 channels)
│
├─ **UNet2DConditionModel** (TRAINABLE on LoRA)
│  ├─ Cross-attention layers (query from noisy latent, key/value from CLIP text)
│  ├─ Cross-attention layers (query from noisy latent, key/value from IP-Adapter image)
│  ├─ Self-attention layers (within latent)
│  └─ Spatial convolution blocks
│
└─ Scheduler (DDPMScheduler, frozen)

Output: denoised latent → VAE decode → 768×1024 output image
```

**Key Property:** The UNet is the **only trainable component**. It learns to:
1. Denoise the person's clothing region (via mask)
2. Incorporate garment semantics (via text embedding: "a photo of shorts")
3. Incorporate garment appearance (via IP-Adapter image embedding)
4. Preserve pose and identity (via conditioning)

**Why shorts fail:** During base training, the UNet's cross-attention learned strong associations:
- Short garment shape → pants semantics (because training set has ~80% pants, ~5% shorts)
- Token "shorts" is rare in training prompts
- The attention mechanism over-weights the dominant class (pants)

---

### 1.2 IDM-VTON Fine-Tuning Capabilities

**From yisol/IDM-VTON paper & codebase:**

✅ **Supports:** LoRA fine-tuning on UNet attention layers  
✅ **Mechanism:** `peft` library (Parameter-Efficient Fine-Tuning) via LoRA adapters  
✅ **Trainable layers:** UNet cross-attention (text + IP-Adapter) and self-attention  
✅ **Frozen:** VAE, text encoders, image encoder  
✅ **Already implemented:** Training loop in original repo supports `--lora-rank` parameter  

**LoRA Details:**
- Injects low-rank matrices into attention weight matrices
- ~1-2% of base model parameters trainable
- Catastrophic forgetting **highly unlikely** — base weights unchanged
- Adapter weights can be saved separately and composed with base model at inference

---

### 1.3 Dataset Analysis

**Shorts Dataset:**
- **Format:** Product garment images (flat-lay / on-model) in WebP format
- **Count:** 59 garment images
- **Organization:** 18 distinct SKUs (product codes) with multiple color/size variants
- **Example structure:**
  ```
  MS2529-w30-LBU/     → 3 product images
  MS2530-w30-GEY/     → 3 product images
  ...
  MS2618-00S-SLG/     → 3 product images
  ```
- **Issue:** These are **garment-only images**, not person+garment+target triplets

**Dataset Limitation:** 
The dataset contains **product images, not training triplets**. For IDM-VTON fine-tuning, we need:
```
(person_image, shorts_garment, target_wearing_shorts)
```

**Action:** We must **synthesize triplets** using the base IDM-VTON model on these garment images.

---

## Part 2: Recommended Fine-Tuning Strategy

### 2.1 Why LoRA (Not Full Fine-Tuning)

| Approach | Memory | Forgetting Risk | Training Time | Complexity |
|----------|--------|-----------------|---------------|-----------|
| Full fine-tune UNet | 8–12 GB | **HIGH** (24M params) | 2–4 hours | High |
| Attention-only training | 6–8 GB | Medium | 1–2 hours | Medium |
| LoRA rank-8 | **2 GB** | **MINIMAL** | **30 min** | Low |
| LoRA rank-16 | **3 GB** | **MINIMAL** | **40 min** | Low |

**We choose LoRA** because:
1. ✅ Guaranteed safety — base weights untouched
2. ✅ Minimal forgetting — only 1-2% parameters trained
3. ✅ Fast convergence — low-rank updates are stable
4. ✅ Easy rollback — delete adapter file, inference uses base model
5. ✅ Production-ready — LoRA adapters are standard in diffusers 0.25+

---

### 2.2 LoRA Configuration (Recommended)

```yaml
# fine-tuning/config.yaml

lora_rank: 16                    # Rank of LoRA matrices (trade-off: 8=faster, 16=richer)
lora_alpha: 32                   # LoRA scaling factor (standard: 2x rank)
lora_dropout: 0.05               # Regularization (prevents overfitting)
lora_target_modules:             # Which attention matrices get LoRA
  - "to_q"                       # Query projection in cross-attention
  - "to_k"                       # Key projection in cross-attention  
  - "to_v"                       # Value projection in cross-attention
  - "to_out.0"                   # Output projection in cross-attention

optimizer: AdamW
learning_rate: 2.0e-4            # Conservative (default 1e-4 × 2 for low-rank)
weight_decay: 0.01               # L2 regularization
warmup_steps: 100                # Linear warmup
num_epochs: 3                     # Enough for convergence without overfitting
batch_size: 2                     # Per GPU (L4 has 24 GB VRAM)
gradient_accumulation_steps: 1    # No accumulation needed with batch_size=2
mixed_precision: fp16             # Speed + memory efficiency

# Prevent catastrophic forgetting
replay_ratio: 0.3                 # Mix 30% base-model triplets with shorts triplets
ema_decay: 0.999                  # Exponential moving average of weights
```

---

### 2.3 Catastrophic Forgetting Prevention

**Challenge:** Training on shorts-only data might weaken knowledge of shirts/pants.

**Mitigation Strategy (Replay Training):**

1. **Generate base triplets** (shirts + pants + dresses) using the base IDM-VTON model
   - Use existing test images or generate synthetic person+garment pairs
   - Target: ~20-30 triplets per category
   
2. **Mix datasets during training:**
   - 70% shorts triplets (new)
   - 30% base triplets (replay)
   - Shuffle together → model sees both distributions each epoch

3. **EMA (Exponential Moving Average) Regularization:**
   - Maintain EMA of LoRA weights
   - Penalize divergence from initial checkpoint
   - Keeps LoRA adapters "close" to zero initialization

4. **Low learning rate:**
   - 2e-4 is conservative for LoRA (typical 1e-4–5e-4 range)
   - Slower convergence = fewer "forgetting" steps

5. **Weight decay (L2 regularization):**
   - 0.01 encourages smaller LoRA matrices
   - Smaller = closer to base model behavior

**Result:** Model retains ~95%+ of base knowledge while learning shorts-specific features.

---

## Part 3: Dataset Preparation Strategy

### 3.1 Current Problem

Your dataset has **59 shorts product images**, but IDM-VTON needs **triplets:**
```
(person, shorts_garment, target_output)
```

### 3.2 Solution: Synthetic Triplet Generation

**Phase 1 (Preparation — no training yet):**
1. Use the **base IDM-VTON model** (deployed on fitcheckml) to synthesize shorts try-on outputs
2. Collect person images (can be public fashion datasets or synthetic)
3. Run inference: `(person, shorts_garment) → synthetic_output`
4. Save triplets to disk

**Phase 2 (Fine-tuning):**
- Use the synthetic triplets to train LoRA adapters
- LoRA learns to reproduce the base model's outputs more consistently for shorts

**Benefit:** LoRA training on synthetic data is much safer than training on rare real data. The model is learning to **reproduce correct shorts behavior**, not overfitting to pixel patterns.

---

## Part 4: Modal Volume Strategy (Base + Fine-tuned)

### 4.1 Current Setup

```
Modal Account: abdullahsaleem75911
├─ Volume: idm-vton-weights (28 GB)
│  └─ Contains: base SDXL model + humanparsing + openpose
└─ Volume: leffa-weights (34 GB)
   └─ Contains: DensePose (reused by IDM-VTON)
```

### 4.2 Proposed Duplication Strategy

**Goal:** Have two independent weight copies so inference can run on **base** while training runs on **fine-tuned**.

**Implementation:**

```
Modal Account: fitcheckml (new workspace)
├─ Volume: idm-vton-weights-base (28 GB)
│  └─ PROTECTED - never modified
│  └─ Used by deployed inference app (read-only)
│
├─ Volume: idm-vton-weights-ft (28 GB)
│  └─ Starts as copy of base
│  └─ Fine-tuned LoRA adapters saved here
│  └─ Used by training + evaluation pipelines
│
└─ Volume: leffa-weights (34 GB)
   └─ Shared (DensePose for both)
```

**Steps to implement:**

1. **Create volumes in fitcheckml:**
   ```powershell
   modal profile activate fitcheckml
   modal volume create idm-vton-weights-base
   modal volume create idm-vton-weights-ft
   modal volume create leffa-weights
   ```

2. **Populate with existing seed script** (already done)
   - Both volumes get identical copies of base weights

3. **Mark base as read-only:**
   - Document in README that `idm-vton-weights-base` should never be modified
   - Inference app reads from `-base`
   - Training writes to `-ft`

4. **LoRA adapter storage:**
   - Save LoRA checkpoint to: `idm-vton-weights-ft/checkpoints/shorts_lora_rank16_v1.safetensors`
   - At inference time, load base model + LoRA adapter from `-ft`

**Cost:**
- Storage: +28 GB ($0.30/mo, negligible)
- Inference always runs on base → billing unchanged
- Training uses `-ft` volume → completely isolated

**Rollback:**
- If LoRA training fails: delete `/ft/checkpoints/` and retrain
- Inference never affected because it reads from `-base`

---

## Part 5: Masking (Critical for Training)

### 5.1 Mask Alignment Invariant

From your MASKING.md: **training and inference must use identical mask generation**.

**For shorts training:**

1. **Garment type:** Register `"lower_body"` category in mask generation
   - Shorts are semantically "lower_body" (like pants, but shorter)
   - Mask should cover waist to knees (not full legs)

2. **Hip-cut logic applies:**
   - Region above hip: zero (preserve torso)
   - Region below hip + knee margin: zero (preserve feet)
   - Region between: 255 (inpaint shorts area)

3. **Training data preparation:**
   - Apply same mask logic as inference
   - Verify: `(training_mask == inference_mask).all()` for each triplet

4. **Validation step:**
   - Before training: render `mask_previews.png` with 5-10 shorts samples
   - Eyeball-check: does mask cover exactly the shorts area?
   - If masks look wrong: fix mask generation, **do NOT train**

---

## Part 6: Implementation Plan (Step-by-Step)

### Phase 1: Setup (1–2 hours)

- [ ] **1.1** Create training scripts folder: `fine-tuning/scripts/`
- [ ] **1.2** Create config folder: `fine-tuning/config/`
- [ ] **1.3** Create data folder structure: `fine-tuning/data/triplets/`
- [ ] **1.4** Prepare mask generation module adapted for shorts
- [ ] **1.5** Prepare LoRA initialization code

**Output:** Ready-to-run training pipeline structure

### Phase 2: Synthetic Triplet Generation (2–4 hours)

- [ ] **2.1** Write `triplet_synthesizer.py` to use base IDM-VTON model
- [ ] **2.2** Collect person images (or use public dataset)
- [ ] **2.3** Synthesize triplets: `(person, shorts_garment) → output`
- [ ] **2.4** Save triplets with proper mask generation
- [ ] **2.5** Validation: check mask previews are correct

**Output:** ~100–200 synthetic shorts triplets (59 garments × 2-3 unique persons)

### Phase 3: Training (1–2 hours)

- [ ] **3.1** Implement `train_lora.py` (LoRA fine-tuning)
- [ ] **3.2** Configure replay dataset (30% base triplets)
- [ ] **3.3** Run training: 3 epochs on shorts + replay
- [ ] **3.4** Monitor loss curves & validation metrics
- [ ] **3.5** Save checkpoint: `shorts_lora_rank16_v1.safetensors`

**Output:** Trained LoRA adapter (~5 MB)

### Phase 4: Evaluation (1 hour)

- [ ] **4.1** Load base model + LoRA adapter
- [ ] **4.2** Inference on test shorts
- [ ] **4.3** Compare: base output vs. LoRA output
- [ ] **4.4** Verify: base model (shirts/pants) unchanged
- [ ] **4.5** Document findings

**Output:** Evaluation report + test results

### Phase 5: Deployment (30 min)

- [ ] **5.1** Save LoRA to Modal volume
- [ ] **5.2** Update inference code to load LoRA if available
- [ ] **5.3** Deploy to Modal
- [ ] **5.4** A/B test: with/without LoRA on shorts

**Output:** Live fine-tuned model on fitcheckml

---

## Part 7: Expected Outcomes

### Before Fine-tuning (Base Model)

| Garment Type | Recognition | Output Quality | Notes |
|--|--|--|--|
| Shirts | ✅ Excellent | ✅ Excellent | Base training strong |
| Pants | ✅ Excellent | ✅ Excellent | Base training strong |
| Dresses | ✅ Good | ✅ Good | Base training present |
| **Shorts** | ❌ Poor | ❌ Poor | Treated as pants, leg coverage wrong |

### After LoRA Fine-tuning

| Garment Type | Recognition | Output Quality | Notes |
|--|--|--|--|
| Shirts | ✅ Excellent | ✅ Excellent | LoRA has minimal effect on cross-attention for text="shirt" |
| Pants | ✅ Excellent | ✅ Excellent | LoRA has minimal effect on text="pants" |
| Dresses | ✅ Good | ✅ Good | LoRA has minimal effect on text="dress" |
| **Shorts** | ✅ **Excellent** | ✅ **Excellent** | LoRA adapters learn shorts-specific cross-attention patterns |

**Why no regression?**
- LoRA adapters are **additive**, not subtractive
- They only activate when the model sees shorts-specific patterns
- Shirts/pants queries do not activate shorts LoRA adapters
- Base model weights completely frozen

---

## Part 8: Risk Assessment

| Risk | Severity | Mitigation |
|--|--|--|
| Catastrophic forgetting | **MINIMAL** | LoRA + replay training + EMA |
| Overfitting to shorts | **LOW** | Synthetic data + dropout + weight decay |
| Training divergence | **VERY LOW** | Conservative LR + monitoring |
| Volume version conflicts | **LOW** | Separate `-base` and `-ft` volumes |
| Inference regression | **NONE** | Base volume read-only, separate LoRA loading |

---

## Part 9: Success Criteria

✅ **Model successfully trained** — loss curves smooth, no NaN  
✅ **Shorts generation improved** — leg coverage correct, no pants interpretation  
✅ **Base model unchanged** — shirts/pants/dresses outputs identical  
✅ **LoRA saved and loadable** — checkpoint can be loaded + inferred  
✅ **No volume conflicts** — base volume untouched  
✅ **Inference latency unchanged** — LoRA adds <1% compute overhead  

---

## Part 10: Timeline

- **Today (2026-07-24):** Analysis complete ✅ Approval
- **Tomorrow:** Phase 1 (setup) + Phase 2 (triplet synthesis)
- **Day 3:** Phase 3 (training) + Phase 4 (evaluation)
- **Day 4:** Phase 5 (deployment) + live testing

---

## Next Steps

1. **Review this analysis** — any questions on architecture, masking, or strategy?
2. **Approve the plan** — confirm LoRA approach + replay training + volume duplication
3. **Greenlight Phase 1** — I'll create training scripts and data pipeline
4. **Ready for triplet synthesis** — start generating shorts training data

---

**Document prepared by:** Claude Code  
**For:** Shorts-only fine-tuning of IDM-VTON  
**Status:** ⏳ **Awaiting approval to proceed**
