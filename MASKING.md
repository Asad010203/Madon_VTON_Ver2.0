# MASKING.md — how we produce agnostic masks for training + inference

**Audience:** future Claude sessions that need to understand, debug, or extend the mask pipeline. Written 2026-07-23 after diagnosing multiple LoRA-failure incidents whose root cause was mask inconsistency.

---

## TL;DR

Every training triplet and every inference call generates a **binary agnostic mask** — a black-and-white image that tells the diffusion model "inpaint this region." Getting the mask right is more important than any hyperparameter. Get it wrong and no amount of training saves you.

The mask pipeline has **three critical properties** that took real debugging to establish:

1. **Two-stage generation**: parsing + openpose → agnostic mask via `leffavton.preprocessing.image_ops.get_agnostic_mask_viton_hd`, then a **hip-line cut** to trim over-dilation.
2. **Applied identically in training AND inference** — this is the invariant. If they differ, the LoRA learns the wrong distribution and produces garbage at test time.
3. **Idempotent double-application** — the hip cut is applied both in `step_06_precompute_masks.py` (data prep) AND in the patched `leffavton/preprocessing/image_ops.py` (inference). Cutting an already-zeroed pixel is a no-op, so it's safe.

If a mask ever looks wrong: the fix is at the mask generation code, not at training. Regenerate masks, don't retrain first.

---

## Why the mask matters

The Leffa VITON-HD generative UNet input is 12 channels wide:

```
model_input = concat([noisy_latent, mask_latent, masked_latent, densepose_latent], dim=1)
                       4               1            4               3           = 12
```

The `mask` tells the model **which pixels to synthesize**. Everything OUTSIDE the mask is preserved from the source person image via `masked_latent = src * (mask < 0.5)`. So:

- **Mask too small** → old garment leaks into the output around the edges (visible ring of old shirt around new shirt)
- **Mask too large** → skin gets painted over (arms disappear, neck vanishes into fabric)
- **Mask misaligned with body** → floating garments, torn geometry

The base VITON-HD dataset used specific mask conventions. Any drift from those conventions produces mode-collapse or blob outputs in downstream LoRA training/inference.

---

## The mask pipeline (step by step)

Reference: [`retrain/step_06_precompute_masks.py`](retrain/step_06_precompute_masks.py) lines ~230-260.

```
person.jpg (any size)
    │
    ▼  resize_and_center to (W=768, H=1024)
person_resized
    │
    ▼  downscale to (PARSE_W=384, PARSE_H=512) for parsing
small
    │
    ├──► parsing_atr.onnx + parsing_lip.onnx → parse (body-part segmentation)
    │
    └──► openpose body_pose_model → keypoints (body-25 skeleton)
                                            │
    ┌───────────────────────────────────────┘
    ▼
get_agnostic_mask_viton_hd(parse, keypoints, garment_type)
    │  (leffavton library function, returns PIL "L" mask at 384×512)
    │  - Uses body parts from `parse` to mask torso for upper, legs for lower
    │  - Applies `_dilate` to make edges soft-ish
    │  - The `_dilate` step is what OVER-EXTENDS the mask (see hip-cut fix below)
    │
    ▼
mask_small = np.array(mask.convert("L"))    # shape (512, 384), uint8, {0, 255}
    │
    ▼  hip_y = _hip_y_from_keypoints(keypoints, PARSE_H=512)
    │  Reads openpose body-25 idx 8 (mid-hip), 9 (right-hip), 12 (left-hip)
    │  Returns the y-coord in 512-tall space of the first non-zero hip keypoint.
    │  None if no hip keypoint detected (rare but possible).
    │
    ▼  _region_cut(mask_small, hip_y, gtype, PARSE_H=512)
    │  For upper_body: zero everything BELOW (hip_y + margin) where margin = 3% of PARSE_H
    │  For lower_body: zero everything ABOVE (hip_y - margin)
    │  If hip_y is None: leave mask unchanged (fallback, mask may over-dilate)
    │
    ▼
mask_final = Image.fromarray(mask_small).resize((W=768, H=1024), Image.NEAREST).convert("L")
    │
    ▼
mask/<slug>.png    (single-channel binary PNG on Modal volume + local disk)
```

Densepose runs in parallel on the full 768×1024 image via `detectron2` densepose predictor. Output: 3-channel RGB PNG showing body-part IUV visualization. Saved next to the mask as `densepose/<slug>.png`.

---

## The hip-cut fix — WHY

The library's `get_agnostic_mask_viton_hd` applies `_dilate` with a kernel that overshoots by ~50 px. Concrete failure modes:

- **upper_body**: mask extends below the belt line → tries to paint shirt over hip area → when swapping to a shirt that has different waist positioning, the old jeans waistband bleeds through
- **lower_body**: mask extends above the waist → tries to paint pants over torso → when swapping pants, existing shirt gets partially painted over

The **hip keypoint from openpose is stable** (99%+ of front-facing person shots have it detected), so we use it as an anatomical reference to hard-cut the mask at the true waistline ±3% margin.

### The margin

`margin = int(H * 0.03)` (~15 px at 512 tall, ~30 px at 1024 tall). This is the **natural waistband overlap zone** — shirts often tuck slightly below the waist, pants often ride slightly above. Cutting exactly at hip_y would leave visible gaps at the waistline seam. 3% preserves the seam continuity.

### Coordinate space matters

- Openpose returns keypoints in **512-tall reference space** by convention
- Parsing runs at 384×512
- Final images are 768×1024

So hip_y from openpose = pixel coordinate in 512-tall space. Do the cut at 512-tall (fast, matches parsing space), then resize to 1024 with NEAREST. The library's post-patch does the cut at full resolution using `hip_y * height / 512`. Both give identical results because NEAREST resize preserves binary edges exactly.

### Code (from step_06)

```python
def _region_cut(mask_arr: np.ndarray, hip_y: int, category: str, H: int) -> np.ndarray:
    if hip_y is None:
        return mask_arr
    margin = int(H * 0.03)
    if category == "upper_body":
        cut = min(H, hip_y + margin)
        mask_arr[cut:, :] = 0        # zero rows below the hip line
    elif category == "lower_body":
        cut = max(0, hip_y - margin)
        mask_arr[:cut, :] = 0        # zero rows above the hip line
    return mask_arr
```

Same logic, mirrored, exists inside the patched library at [`leffavton/preprocessing/image_ops.py:328-344`](https://github.com/franciszzj/Leffa/blob/main/leffavton/preprocessing/image_ops.py) (our fork).

---

## The critical training/inference alignment invariant

**This is the single most important thing in this document.**

Training data (`step_06_precompute_masks.py`) generates masks and writes them to disk. Training then reads those masks from disk each epoch. So training uses "step_06 masks."

Inference (via `VirtualTryOnInferencer.generate`) generates masks ONLINE at request time via the leffavton library's `get_agnostic_mask_viton_hd`. So inference uses "library masks."

**If step_06 does hip cut and the library doesn't → training and inference use different mask distributions → the LoRA overfits to step_06-style masks and misfires on library-style masks at inference.**

This was one of our real bugs. Fix: patched the library to ALSO apply the hip cut inside `get_agnostic_mask_viton_hd`. Now both paths produce identical masks for any given input.

**How to verify the invariant holds** (whenever mask code changes):

1. Pick a person image
2. Run it through step_06 mask pipeline → save mask_a.png
3. Run it through library `get_agnostic_mask_viton_hd` (via inference path) → save mask_b.png
4. Compare: `(mask_a == mask_b).all()` should be True (or differ by ≤1 pixel due to resize rounding)

If they differ meaningfully, DO NOT train more — fix the mask code first.

---

## Failure modes (real ones we hit)

### 1. `hip_y is None` — missing openpose keypoint
- **Cause**: extreme pose, occluded lower body, side view, or openpose confidence too low
- **Effect**: `_region_cut` returns the mask unchanged → mask over-dilates → training on this triplet teaches wrong signal
- **Frequency**: rare (~1-2% of front-facing shots) but present
- **Mitigation**: none currently — we accept the bad mask. If it becomes a problem, filter these triplets out of the manifest in step_03.

### 2. densepose fails silently on unusual poses
- **Cause**: detectron2 densepose has confidence thresholds that reject some poses
- **Effect**: densepose output is mostly black/empty → generative UNet gets garbage body geometry channel
- **Detection**: manually inspect `mask_previews.png` — the densepose row should look like a body silhouette. If it's mostly black for many persons, drop those.

### 3. Library patch not applied on the container
- **Cause**: `add_local_dir` in Modal image caches an OLD version of the leffavton library without the hip-cut patch
- **Effect**: training uses old masks, inference uses old masks — actually consistent but with the over-dilation problem — LoRA quality degrades but doesn't blob
- **Detection**: `modal image ls` to check if the image ID matches the current library HEAD, OR: eyeball fresh inference outputs against `mask_previews.png` — if they systematically over-cover, patch didn't apply
- **Fix**: rebuild the Modal image (`.add_local_dir(..., copy=True)` forces re-copy on image build)

### 4. Mask saved with wrong dtype/channels
- **Cause**: saving a boolean array or a 3-channel array as the mask
- **Effect**: `_prepare_mask` in `leffavton/transform.py` converts to L, but bad dtype can produce values other than {0, 1} after thresholding
- **Detection**: `PIL.Image.open("mask.png").mode` should be "L" (grayscale)
- **Fix**: ensure `.convert("L")` before save

### 5. Wrong resize interpolation (bilinear instead of nearest)
- **Cause**: someone edits step_06 or the transform to use `Image.BILINEAR` because "it looks smoother"
- **Effect**: mask edges become soft (values 0-255 in a gradient) → `mask < 0.5` boundary shifts unpredictably → the "inpaint region" changes shape at inference
- **Fix**: masks MUST use `Image.NEAREST` resize everywhere. Bilinear is for RGB images only.

---

## File formats on disk

Location on Modal volume: `/weights/brand_dataset_v2/`

```
mask/<slug>.png         # single-channel PIL "L" mode, uint8, values ∈ {0, 255}, size 768×1024
densepose/<slug>.png    # RGB PIL "RGB" mode, uint8, 3 channels IUV visualization, size 768×1024
```

Slug generation (`step_06_precompute_masks.py::_slug`): lowercase the person filename, replace `/` with `_`, strip trailing `.jpg`. Example: `POLO/PE2667-00S-CWT/model_3.jpg` → `polo_pe2667_00s_cwt_model_3`.

Slugs are deduped **per unique person path** — if the same person shot appears in multiple triplets (via cross_outfit pairing), only one mask file is generated. The manifest entries all point at the same mask/densepose paths.

---

## Human validation (Gate 2)

`step_06_precompute_masks.py` also renders `artifacts/mask_previews.png`:
- 24 random triplets (mix of upper/lower)
- Each row: person | mask overlay on person | densepose | garment | target
- Visual sanity check for the human before training

**Reject criteria:**
- Mask covers too much (paints onto skin/hair/face) → reject
- Mask covers too little (leaves visible ring of old garment) → reject
- Mask completely missing a limb (arm cut off, chest not covered) → reject
- Densepose is mostly black or wildly wrong → reject the triplet

If >10% of previewed triplets fail, fix the mask code before proceeding to training. If <10% fail, drop those specific SKUs from the manifest and re-render.

---

## Where the code lives

| Concern | File | Line |
|---|---|---|
| Mask generation for training data | [`retrain/step_06_precompute_masks.py`](retrain/step_06_precompute_masks.py) | 200-260 |
| Hip-cut logic (training side) | [`retrain/step_06_precompute_masks.py`](retrain/step_06_precompute_masks.py) | 207-228 |
| Mask preview / Gate 2 rendering | [`retrain/step_06_precompute_masks.py`](retrain/step_06_precompute_masks.py) | end of file |
| Library agnostic mask + hip-cut patch | `leffavton/preprocessing/image_ops.py` (d:/madon finalization/) | 315-346 |
| Mask consumed during training | `leffavton/training/train.py` | 216 (`masked_image = src * (mask < 0.5)`) |
| Mask consumed during inference | `leffavton/pipeline.py` | 104 (`masked_image = src_image * (mask < 0.5)`) |
| Mask preprocessing (PIL → tensor) | `leffavton/transform.py` | 119-142 (`_prepare_mask`) |

---

## Extending to new garment types

To add a new type (e.g., `dress`, `outerwear`):

1. Add the type to `GarmentType` enum in `leffavton/constants.py`
2. In `get_agnostic_mask_viton_hd`, add the body-part masking rules for the new type (which parse labels to include/exclude)
3. Decide if hip-cut applies. For `dress`: probably not — the mask spans hip. For `outerwear`: yes, cut at hip like upper_body.
4. Update `_region_cut` in step_06 with the new logic
5. Regenerate all masks (`modal run retrain/step_06_precompute_masks.py`)
6. Retrain LoRA (`modal_train_v2.py --garment-type <new_type>`)

Do NOT skip step 5 — reusing old masks with new logic guarantees the training/inference invariant will break.

---

## Debug recipes

**Q: Is the LoRA producing garbage. Is it a mask problem?**

1. Look at `mask_previews.png` — do masks look sane?
2. Regenerate ONE inference-time mask (via the library) and diff against the step_06 mask for the same person. Same? → mask alignment is fine, look elsewhere. Different? → library patch missing on the inference container.
3. If both look sane and match → mask isn't the issue; check pipeline.unet snapshot bug (see `leffavton-pipeline-unet-snapshot-gotcha.md` in claude memory).

**Q: Can I visualize just the masked region on a person?**

Use [`retrain/_view_masks.py`](retrain/_view_masks.py) — loads a batch of manifest entries and overlays mask on person for eyeballing.

**Q: New Modal image after library patch — how do I force it?**

```powershell
# Bump the leffavton library files in D:\madon finalization\leffavton\
# Then next `modal run` or `modal deploy` will detect changed files and rebuild.
# If it doesn't rebuild for some reason:
modal image list                    # find the stale image
modal image rm <image-id>          # force rebuild on next use
```

---

## History of mask bugs we've hit (chronological)

1. **Over-dilation from library `_dilate`** — mask bled below waist for upper, above waist for lower. Fixed by adding hip-cut in step_06.
2. **Training/inference mismatch** — training used step_06 masks with cut, inference used unpatched library without cut. LoRA outputs collapsed to blob figures. Fixed by patching library to also apply hip-cut.
3. **`hip_y is None` on some shots** — silently skipped cut → over-dilated mask for those triplets. Currently ignored (rare). If it grows: filter out via manifest.
4. **Container caching of old library** — patched locally but Modal container still had old code. Fixed by using `add_local_dir(..., copy=True)` which invalidates cache when source files change.

All four bugs share a theme: **anything that breaks the "training and inference produce identical masks for identical input" invariant will destroy LoRA quality.** Guard that invariant like your project depends on it — it does.
