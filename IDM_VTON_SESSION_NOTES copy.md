# IDM-VTON on Modal — Session Notes (2026-07-22)

Full context of the IDM-VTON setup session so next time is fast.

---

## Status at end of session

- **Deployed and running** as Modal app `idm-vton` on account `abdullahsaleem75911`
- Testing paused for the day; containers manually stopped (billing off)
- All 20+ setup errors resolved; app should be usable now (final result not visually verified in this session — build+debug consumed the whole session)

## The two files you interact with

| File | Purpose |
|---|---|
| `modal_idm_inference.py` | Modal-side app. Defines the container image, the `run_idm` function on L4 GPU, and how weights are wired. Deploy with `modal deploy modal_idm_inference.py` (from `D:\madon finalization\`) |
| `test_idm.py` | Laptop-side launcher. Tkinter file pickers → sends bytes to Modal → saves result to `outputs/vton_idm_TIMESTAMP.png` |

## To resume next session

```powershell
cd "D:\madon finalization"
.\.venv\Scripts\python.exe test_idm.py
```

That's it. If `idm-vton` app is still deployed on Modal (check https://modal.com/apps/abdullahsaleem75911/main/deployed/idm-vton), the first call cold-starts (~90 s) and subsequent ones are ~40 s.

If deployed app is gone or you edited `modal_idm_inference.py`, redeploy first:
```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe -m modal deploy modal_idm_inference.py
```

**Encoding note**: Always set `PYTHONIOENCODING=utf-8` in PowerShell before running Modal commands. Modal emits Unicode progress bars (`━`) that crash PowerShell's default cp1252 encoder if you pipe or `Tee-Object` the output.

## Modal resources

| Resource | Purpose | Cost |
|---|---|---|
| App `idm-vton` | Deployed IDM-VTON inference | Only while running (~$0.10/inference) |
| App `leffa-vton` | Older Leffa deployment (still there) | Only while running |
| Volume `idm-vton-weights` | 28 GB — IDM-VTON SDXL weights, humanparsing, openpose | ~$0.60/mo storage |
| Volume `leffa-weights` | 34 GB — Leffa base weights + DensePose (**reused** by IDM-VTON!) | ~$0.70/mo storage |
| Secret `hf-token` | HF token for weight downloads | free |

## Winning configuration (final, working)

**Container image (Modal side)**:
- `python==3.10`, base `debian_slim`
- `torch==2.6.0+cu124`, `torchvision==0.21.0`
- `diffusers==0.25.0` + `huggingface_hub==0.25.0` + `transformers==4.36.2` + `accelerate==0.26.1`
- `numpy>=1.26,<3` (flexible — let pip pick a version compatible with skimage's wheel)
- `onnxruntime>=1.19` (**NOT** 1.16.2 — that was compiled for numpy 1.x → SIGSEGV on numpy 2)
- `scikit-image>=0.22`, `scipy>=1.10`, `pillow>=9.4,<11`
- `detectron2` + DensePose (installed from github, needs `gpu="L4"` at build time)
- Repo cloned: `git clone --depth 1 https://github.com/yisol/IDM-VTON /app/IDM-VTON`
- `PYTHONPATH=/app/IDM-VTON:/app/IDM-VTON/gradio_demo`

**Function config**:
- GPU: `L4` (24 GB VRAM — fits SDXL comfortably)
- Timeout: 900 s
- Volumes: **BOTH** `idm-vton-weights` at `/idm-weights` AND `leffa-weights` at `/weights` (for DensePose files)
- `retries=0` (never auto-retry; failure means fix the code)

**Runtime setup inside `run_idm`**:
- `os.chdir("/app/IDM-VTON")` (IDM code uses relative paths like `./ckpt/`)
- Symlinks:
  - `/app/IDM-VTON/ckpt/humanparsing` → `/idm-weights/IDM-VTON/humanparsing`
  - `/app/IDM-VTON/ckpt/openpose` → `/idm-weights/IDM-VTON/openpose`
  - `/app/IDM-VTON/ckpt/densepose` → `/weights/densepose` (**from Leffa volume**)
  - `/app/IDM-VTON/configs/densepose_rcnn_R_50_FPN_s1x.yaml` → `/weights/densepose/densepose_rcnn_R_50_FPN_s1x.yaml`
- **The cloned repo has an `./ckpt/` (singular!) directory with Git LFS pointer stubs. Must `shutil.rmtree` those before symlinking real files or ONNX Runtime tries to load the pointer as a model and dies with "Protobuf parsing failed".**

## Critical imports (this is the part I got wrong 15 times)

```python
from src.tryon_pipeline      import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon   import UNet2DConditionModel   # MAIN UNet — MUST be from here
from utils_mask              import get_mask_location       # in gradio_demo/, on PYTHONPATH
```

- **The main UNet is loaded from `src/unet_hacked_tryon.py`**, NOT from `diffusers`. This is the hacked class that supports `encoder_hid_dim_type='ip_image_proj'` (which stock diffusers 0.25 rejects).
- Files live in `src/` and `gradio_demo/` sub-packages — NOT at the repo root. Older IDM-VTON tutorials/READMEs show them at root; the repo was reorganized.
- **No monkey-patches needed** if you use the correct imports. I wasted hours patching PositionNet and `_set_encoder_hid_proj` when the real fix was just using `src.unet_hacked_tryon`.

## Pipeline call (exact — matches HF Spaces app.py)

```python
images = pipe(
    prompt_embeds=..., negative_prompt_embeds=..., pooled_prompt_embeds=..., negative_pooled_prompt_embeds=...,
    num_inference_steps=30,
    generator=torch.Generator("cuda").manual_seed(42),
    strength=1.0,
    pose_img=pose_tensor,               # DensePose viz, [1,3,1024,768] in [-1,1]
    text_embeds_cloth=prompt_embeds_c,  # from encode_prompt of "a photo of a shirt"
    cloth=garm_tensor,                  # garment tensor [1,3,1024,768] in [-1,1]
    mask_image=mask,                    # PIL binary mask
    image=human_img,                    # RAW person image (768x1024), NOT agnostic
    height=1024, width=768,
    ip_adapter_image=garm_img.resize((768, 1024)),
    guidance_scale=2.0,
)[0]
```

**Not "agnostic"** — I initially built a masked-out "agnostic" version of the person; the pipeline does that internally. Just pass the raw resized 768×1024 person image.

## pose_img = DensePose, NOT OpenPose visualization

OpenPose is used only for keypoints that feed into `get_mask_location`. The `pose_img` passed to the pipeline is a DensePose segmentation:

```python
import apply_net
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

human_arg = _apply_exif_orientation(human_img.resize((384, 512)))
human_arg = convert_PIL_to_numpy(human_arg, format="BGR")
args = apply_net.create_argument_parser().parse_args((
    "show",
    "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
    "./ckpt/densepose/model_final_162be9.pkl",
    "dp_segm", "-v",
    "--opts", "MODEL.DEVICE", "cuda",
))
pose_img = args.func(args, human_arg)
pose_img = pose_img[:, :, ::-1]  # BGR → RGB
pose_img = Image.fromarray(pose_img).resize((768, 1024))
```

This is why we need `detectron2` in the image (heavy build, ~5-8 min) AND why we mount the `leffa-weights` volume (has the DensePose weights).

---

## Every error we hit (chronological) and its actual fix

Use this table when the same class of error recurs.

| # | Error | Root cause | Fix |
|---|---|---|---|
| 1 | `charmap codec can't encode` on `Tee-Object` | PowerShell's cp1252 can't handle Modal's Unicode `━` progress bars | `$env:PYTHONIOENCODING = "utf-8"`; skip `Tee-Object` or use `-Encoding utf8` |
| 2 | "Image build ... terminated due to external shut-down" | Previous interrupted deploy left a half-built image on Modal | Just retry — Modal recognizes and resumes |
| 3 | `ModuleNotFoundError: matplotlib` | IDM's OpenPose util imports matplotlib | Add `"matplotlib"` to pip list |
| 4 | `Protobuf parsing failed` on `parsing_atr.onnx` | Repo has `ckpt/humanparsing/parsing_atr.onnx` as **Git LFS pointer stub** (~130 B text file), not a real model | `shutil.rmtree` the pointer dir before symlinking real weights from volume |
| 5 | Path was `ckpts/` (my code) vs `ckpt/` (IDM code) | Off-by-one on directory name (plural vs singular) | Use singular `ckpt/` |
| 6 | `ImportError: cannot import name 'cached_download' from 'huggingface_hub'` | `diffusers==0.25` still imports `cached_download`; modal mirror gave us `huggingface_hub==0.36.2` which removed it | Pin `huggingface_hub==0.25.0` (matches HF Space) |
| 7 | `ModuleNotFoundError: tryon_pipeline` | Not at repo root — lives in `src/tryon_pipeline.py` | Add `/app/IDM-VTON` to sys.path and import as `src.tryon_pipeline` |
| 8 | `ValueError: encoder_hid_dim_type: ip_image_proj must be None, 'text_proj' or 'text_image_proj'` | Loading UNet from stock `diffusers.UNet2DConditionModel` which doesn't support `ip_image_proj` | Import from `src.unet_hacked_tryon` — the HACKED version. `unet_hacked.py` doesn't exist in current repo; it's `unet_hacked_tryon.py`. |
| 9 | `ModuleNotFoundError: unet_hacked` | I invented a file name. Correct name is `unet_hacked_tryon.py` | See #8 |
| 10 | `ImportError: cannot import name 'PositionNet'` | `unet_hacked_garmnet.py` imports GLIGEN's `PositionNet` which was removed in diffusers 0.27+. Doesn't matter with diffusers 0.25 (where it still exists). | Use `diffusers==0.25.0` |
| 11 | `RuntimeError: Numpy is not available` inside diffusers scheduler | `numpy==2.x` incompatible with `torch==2.1.0` (torch 2.x only supports numpy 2.x from 2.3+) | Upgraded torch to 2.6.0 → OK |
| 12 | `numpy.dtype size changed 96 vs 88` in skimage | Modal mirror shipped `scikit-image==0.21.0` compiled for numpy 2.x, but numpy pinned to 1.24.4 (dtype size 88) | Let pip pick both: `numpy>=1.26,<3` + `scikit-image>=0.22` |
| 13 | `SIGSEGV` on first diffusion step; `"module compiled with NumPy 1.x"` | `onnxruntime==1.16.2` compiled for numpy 1.x; numpy 2.x installed | Bump `onnxruntime>=1.19` |
| 14 | Same error 3× in a row wasting money | Modal's default retry policy attempts failed inputs up to 3 times | `retries=0` on `@app.function` |

## Config attributes warning — safe to ignore

```
The config attributes {'decay': 0.9999, 'inv_gamma': 1.0, ...} were passed to
UNet2DConditionModel, but are not expected and will be ignored.
```

Comes from IDM-VTON's UNet config having EMA training-checkpoint metadata that isn't inference-relevant. Harmless.

## `add_embedding.linear_*` weight warning — safe to ignore

```
Some weights of the model checkpoint were not used when initializing UNet2DConditionModel:
['add_embedding.linear_1.bias, add_embedding.linear_1.weight, ...']
```

The hacked UNet doesn't use those add_embedding weights. Harmless.

---

## Cost tally (this session)

Modal usage during setup: **~$3-4** across many failed cold-starts, retries, and container SIGSEGVs. Most burned on:
- The 3× auto-retry issue before I added `retries=0`
- Multiple full image rebuilds (detectron2 alone is ~5 min = ~$0.07 each)

Running budget: user had ~$21 remaining before this session; probably ~$17 now.

## Retraining with LoRA — the honest answer

**Same dataset problem as Leffa** applies to IDM-VTON. LoRA fine-tuning on brand's self-reconstruction triplets (person=target, garment already-worn) will:
- Overfit to memorized pixels
- Produce phantom garments / poor generalization
- Same failure mode Leffa showed

Switching backbones does NOT fix a broken dataset. IDM-VTON LoRA needs **cross-outfit triplets**: same person, garment A → same person garment B target. Brand doesn't ship this data.

**Options if LoRA needed later**:
1. Get real cross-outfit triplets (best — requires photographer session)
2. Synthesize triplets with base IDM-VTON first, then fine-tune on those
3. Skip fine-tune entirely — base IDM-VTON quality may be enough

For now: **test base IDM-VTON on real brand garments** before deciding whether to fine-tune.

## License reminder — CC BY-NC-SA 4.0

- Personal / research / hobby use: free
- Commercial (selling try-on service, brand e-commerce use, integrating into paid app): needs commercial license from KAIST (`yisol@kaist.ac.kr`)
- If Madon plans to use this on customer-facing e-comm site, that IS commercial use — legal review needed

Leffa (Apache 2.0) is commercial-safe if IDM-VTON quality isn't worth the license issue.

## Related memory files

- `C:\Users\ASAD\.claude\projects\d--madon\memory\project_leffa_modal.md` — Leffa project state (winning config, dataset issues, checkpoint findings)
- `SESSION_NOTES.md` in same folder — earlier Leffa debugging session notes
