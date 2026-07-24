"""Modal.com deployment for IDM-VTON inference.

Mirrors the exact working configuration from yisol/IDM-VTON's HF Space:
  - diffusers==0.25.0, huggingface_hub==0.25.0, transformers==4.36.2
  - numpy==1.24.4 (torch 2.x doesn't support numpy 2.x)
  - Main UNet loaded from `src.unet_hacked_tryon` (patched for ip_image_proj)
  - Garment encoder from `src.unet_hacked_garmnet`
  - Pose conditioning = DensePose (via apply_net + detectron2)
  - Mask = OpenPose keypoints + SCHP parsing -> utils_mask.get_mask_location
  - Volume mounts:
      /idm-weights  ← idm-vton-weights (SDXL models, humanparsing, openpose)
      /weights      ← leffa-weights   (reuses existing DensePose model files)

Usage:
    cd "D:\\madon finalization"
    .\.venv\Scripts\python.exe test_idm.py

Cost: L4 ~$0.80/hr — ~$0.10 per inference (SDXL is heavier than SD1.5)
"""

from __future__ import annotations

import io
from pathlib import Path

import modal


IDM_VOLUME    = "idm-vton-weights"
LEFFA_VOLUME  = "leffa-weights"          # reused for DensePose weights only
IDM_MOUNT     = "/idm-weights"
LEFFA_MOUNT   = "/weights"
MODEL_PATH    = "/idm-weights/IDM-VTON"

app          = modal.App("idm-vton")
idm_volume   = modal.Volume.from_name(IDM_VOLUME)
leffa_volume = modal.Volume.from_name(LEFFA_VOLUME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "git-lfs",
        "libgl1", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev",
        "build-essential", "ninja-build",
    )
    # Torch 2.6+cu124 — matches what worked for Leffa (detectron2 builds against this)
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Exact versions from IDM-VTON's HF Space requirements.txt
    .pip_install([
        "diffusers==0.25.0",
        "huggingface_hub==0.25.0",
        "transformers==4.36.2",
        "accelerate==0.26.1",
        # Modal's mirror ships scikit-image 0.21 built against numpy 2.x; torch 2.6+cu124
        # supports numpy 2, so let pip pick compatible pair rather than fighting the ABI.
        "numpy>=1.26,<3",
        "scipy>=1.10",
        "scikit-image>=0.22",
        "opencv-python==4.7.0.72",
        "pillow>=9.4,<11",
        "einops==0.7.0",
        "matplotlib==3.7.4",
        "onnxruntime>=1.19",             # 1.16 was compiled for numpy 1.x -> SIGSEGV on numpy 2
        "omegaconf",
        "fvcore",
        "pycocotools",
        "cloudpickle",
        "av",
        "config==0.5.1",
        "safetensors",
        "tqdm==4.64.1",
    ])
    # detectron2 + DensePose — used by gradio_demo/apply_net.py for pose_img
    .run_commands(
        "pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation",
        "pip install 'git+https://github.com/facebookresearch/detectron2.git#subdirectory=projects/DensePose' --no-build-isolation",
        gpu="L4",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/yisol/IDM-VTON /app/IDM-VTON",
    )
    # Repo layout: src/ (pipeline, hacked UNets), gradio_demo/ (utils_mask, apply_net)
    .env({"PYTHONPATH": "/app/IDM-VTON:/app/IDM-VTON/gradio_demo"})
)


@app.function(
    gpu="L4",
    image=image,
    volumes={
        IDM_MOUNT:   idm_volume,
        LEFFA_MOUNT: leffa_volume,   # for DensePose model files
    },
    timeout=900,
    retries=0,                       # never auto-retry: failure means fix the code, not burn cycles
    scaledown_window=900,            # keep container warm 15 min between tests
)
def run_idm(
    person_bytes:   bytes,
    garment_bytes:  bytes,
    garment_type:   str   = "upper_body",   # upper_body | lower_body | dresses
    garment_desc:   str   = "",
    steps:          int   = 30,
    guidance_scale: float = 2.0,
    seed:           int   = 42,
) -> dict:
    """Run IDM-VTON on L4. Returns {'image': PNG bytes, 'timings': dict, 'steps': int}."""
    import os
    import sys
    import time
    import shutil
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    # IDM-VTON code uses relative paths like ./ckpt/ and ./configs/. chdir into repo.
    os.chdir("/app/IDM-VTON")
    for p in ("/app/IDM-VTON", "/app/IDM-VTON/gradio_demo"):
        if p not in sys.path:
            sys.path.insert(0, p)

    t_fn_start = time.time()
    timings: dict[str, float] = {}

    # ── 1. Wire up model files into repo-relative paths ─────────────────────
    # apply_net wants ./configs/densepose_*.yaml and ./ckpt/densepose/*.pkl
    # SCHP parser wants ./ckpt/humanparsing/*.onnx
    # OpenPose wants ./ckpt/openpose/*.pth
    ckpt = Path("/app/IDM-VTON/ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)

    def _relink(dst: Path, src: str) -> None:
        if dst.exists() and not dst.is_symlink():
            shutil.rmtree(str(dst))
        if not dst.is_symlink():
            os.symlink(src, str(dst))

    _relink(ckpt / "humanparsing", f"{MODEL_PATH}/humanparsing")
    _relink(ckpt / "openpose",     f"{MODEL_PATH}/openpose")
    _relink(ckpt / "densepose",    f"{LEFFA_MOUNT}/densepose")

    configs_dir = Path("/app/IDM-VTON/configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    dp_yaml_dst = configs_dir / "densepose_rcnn_R_50_FPN_s1x.yaml"
    dp_yaml_src = f"{LEFFA_MOUNT}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml"
    if not dp_yaml_dst.exists() and not dp_yaml_dst.is_symlink():
        os.symlink(dp_yaml_src, str(dp_yaml_dst))

    # ── 2. Load SCHP parser + OpenPose (for mask) ────────────────────────────
    t0 = time.time()
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose   import OpenPose
    parsing_model  = Parsing(0)
    openpose_model = OpenPose(0)
    timings["preprocess_load_sec"] = time.time() - t0
    print(f"[IDM] SCHP+OpenPose loaded in {timings['preprocess_load_sec']:.1f}s", flush=True)

    # ── 3. Load IDM-VTON pipeline (matches HF Space's app.py exactly) ────────
    t0 = time.time()
    from transformers import (
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )
    from diffusers import AutoencoderKL, DDPMScheduler
    from src.tryon_pipeline      import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon   import UNet2DConditionModel   # patched for ip_image_proj

    dtype = torch.float16
    print(f"[IDM] Loading SDXL components from {MODEL_PATH} ...", flush=True)

    unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet", torch_dtype=dtype)
    unet.requires_grad_(False)

    tokenizer_one = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer",   revision=None, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2", revision=None, use_fast=False)
    scheduler     = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    text_encoder_one = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder="text_encoder_2", torch_dtype=dtype)
    image_encoder    = CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder="image_encoder", torch_dtype=dtype)
    vae              = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=dtype)
    unet_encoder     = UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder="unet_encoder", torch_dtype=dtype)

    for _m in (text_encoder_one, text_encoder_two, image_encoder, vae, unet_encoder):
        _m.requires_grad_(False)

    pipe = TryonPipeline.from_pretrained(
        MODEL_PATH,
        unet             = unet,
        vae              = vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder     = text_encoder_one,
        text_encoder_2   = text_encoder_two,
        tokenizer        = tokenizer_one,
        tokenizer_2      = tokenizer_two,
        scheduler        = scheduler,
        image_encoder    = image_encoder,
        torch_dtype      = dtype,
    )
    pipe.unet_encoder = unet_encoder
    pipe = pipe.to("cuda")
    pipe.unet_encoder = pipe.unet_encoder.to("cuda")
    timings["model_load_sec"] = time.time() - t0
    print(f"[IDM] Pipeline loaded in {timings['model_load_sec']:.1f}s", flush=True)

    # ── 4. Preprocess input images ───────────────────────────────────────────
    t0 = time.time()
    human_img = Image.open(io.BytesIO(person_bytes)).convert("RGB").resize((768, 1024))
    garm_img  = Image.open(io.BytesIO(garment_bytes)).convert("RGB").resize((768, 1024))

    # DensePose visualization → pose_img (exact code path from HF Space app.py)
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
    pose_img = pose_img[:, :, ::-1]                          # BGR → RGB
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    # Clothing-agnostic mask via SCHP + OpenPose
    keypoints        = openpose_model(human_img.resize((384, 512)))
    model_parse, _   = parsing_model(human_img.resize((384, 512)))
    from utils_mask import get_mask_location
    mask, _ = get_mask_location("hd", garment_type, model_parse, keypoints)
    mask = mask.resize((768, 1024))

    tensor_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    garm_tensor = tensor_tfm(garm_img).unsqueeze(0)
    pose_tensor = tensor_tfm(pose_img).unsqueeze(0)
    timings["preprocess_sec"] = time.time() - t0
    print(f"[IDM] Preprocess done in {timings['preprocess_sec']:.1f}s "
          f"(DensePose + SCHP + OpenPose)", flush=True)

    # ── 5. Encode prompts ────────────────────────────────────────────────────
    _desc_default = {
        "upper_body": "a shirt",
        "lower_body": "pants",
        "dresses":    "a dress",
    }
    desc       = garment_desc.strip() or _desc_default.get(garment_type, "a garment")
    prompt     = f"model is wearing {desc}"
    neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    with torch.inference_mode():
        (prompt_embeds,
         negative_prompt_embeds,
         pooled_prompt_embeds,
         negative_pooled_prompt_embeds) = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=neg_prompt,
        )
        (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
            f"a photo of {desc}",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=neg_prompt,
        )

    # ── 6. Diffusion inference (call signature matches HF Space app.py) ──────
    t0 = time.time()
    device    = "cuda"
    generator = torch.Generator(device).manual_seed(seed)
    print(f"[IDM] Running {steps}-step diffusion (guidance={guidance_scale}) ...", flush=True)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            images = pipe(
                prompt_embeds                 = prompt_embeds.to(device, dtype),
                negative_prompt_embeds        = negative_prompt_embeds.to(device, dtype),
                pooled_prompt_embeds          = pooled_prompt_embeds.to(device, dtype),
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device, dtype),
                num_inference_steps           = steps,
                generator                     = generator,
                strength                      = 1.0,
                pose_img                      = pose_tensor.to(device, dtype),
                text_embeds_cloth             = prompt_embeds_c.to(device, dtype),
                cloth                         = garm_tensor.to(device, dtype),
                mask_image                    = mask,
                image                         = human_img,             # raw, not agnostic
                height                        = 1024,
                width                         = 768,
                ip_adapter_image              = garm_img.resize((768, 1024)),
                guidance_scale                = guidance_scale,
            )[0]

    timings["inference_sec"]       = time.time() - t0
    timings["total_container_sec"] = time.time() - t_fn_start
    print(f"[IDM] Inference done in {timings['inference_sec']:.1f}s "
          f"(total: {timings['total_container_sec']:.1f}s)", flush=True)

    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return {"image": buf.getvalue(), "timings": timings, "steps": steps}


@app.local_entrypoint()
def main(
    person:       str = "",
    garment:      str = "",
    output:       str = "idm_output.png",
    garment_type: str = "upper_body",
    steps:        int = 30,
    desc:         str = "",
) -> None:
    import time

    if not person or not garment:
        print("Usage: modal run modal_idm_inference.py --person p.jpg --garment g.jpg")
        return

    person_bytes  = Path(person).read_bytes()
    garment_bytes = Path(garment).read_bytes()

    print(f"Person  : {person}")
    print(f"Garment : {garment}")
    print(f"Type    : {garment_type}  |  Steps: {steps}")
    print("Running IDM-VTON on Modal L4 GPU...")

    t0 = time.time()
    result = run_idm.remote(
        person_bytes, garment_bytes,
        garment_type = garment_type,
        garment_desc = desc,
        steps        = steps,
    )
    elapsed = time.time() - t0

    out = Path(output)
    out.write_bytes(result["image"])
    print(f"\nDone in {elapsed:.1f}s  →  {out.resolve()}")
    print(f"Timings: {result['timings']}")
