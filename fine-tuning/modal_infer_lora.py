"""
IDM-VTON Inference with Trained LoRA (Modal side).

Loads the base IDM-VTON pipeline PLUS the trained shorts LoRA checkpoint
and runs inference. Mirrors modal_idm_inference.py's run_idm() but merges
LoRA weights into the UNet before inference.

Called by test_lora.py (local interactive CLI).
"""

from __future__ import annotations

import io
from pathlib import Path
import modal


IDM_VOLUME    = "idm-vton-weights"
LEFFA_VOLUME  = "leffa-weights"
IDM_MOUNT     = "/idm-weights"
LEFFA_MOUNT   = "/weights"
MODEL_PATH    = "/idm-weights/IDM-VTON"

app                 = modal.App("idm-vton-lora-infer")
idm_volume          = modal.Volume.from_name(IDM_VOLUME)
leffa_volume        = modal.Volume.from_name(LEFFA_VOLUME)
checkpoints_volume  = modal.Volume.from_name("idm-vton-checkpoints", create_if_missing=True)

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
        "fvcore",
        "pycocotools",
        "cloudpickle",
        "av",
        "config==0.5.1",
        "safetensors",
        "tqdm==4.64.1",
    ])
    .run_commands(
        "pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation",
        "pip install 'git+https://github.com/facebookresearch/detectron2.git#subdirectory=projects/DensePose' --no-build-isolation",
        gpu="L4",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/yisol/IDM-VTON /app/IDM-VTON",
    )
    .env({"PYTHONPATH": "/app/IDM-VTON:/app/IDM-VTON/gradio_demo"})
)


@app.function(
    gpu="L4",
    image=image,
    volumes={
        IDM_MOUNT:      idm_volume,
        LEFFA_MOUNT:    leffa_volume,
        "/checkpoints": checkpoints_volume,
    },
    timeout=900,
    retries=0,
    scaledown_window=900,
)
def run_idm_lora(
    person_bytes:   bytes,
    garment_bytes:  bytes,
    garment_type:   str   = "lower_body",
    garment_desc:   str   = "",
    steps:          int   = 30,
    guidance_scale: float = 2.0,
    seed:           int   = 42,
    lora_checkpoint:str   = "",
) -> dict:
    """Run IDM-VTON with LoRA applied. Returns {'image': PNG bytes, 'timings': dict}."""
    import os, sys, time, shutil
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    os.chdir("/app/IDM-VTON")
    for p in ("/app/IDM-VTON", "/app/IDM-VTON/gradio_demo"):
        if p not in sys.path:
            sys.path.insert(0, p)

    t_start = time.time()
    timings: dict[str, float] = {}

    # Wire model files (same as modal_idm_inference.py)
    ckpt = Path("/app/IDM-VTON/ckpt"); ckpt.mkdir(parents=True, exist_ok=True)
    def _relink(dst: Path, src: str) -> None:
        if dst.exists() and not dst.is_symlink():
            shutil.rmtree(str(dst))
        if not dst.is_symlink():
            os.symlink(src, str(dst))
    _relink(ckpt / "humanparsing", f"{MODEL_PATH}/humanparsing")
    _relink(ckpt / "openpose",     f"{MODEL_PATH}/openpose")
    _relink(ckpt / "densepose",    f"{LEFFA_MOUNT}/densepose")
    configs_dir = Path("/app/IDM-VTON/configs"); configs_dir.mkdir(parents=True, exist_ok=True)
    dp_yaml_dst = configs_dir / "densepose_rcnn_R_50_FPN_s1x.yaml"
    dp_yaml_src = f"{LEFFA_MOUNT}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml"
    if not dp_yaml_dst.exists() and not dp_yaml_dst.is_symlink():
        os.symlink(dp_yaml_src, str(dp_yaml_dst))

    # Load preprocessors
    t0 = time.time()
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose   import OpenPose
    parsing_model  = Parsing(0)
    openpose_model = OpenPose(0)
    timings["preprocess_load_sec"] = time.time() - t0

    # Load IDM-VTON pipeline
    t0 = time.time()
    from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPTextModel,
                              CLIPTextModelWithProjection, CLIPVisionModelWithProjection)
    from diffusers import AutoencoderKL, DDPMScheduler
    from src.tryon_pipeline      import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon   import UNet2DConditionModel

    dtype = torch.float16
    unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet", torch_dtype=dtype)
    unet.requires_grad_(False)

    # ── Apply LoRA if checkpoint provided ────────────────────────────────────
    lora_applied = False
    if lora_checkpoint:
        ckpt_path = Path(lora_checkpoint)
        if ckpt_path.exists():
            from peft import LoraConfig, get_peft_model
            print(f"[LoRA] Loading {ckpt_path.name}...", flush=True)
            data = torch.load(ckpt_path, map_location="cpu")
            cfg  = data["lora_config"]
            lora_cfg = LoraConfig(
                r=cfg["r"], lora_alpha=cfg["alpha"],
                target_modules=cfg["target_modules"],
                lora_dropout=0.0, bias="none",
            )
            unet = get_peft_model(unet, lora_cfg)
            missing, unexpected = unet.load_state_dict(data["lora_state_dict"], strict=False)
            print(f"[LoRA] Loaded ({len(data['lora_state_dict'])} tensors, "
                  f"missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
            # Merge LoRA into base so downstream code sees a normal UNet
            unet = unet.merge_and_unload()
            lora_applied = True
        else:
            print(f"[LoRA] WARN: checkpoint not found: {ckpt_path}", flush=True)

    tokenizer_one    = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer",   use_fast=False)
    tokenizer_two    = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2", use_fast=False)
    scheduler        = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
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

    # Preprocess
    t0 = time.time()
    human_img = Image.open(io.BytesIO(person_bytes)).convert("RGB").resize((768, 1024))
    garm_img  = Image.open(io.BytesIO(garment_bytes)).convert("RGB").resize((768, 1024))

    import apply_net
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    human_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_arg = convert_PIL_to_numpy(human_arg, format="BGR")
    args = apply_net.create_argument_parser().parse_args((
        "show", "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
        "./ckpt/densepose/model_final_162be9.pkl",
        "dp_segm", "-v", "--opts", "MODEL.DEVICE", "cuda",
    ))
    pose_img = args.func(args, human_arg)[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    keypoints      = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    from utils_mask import get_mask_location
    mask, _ = get_mask_location("hd", garment_type, model_parse, keypoints)
    mask = mask.resize((768, 1024))

    tensor_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    garm_tensor = tensor_tfm(garm_img).unsqueeze(0)
    pose_tensor = tensor_tfm(pose_img).unsqueeze(0)
    timings["preprocess_sec"] = time.time() - t0

    # Prompt
    _defaults = {"upper_body": "a shirt", "lower_body": "shorts", "dresses": "a dress"}
    desc       = garment_desc.strip() or _defaults.get(garment_type, "a garment")
    prompt     = f"model is wearing {desc}"
    neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    with torch.inference_mode():
        (pe, npe, ppe, nppe) = pipe.encode_prompt(
            prompt, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=neg_prompt)
        (pe_c, _, _, _) = pipe.encode_prompt(
            f"a photo of {desc}", num_images_per_prompt=1,
            do_classifier_free_guidance=False, negative_prompt=neg_prompt)

    # Inference
    t0 = time.time()
    generator = torch.Generator("cuda").manual_seed(seed)
    with torch.no_grad(), torch.cuda.amp.autocast():
        images = pipe(
            prompt_embeds=pe.to("cuda", dtype),
            negative_prompt_embeds=npe.to("cuda", dtype),
            pooled_prompt_embeds=ppe.to("cuda", dtype),
            negative_pooled_prompt_embeds=nppe.to("cuda", dtype),
            num_inference_steps=steps,
            generator=generator,
            strength=1.0,
            pose_img=pose_tensor.to("cuda", dtype),
            text_embeds_cloth=pe_c.to("cuda", dtype),
            cloth=garm_tensor.to("cuda", dtype),
            mask_image=mask,
            image=human_img,
            height=1024, width=768,
            ip_adapter_image=garm_img.resize((768, 1024)),
            guidance_scale=guidance_scale,
        )[0]
    timings["inference_sec"]       = time.time() - t0
    timings["total_container_sec"] = time.time() - t_start

    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return {"image": buf.getvalue(), "timings": timings,
            "steps": steps, "lora_applied": lora_applied}


@app.function(
    image=modal.Image.debian_slim().pip_install("torch"),
    volumes={"/checkpoints": checkpoints_volume},
)
def list_checkpoints() -> list[str]:
    """List all LoRA checkpoints available on Modal volume."""
    return sorted(str(p) for p in Path("/checkpoints").glob("idmvton_lora_*.pt"))
