"""One-shot seed script: populate fitcheckml's idm-vton-weights and leffa-weights
volumes by downloading directly from HuggingFace and Facebook public URLs.

Run this ONCE, from the fitcheckml Modal profile:

    modal profile activate fitcheckml
    .\.venv\Scripts\python.exe -m modal run seed_fitcheckml_volumes.py

Everything happens inside Modal — no bytes traverse your laptop. Total runtime
~15-30 min depending on HuggingFace throughput. Requires the `hf-token` secret
to already exist in fitcheckml (create it in the dashboard first).

After it finishes, the two volumes will have the exact layout that
modal_idm_inference.py expects, and you can `modal deploy modal_idm_inference.py`
unchanged.
"""

from __future__ import annotations

import modal


IDM_VOLUME   = "idm-vton-weights"
LEFFA_VOLUME = "leffa-weights"
IDM_MOUNT    = "/idm-weights"
LEFFA_MOUNT  = "/weights"

# get_or_create so a rerun after partial success doesn't blow up.
idm_volume   = modal.Volume.from_name(IDM_VOLUME,   create_if_missing=True)
leffa_volume = modal.Volume.from_name(LEFFA_VOLUME, create_if_missing=True)

app = modal.App("seed-fitcheckml-volumes")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("wget", "ca-certificates")
    .pip_install(
        "huggingface_hub==0.25.0",
        "hf_transfer==0.1.8",          # multipart parallel downloader, ~5x faster
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    volumes={IDM_MOUNT: idm_volume, LEFFA_MOUNT: leffa_volume},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=60 * 60,                    # 1 hour hard cap
    cpu=4.0,
    memory=8192,
)
def seed() -> dict:
    """Download IDM-VTON + DensePose into the two volumes with the exact layout
    that modal_idm_inference.py expects."""
    import os
    import shutil
    import subprocess
    import time
    from pathlib import Path

    from huggingface_hub import snapshot_download

    t0 = time.time()
    report: dict = {}

    # ── 1. IDM-VTON weights from yisol/IDM-VTON ─────────────────────────────
    # The HF repo contains both the SDXL model files AND the ckpt/ subtree
    # (humanparsing, openpose, densepose). We normalize the layout so it
    # matches what modal_idm_inference.py's symlink code expects.
    idm_target = Path(IDM_MOUNT) / "IDM-VTON"
    idm_target.mkdir(parents=True, exist_ok=True)

    print(f"[seed] Downloading yisol/IDM-VTON -> {idm_target} ...", flush=True)
    t = time.time()
    snapshot_download(
        repo_id="yisol/IDM-VTON",
        local_dir=str(idm_target),
        local_dir_use_symlinks=False,   # write real files to the volume
        max_workers=8,
        token=os.environ.get("HF_TOKEN"),
    )
    report["idm_download_sec"] = round(time.time() - t, 1)
    print(f"[seed] IDM-VTON download done in {report['idm_download_sec']}s", flush=True)

    # Normalize ckpt/ subtree: modal_idm_inference.py looks for humanparsing
    # and openpose at /idm-weights/IDM-VTON/{humanparsing,openpose}, but the
    # HF repo puts them under ckpt/. Move them up if present.
    ckpt_src = idm_target / "ckpt"
    if ckpt_src.exists():
        for sub in ("humanparsing", "openpose", "densepose", "image_adapter"):
            src = ckpt_src / sub
            dst = idm_target / sub
            if src.exists() and not dst.exists():
                print(f"[seed] Moving {src} -> {dst}", flush=True)
                shutil.move(str(src), str(dst))

    # Detect what actually landed so we can verify.
    def _listdir(p: Path) -> list[str]:
        return sorted([x.name for x in p.iterdir()]) if p.exists() else []

    report["idm_top_level"] = _listdir(idm_target)
    report["idm_humanparsing"] = _listdir(idm_target / "humanparsing")
    report["idm_openpose"] = _listdir(idm_target / "openpose")

    idm_volume.commit()
    print("[seed] idm-vton-weights volume committed", flush=True)

    # ── 2. DensePose weights into leffa-weights:/densepose ──────────────────
    # modal_idm_inference.py hard-codes /weights/densepose/model_final_162be9.pkl
    # and /weights/densepose/densepose_rcnn_R_50_FPN_s1x.yaml. The .pkl is
    # already inside yisol/IDM-VTON's ckpt/densepose/, so copy it over rather
    # than re-fetching from the internet. The .yaml is a detectron2 config
    # file — grab it from the facebookresearch GitHub raw URL.
    dp_target = Path(LEFFA_MOUNT) / "densepose"
    dp_target.mkdir(parents=True, exist_ok=True)

    idm_densepose_dir = idm_target / "densepose"
    if idm_densepose_dir.exists():
        for f in idm_densepose_dir.iterdir():
            if f.is_file():
                dst = dp_target / f.name
                if not dst.exists():
                    print(f"[seed] Copying {f.name} -> {dst}", flush=True)
                    shutil.copy2(str(f), str(dst))
    else:
        print("[seed] WARN: yisol/IDM-VTON did not ship a densepose/ dir. "
              "Fetching model_final_162be9.pkl from facebookresearch instead.",
              flush=True)
        subprocess.check_call([
            "wget", "-q", "--show-progress",
            "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_s1x/"
            "165712039/model_final_162be9.pkl",
            "-O", str(dp_target / "model_final_162be9.pkl"),
        ])

    # DensePose YAML config — never in HF repos, always from GitHub.
    yaml_dst = dp_target / "densepose_rcnn_R_50_FPN_s1x.yaml"
    if not yaml_dst.exists():
        print("[seed] Fetching densepose YAML config from facebookresearch/detectron2 ...", flush=True)
        subprocess.check_call([
            "wget", "-q",
            "https://raw.githubusercontent.com/facebookresearch/detectron2/main/"
            "projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
            "-O", str(yaml_dst),
        ])
        # This config `_BASE_`s Base-DensePose-RCNN-FPN.yaml — grab that too.
        base_yaml = dp_target / "Base-DensePose-RCNN-FPN.yaml"
        if not base_yaml.exists():
            subprocess.check_call([
                "wget", "-q",
                "https://raw.githubusercontent.com/facebookresearch/detectron2/main/"
                "projects/DensePose/configs/Base-DensePose-RCNN-FPN.yaml",
                "-O", str(base_yaml),
            ])

    report["densepose_files"] = _listdir(dp_target)
    leffa_volume.commit()
    print("[seed] leffa-weights volume committed", flush=True)

    report["total_sec"] = round(time.time() - t0, 1)

    # ── 3. Sanity checks ────────────────────────────────────────────────────
    required_sdxl = {"unet", "vae", "text_encoder", "text_encoder_2",
                     "image_encoder", "tokenizer", "tokenizer_2",
                     "scheduler", "unet_encoder"}
    missing_sdxl = required_sdxl - set(report["idm_top_level"])
    report["missing_sdxl_subfolders"] = sorted(missing_sdxl)

    required_dp = {"model_final_162be9.pkl", "densepose_rcnn_R_50_FPN_s1x.yaml"}
    missing_dp = required_dp - set(report["densepose_files"])
    report["missing_densepose_files"] = sorted(missing_dp)

    if missing_sdxl or missing_dp:
        print(f"[seed] ⚠ Missing files: SDXL={missing_sdxl}  DensePose={missing_dp}",
              flush=True)
    else:
        print("[seed] ✓ All required files present in both volumes", flush=True)

    return report


@app.local_entrypoint()
def main() -> None:
    print("Seeding fitcheckml volumes (this will take 15-30 min) ...")
    result = seed.remote()
    print("\n=== SEED REPORT ===")
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 12:
            print(f"  {k}: {v[:12]} ... ({len(v)} total)")
        else:
            print(f"  {k}: {v}")
