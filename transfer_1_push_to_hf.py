"""STEP 1 of workspace transfer: push idm-vton-weights volume from the source
workspace to a private HuggingFace dataset repo, which acts as the courier.

Run from the SOURCE workspace:

    modal profile activate abdullahsaleem75911
    .\.venv\Scripts\python.exe -m modal run transfer_1_push_to_hf.py \
        --hf-user YOUR_HF_USERNAME

Prereqs:
  - HF token with WRITE scope stored as secret `hf-token-write` in the
    source workspace, key HF_TOKEN=hf_xxx.
  - You own the target HF username.
"""

from __future__ import annotations

import modal


IDM_VOLUME = "idm-vton-weights"
IDM_MOUNT  = "/idm-weights"

idm_volume = modal.Volume.from_name(IDM_VOLUME)
app = modal.App("transfer-push-to-hf")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface_hub==0.25.0", "hf_transfer==0.1.8")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    volumes={IDM_MOUNT: idm_volume},
    secrets=[modal.Secret.from_name("hf-token-write")],
    timeout=60 * 60 * 3,
    cpu=4.0,
    memory=8192,
)
def push(hf_user: str) -> dict:
    import os
    import time
    from pathlib import Path

    from huggingface_hub import HfApi, create_repo

    token   = os.environ["HF_TOKEN"]
    repo_id = f"{hf_user}/idm-vton-weights-mirror"
    root    = Path(IDM_MOUNT)

    create_repo(repo_id=repo_id, repo_type="dataset",
                private=True, exist_ok=True, token=token)

    files = [p for p in root.rglob("*") if p.is_file()]
    total_gb = sum(p.stat().st_size for p in files) / 1e9
    print(f"[push] Uploading {len(files)} files ({total_gb:.1f} GB) -> {repo_id}", flush=True)

    t0 = time.time()
    HfApi(token=token).upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="mirror of Modal volume idm-vton-weights",
        ignore_patterns=["*.pyc", "__pycache__/*", ".git/*", ".cache/*"],
    )
    elapsed = time.time() - t0
    print(f"[push] Done in {elapsed:.1f}s", flush=True)

    return {
        "files":   len(files),
        "gb":      round(total_gb, 2),
        "seconds": round(elapsed, 1),
        "url":     f"https://huggingface.co/datasets/{repo_id}",
    }


@app.local_entrypoint()
def main(hf_user: str = "") -> None:
    if not hf_user:
        print("Usage: modal run transfer_1_push_to_hf.py --hf-user YOUR_HF_USERNAME")
        return
    report = push.remote(hf_user)
    print("\n=== PUSH REPORT ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("\nNext:")
    print("  modal profile activate fitcheckml")
    print(f"  modal run transfer_2_pull_from_hf.py --hf-user {hf_user}")
