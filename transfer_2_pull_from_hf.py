"""STEP 2 of workspace transfer: pull idm-vton-weights from the HF courier
into a fresh volume in fitcheckml.

Run from the TARGET workspace:

    modal profile activate fitcheckml
    .\.venv\Scripts\python.exe -m modal run transfer_2_pull_from_hf.py \
        --hf-user YOUR_HF_USERNAME

Prereqs:
  - Step 1 completed.
  - Secret `hf-token` exists in fitcheckml, key HF_TOKEN=hf_xxx (read scope OK).
"""

from __future__ import annotations

import modal


IDM_VOLUME = "idm-vton-weights"
IDM_MOUNT  = "/idm-weights"

idm_volume = modal.Volume.from_name(IDM_VOLUME, create_if_missing=True)
app = modal.App("transfer-pull-from-hf")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface_hub==0.25.0", "hf_transfer==0.1.8")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    volumes={IDM_MOUNT: idm_volume},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=60 * 60 * 3,
    cpu=4.0,
    memory=8192,
)
def pull(hf_user: str) -> dict:
    import os
    import time
    from pathlib import Path

    from huggingface_hub import snapshot_download

    token   = os.environ["HF_TOKEN"]
    repo_id = f"{hf_user}/idm-vton-weights-mirror"
    target  = Path(IDM_MOUNT)
    target.mkdir(parents=True, exist_ok=True)

    print(f"[pull] Downloading {repo_id} -> {IDM_MOUNT}", flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        max_workers=8,
        token=token,
    )
    elapsed = time.time() - t0

    files = [p for p in target.rglob("*") if p.is_file()]
    total_gb = sum(p.stat().st_size for p in files) / 1e9

    idm_volume.commit()
    print(f"[pull] Done in {elapsed:.1f}s ({len(files)} files, {total_gb:.1f} GB), volume committed", flush=True)

    return {
        "files":   len(files),
        "gb":      round(total_gb, 2),
        "seconds": round(elapsed, 1),
    }


@app.local_entrypoint()
def main(hf_user: str = "") -> None:
    if not hf_user:
        print("Usage: modal run transfer_2_pull_from_hf.py --hf-user YOUR_HF_USERNAME")
        return
    report = pull.remote(hf_user)
    print("\n=== PULL REPORT ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
