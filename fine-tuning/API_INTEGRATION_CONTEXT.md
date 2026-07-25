# IDM-VTON + Shorts LoRA — API Integration Context

**For:** the next engineer/agent building an HTTP API wrapper around this virtual try-on model.
**Serving model:** Modal.com (already deployed, don't re-deploy the model — just call it).
**Behavior target:** produce identical results to the current `test_lora.py` local CLI.

Everything you need to build a REST/gRPC/whatever API is here. Skip nothing.

---

## 1. What the Model Is

**Composite:** Base IDM-VTON (SDXL virtual try-on) **+** shorts LoRA adapter, merged at load time.

- **Base:** ~3B params, `/idm-weights/IDM-VTON/` on Modal volume `idm-vton-weights`. Unchanged from upstream `yisol/IDM-VTON`.
- **LoRA:** 23M params (0.78% of base), cross-attention only (to_q, to_k, to_v, to_out.0), rank 16 / alpha 32. Trained on 16 shorts triplets, 30 epochs. Best loss 0.0176.
- **Merge:** at inference start, `peft.merge_and_unload()` bakes LoRA into base UNet in memory. Base weights on disk are never modified.
- **Result:** works well on ALL garment types (shirts, dresses, shorts) because base is intact. Slight positive bias on shorts because that's what LoRA was trained on.

---

## 2. What's Already Deployed on Modal — DO NOT REDEPLOY

**App name:** `idm-vton-lora-infer` (in Modal workspace `fitcheckml`)
**Live deployment URL:** https://modal.com/apps/fitcheckml/main/deployed/idm-vton-lora-infer

**Two functions exposed:**

### `run_idm_lora` — the main try-on function
- **GPU:** L4
- **Timeout:** 900 s
- **scaledown_window:** 900 s (container stays warm 15 min between calls)
- **Cold start:** ~90 s (SDXL load + preprocessors)
- **Warm inference:** ~30 s per request (30 diffusion steps)

**Signature:**
```python
run_idm_lora(
    person_bytes:   bytes,          # raw file bytes (jpg/png/webp)
    garment_bytes:  bytes,          # raw file bytes
    garment_type:   str   = "lower_body",   # upper_body | lower_body | dresses
    garment_desc:   str   = "",             # optional free-form, e.g. "denim shorts"
    steps:          int   = 30,             # diffusion steps
    guidance_scale: float = 2.0,
    seed:           int   = 42,
    lora_checkpoint:str   = "",             # Modal-side path — see §3
) -> dict
```

**Returns:**
```python
{
    "image":         bytes,      # generated try-on PNG
    "timings":       dict,       # preprocess_load_sec / preprocess_sec / model_load_sec / inference_sec / total_container_sec
    "steps":         int,
    "lora_applied":  bool,
}
```

### `list_checkpoints` — list available LoRA checkpoints on Modal volume
```python
list_checkpoints() -> list[str]   # e.g. ["/checkpoints/idmvton_lora_official_20260724_164142_e30.pt", ...]
```

---

## 3. The LoRA Checkpoint (the exact path that must be passed)

**Latest, best checkpoint (use this by default):**
```
/checkpoints/idmvton_lora_official_20260724_164142_e30.pt
```

Where this lives: Modal volume `idm-vton-checkpoints`, mounted at `/checkpoints` inside the container.

**Always pass this exact string as `lora_checkpoint` argument.** If you pass `""` (empty string), the model runs as pure baseline IDM-VTON (no shorts bias) — useful for A/B tests but not the default.

To discover other checkpoints programmatically:
```python
import modal
lister = modal.Function.from_name("idm-vton-lora-infer", "list_checkpoints")
ckpts  = lister.remote()   # returns sorted list
```

---

## 4. How to Call from Your API Code (the pattern to copy)

**Prerequisites on the API server:**
- `pip install modal`
- Authenticated: `modal token new` (uses the same fitcheckml workspace)

**Minimal invocation:**
```python
import modal
from pathlib import Path

# 1. Get a handle to the deployed function (does not deploy anything)
run_fn = modal.Function.from_name("idm-vton-lora-infer", "run_idm_lora")

# 2. Read bytes from wherever your API received the files (upload, S3, etc.)
person_bytes  = Path("uploads/person.jpg").read_bytes()
garment_bytes = Path("uploads/garment.jpg").read_bytes()

# 3. Call remote — blocking, returns dict
result = run_fn.remote(
    person_bytes,
    garment_bytes,
    garment_type   = "lower_body",
    steps          = 30,
    guidance_scale = 2.0,
    lora_checkpoint = "/checkpoints/idmvton_lora_official_20260724_164142_e30.pt",
)

# 4. result["image"] is PNG bytes — return to your client as-is
generated_png_bytes = result["image"]
```

**For async / non-blocking (Modal supports it):**
```python
call = run_fn.spawn(person_bytes, garment_bytes, ...)   # returns FunctionCall
# later: result = call.get(timeout=180)
```

**Reference implementation to study:** [test_lora.py](test_lora.py) lines 130-160 shows the exact call pattern that produces correct output.

---

## 5. Reference File Map — where things live in this repo

| File | Purpose | Read when |
|---|---|---|
| `fine-tuning/modal_infer_lora.py` | The Modal-side inference function (source of `run_idm_lora`). **Already deployed — do not redeploy.** | You want to understand the internals (VAE, DensePose, SCHP, OpenPose, pipeline call) or need to change GPU/timeout. |
| `fine-tuning/test_lora.py` | Local interactive CLI. **This is the reference for correct API behavior.** | Copy its `run_fn.remote(...)` invocation pattern. Composite image code (make_composite) is optional. |
| `fine-tuning/modal_train_lora_official.py` | LoRA training script (already-run) | Only if you need to retrain with a new dataset. Otherwise ignore. |
| `fine-tuning/TRAINING_CONTEXT.md` | Full log of how the LoRA was produced | Background context; NOT needed for API work. |
| `fine-tuning/data set/artifacts_shorts/dataset/` | The 19-sample training dataset | Reference / evaluation set. Don't ship it in your API. |

---

## 6. Modal Volumes (only relevant to the deployed inference function — you don't touch these from the API)

| Volume | Mount | Purpose |
|---|---|---|
| `idm-vton-weights` | `/idm-weights` | Base SDXL IDM-VTON checkpoint |
| `leffa-weights` | `/weights` | DensePose model files (reused) |
| `idm-vton-checkpoints` | `/checkpoints` | **Your LoRA lives here** — pass path as `lora_checkpoint` arg |
| `idm-vton-datasets` | `/data` | Training/eval data (not used by inference) |

---

## 7. Suggested HTTP API Shape (recommendation, adapt as needed)

```
POST /api/tryon
Content-Type: multipart/form-data

Fields:
  person:       file (required) — jpg/png/webp
  garment:      file (required) — jpg/png/webp
  garment_type: string (optional, default "lower_body") — upper_body | lower_body | dresses
  garment_desc: string (optional) — e.g. "denim shorts"
  steps:        int (optional, default 30)
  seed:         int (optional, default 42)
  use_lora:     bool (optional, default true)

Response:
  200 OK
  Content-Type: image/png
  <binary PNG bytes = result["image"]>

  Or JSON if you prefer:
  200 OK
  { "image_base64": "...", "timings": {...}, "lora_applied": true }
```

**Suggested wrapper server:** FastAPI + `uvicorn`. See minimal example in §8.

---

## 8. Minimal FastAPI Example (copy-paste-ready starting point)

```python
# api_server.py
import modal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response

app = FastAPI()

# Handle to already-deployed Modal function
run_fn = modal.Function.from_name("idm-vton-lora-infer", "run_idm_lora")

LORA_CKPT = "/checkpoints/idmvton_lora_official_20260724_164142_e30.pt"

@app.post("/api/tryon")
async def tryon(
    person:       UploadFile = File(...),
    garment:      UploadFile = File(...),
    garment_type: str  = Form("lower_body"),
    garment_desc: str  = Form(""),
    steps:        int  = Form(30),
    seed:         int  = Form(42),
    use_lora:     bool = Form(True),
):
    person_bytes  = await person.read()
    garment_bytes = await garment.read()

    result = run_fn.remote(
        person_bytes,
        garment_bytes,
        garment_type    = garment_type,
        garment_desc    = garment_desc,
        steps           = steps,
        seed            = seed,
        lora_checkpoint = LORA_CKPT if use_lora else "",
    )

    return Response(content=result["image"], media_type="image/png")
```

Run: `uvicorn api_server:app --host 0.0.0.0 --port 8000`

---

## 9. Behavior Guarantees — MUST match

To match `test_lora.py` output exactly, the API MUST:

1. **Always pass `lora_checkpoint`** = the path in §3 (unless client explicitly disables LoRA)
2. Pass raw file **bytes**, not paths, not base64 (Modal serializes bytes efficiently)
3. Preserve **default steps=30 and guidance_scale=2.0** unless client overrides
4. Not resize/preprocess images locally — the Modal function handles that
5. Return the **`result["image"]` PNG unchanged** — don't re-encode

If you follow these, results will be byte-identical (up to non-deterministic diffusion noise controlled by `seed`).

---

## 10. Cost + Performance Notes (relevant for capacity planning)

- **L4 GPU on Modal:** ~$0.80 / hr
- **Per-call cost:** ~$0.01–$0.03 (30 s warm, 30 diffusion steps)
- **Cold start:** ~90 s (~$0.02 extra one-off per cold container)
- **scaledown_window:** 900 s → container stays warm 15 min after last call, then spins down
- **Concurrency:** Modal auto-scales. Each parallel request gets its own container (each with its own ~90s cold start unless already warm)
- **Recommendation:** if you expect steady traffic, consider setting `min_containers=1` on the Modal function to keep one always warm (requires redeploying `modal_infer_lora.py` with that arg — talk to whoever owns the model first)

---

## 11. Auth / Secrets

- Modal auth: install `modal` package on API server, run `modal token new` once. Token is stored in `~/.modal.toml`. Same fitcheckml workspace.
- No API keys to inject into the model call itself.
- Add your own auth layer on top (JWT / API keys / whatever) in FastAPI middleware.

---

## 12. Error Handling — what to catch

```python
try:
    result = run_fn.remote(...)
except modal.exception.TimeoutError:
    # Modal function exceeded its 900s timeout
    return 504
except modal.exception.RemoteError as e:
    # Container crashed (OOM, model load fail, etc.)
    return 500, str(e)
except Exception as e:
    # Network / auth / other
    return 500, str(e)
```

`result` will always have `image`, `timings`, `steps`, `lora_applied` keys on success — no partial results.

---

## 13. When to Retrain (skip for now — but if you do)

The current LoRA was trained on 16 shorts triplets. To retrain:

- Extend dataset: aim for 200+ triplets for meaningfully-different output
- Follow format in `fine-tuning/data set/artifacts_shorts/dataset/manifest.jsonl`
- Run `modal run modal_train_lora_official.py::train` — produces new `.pt` in `/checkpoints/`
- Update `LORA_CKPT` constant in your API to the new file
- Zero downtime: old and new checkpoints coexist on the volume; changing the string is enough

---

## 14. Prompt for Your Coding Agent

Paste this to an LLM coding agent to bootstrap the API:

> Build a FastAPI server that wraps the Modal-deployed function `run_idm_lora` in app `idm-vton-lora-infer` for virtual try-on. Read all context from `fine-tuning/API_INTEGRATION_CONTEXT.md`. Requirements:
> 1. `POST /api/tryon` accepts person + garment image uploads (multipart) + optional garment_type/steps/seed.
> 2. Always pass `lora_checkpoint="/checkpoints/idmvton_lora_official_20260724_164142_e30.pt"` unless client sets `use_lora=false`.
> 3. Return the generated PNG as `image/png` response.
> 4. Add health check at `GET /health` that pings `list_checkpoints` remotely.
> 5. Do NOT redeploy the model — it's already deployed. Only call it via `modal.Function.from_name(...)`.
> 6. Match the exact call pattern in `fine-tuning/test_lora.py` lines 130-140.

---

## 15. Quick Sanity Test (before wiring your API)

```powershell
cd "d:\modal.com env\IDM-VTON\fine-tuning"
python test_lora.py
# pick checkpoint 3 (e30), select a person, select a garment, pick lower_body
# → should save vton_lora_<timestamp>.png in ../outputs/
```

If that works, the Modal-side model is fine. Any API bug is in your wrapper, not the model.

---

**End of context. Every path, argument name, and constant above is verified against the current deployment as of the last training run (2026-07-24).**
