# TRIPLETS.md — how triplets get built + how folders are laid out

**Audience:** future Claude sessions that need to understand, extend, or debug the data preparation pipeline. Read alongside [`MASKING.md`](MASKING.md) — this doc explains WHAT we train on, that one explains WHAT PREPROCESSING each element gets.

---

## TL;DR

A **triplet** is the unit of training data. It says:

> "Given this person photo (`person`), this garment flatlay (`garment`), and precomputed conditioning (`mask`, `densepose`), produce this target photo (`target`)."

The Leffa VITON-HD training loop consumes triplets one at a time and computes MSE noise-prediction loss against `target`. There are **two kinds** of triplets, both crucial for the LoRA to learn anything useful:

1. **`self_recon`** — person A wearing garment X + garment X flatlay → target = person A (i.e., same photo as person). Teaches identity reconstruction.
2. **`cross_outfit`** — person A wearing garment X + garment Y flatlay → target = **real photo of person A actually wearing Y** (from the same shoot). Teaches actual try-on with ground-truth targets.

**Both are stored as one flat `manifest.jsonl` file**, each line is one triplet. Filenames reference PNG/JPG files in five sibling folders (`person/`, `garment/`, `target/`, `mask/`, `densepose/`).

The manifest is what the trainer iterates over. The folder layout supports **file dedup** — the same person shot may appear in dozens of cross_outfit triplets but the file lives on disk exactly once.

---

## The two triplet kinds — WHY both

### self_recon (~15% of data)

```
person   = brand_dataset_v2/person/polo_ps2624_grn_<hash>.jpg
garment  = brand_dataset_v2/garment/polo_ps2624_grn_<hash>.jpg
target   = brand_dataset_v2/target/polo_ps2624_grn_<hash>.jpg    ← same file as person
_kind    = "self_recon"
```

**What it teaches**: given a person wearing garment X and the flatlay of X, reproduce that same person wearing X. Trivially "easy" from a reconstruction standpoint — the answer is right there in the input — but it forces the LoRA to learn faithful garment→body mapping.

**Why include it at all**: without self_recon, the LoRA would only see novel combinations and might drift from identity preservation. Self_recon anchors the model to "yes, you should reproduce the reference garment closely."

### cross_outfit (~85% of data)

```
person   = brand_dataset_v2/person/polo_ps2624_grn_<hash_A>.jpg   ← model_3 in polo
garment  = brand_dataset_v2/garment/tank_top_mtt506_blk_<hash>.jpg ← different SKU flatlay
target   = brand_dataset_v2/target/tank_top_mtt506_blk_<hash_B>.jpg ← model_3 wearing tank top
_kind    = "cross_outfit"
_sku_person   = "POLO/PS2624-00S-GRN"
_sku_garment  = "TANK TOP/MTT506-00S-BLK"
_model_id     = "model_3"
```

**What it teaches**: real virtual try-on. Person is wearing one thing, garment reference is a different thing, target is what they ACTUALLY look like wearing the reference (a real photograph, not a synthetic composite).

This is the gold signal. It's why we spent effort on face clustering (step_02) — we NEED to know that person_A and target_B are the same actual person.

### The session-histogram gate (why we need it)

If cross_outfit pairs were random, we'd get shots from different photoshoots — different lighting, different backgrounds, different poses. That would teach the model "when you try on a new garment, everything about the image should change." Wrong signal.

Fix: **pair only shots from the same shoot / session** via a color histogram similarity test on the *non-garment* regions of the two person images.

```python
# From step_03_build_manifest.py
sim = hist_cache[(a["person"], gt)] @ hist_cache[(b["person"], gt)]
if sim < SESSION_HIST_SIM_THRESHOLD:
    continue    # reject this pair
```

`_non_target_hist` computes color histograms while masking out the garment region itself (so we compare BACKGROUND + skin + hair, not garment which is expected to differ). Threshold is tuned so only same-session pairs survive.

Also caps pairs per (garment_type, model) via `MAX_CROSS_PAIRS_PER_MODEL_CATEGORY` so one busy model doesn't dominate the training distribution.

---

## The manifest format

Two files, both in `brand_dataset_v2/`:

- **`manifest.jsonl`** — training triplets (804 lines currently: 504 upper + 300 lower)
- **`manifest_heldout.jsonl`** — held-out for eval (13 lines, no overlap with training)

Each line is one JSON object:

```json
{
  "person":       "person/polo_ps2624_grn_4f52a11a.jpg",
  "garment":      "garment/polo_ps2624_grn_ab72d4bc.jpg",
  "target":       "target/polo_ps2624_grn_4f52a11a.jpg",
  "garment_type": "upper_body",
  "_kind":        "self_recon",
  "_sku_person":  "POLO/PS2624-00S-GRN",
  "_sku_garment": "POLO/PS2624-00S-GRN",
  "_model_id":    "model_3",
  "mask":         "mask/polo_ps2624_grn_4f52a11a.png",
  "densepose":    "densepose/polo_ps2624_grn_4f52a11a.png"
}
```

**Field-by-field:**

| Field | Type | Purpose |
|---|---|---|
| `person` | str, relative path | The source person image (input to inference) |
| `garment` | str, relative path | The garment flatlay to try on |
| `target` | str, relative path | The ground-truth output image (training target) |
| `garment_type` | `"upper_body"` \| `"lower_body"` | Which LoRA this triplet trains |
| `_kind` | `"self_recon"` \| `"cross_outfit"` | Kind of pair (leading underscore = metadata only) |
| `_sku_person` | str | SKU of what person is wearing (metadata, for auditing) |
| `_sku_garment` | str | SKU of the garment reference (metadata) |
| `_model_id` | str | Face-cluster ID (`"model_3"`, etc.) |
| `mask` | str, relative path | Precomputed agnostic mask (added by step_06) |
| `densepose` | str, relative path | Precomputed densepose visualization (added by step_06) |

**Path convention**: all paths are **relative to `brand_dataset_v2/`**. Absolute paths would break Modal ↔ local portability.

**Field naming**: fields with a leading underscore (`_kind`, `_sku_*`, `_model_id`) are **metadata for humans and for auditing** — the trainer ignores them. Non-underscore fields (`person`, `garment`, `target`, `garment_type`, `mask`, `densepose`) are what the training loop actually reads.

---

## Folder layout on disk (and on the Modal volume)

Structure of `brand_dataset_v2/` (identical local and on Modal volume `leffa-weights:/brand_dataset_v2/`):

```
brand_dataset_v2/
├── manifest.jsonl                    # 804 training triplets
├── manifest_heldout.jsonl            # 13 held-out triplets (for eval)
├── person/                           # 129 unique person shots (JPG)
│   ├── polo_ps2624_grn_4f52a11a.jpg
│   ├── polo_pe2667_cwt_d6abeb0a.jpg
│   ├── shirt_5_e3b3ac5c.jpg
│   └── ... (~127 more)
├── garment/                          # 124 unique garment flatlays (JPG)
│   ├── polo_ps2624_grn_ab72d4bc.jpg
│   ├── tank_top_mtt506_blk_<hash>.jpg
│   └── ...
├── target/                           # 129 target photos (JPG, mostly same set as person/)
│   ├── polo_ps2624_grn_4f52a11a.jpg
│   ├── tank_top_mtt506_blk_<hash>.jpg
│   └── ...
├── mask/                             # 128 binary masks (PNG, one per unique person shot)
│   ├── polo_ps2624_grn_4f52a11a.png
│   └── ...
└── densepose/                        # 128 IUV visualizations (PNG, one per unique person shot)
    ├── polo_ps2624_grn_4f52a11a.png
    └── ...
```

### File counts don't match triplet count — WHY

- **Triplets**: 804 in manifest
- **Unique person files**: 129 (each person appears in ~6 cross_outfit triplets on average)
- **Unique garment files**: 124 (each garment flatlay reused across triplets)
- **Unique target files**: 129 (same set as person/ for self_recon, superset for cross_outfit)
- **Mask + densepose files**: 128 (one per unique person shot — dedup key is the person filename)

**Why the dedup**: writing 804 copies of the same person shot would waste disk. Instead the manifest references shared files.

**Why 128 instead of 129 masks**: one person image failed mask generation (openpose returned no keypoints on a heavily-cropped shot) and was silently dropped in step_06. See "Failure modes" in [`MASKING.md`](MASKING.md).

### Naming convention: `<garment_type_short>_<sku_slug>_<hash>.jpg`

Example: `polo_ps2624_grn_4f52a11a.jpg`
- `polo` = garment type short name from SKU (POLO → polo)
- `ps2624_grn` = SKU code + color, lowercased, hyphens → underscores
- `4f52a11a` = 8-char hash from the original filename → guarantees uniqueness even if two SKUs collide

Slug generation is in [`retrain/step_05_upload.py`](retrain/step_05_upload.py) `_slug()`.

**The mask/densepose files use the same slug as the person file** (minus the extension → `.png`). So the mapping is:
```python
mask_path = f"mask/{person_slug}.png"        # same slug as person, .png extension
dp_path   = f"densepose/{person_slug}.png"
```

This is the **primary key that links a triplet to its mask + densepose files.** Never randomize slugs — the join breaks.

---

## The full pipeline: from raw catalog to manifest

Reference: `retrain/step_01_*.py` through `retrain/step_06_*.py`. Ordered execution:

### Step 01 — classify (`step_01_classify.py`)

Input: raw brand catalog with ad-hoc folder structure like `POLO/PS2624-00S-GRN/model_a.jpg`, `POLO/PS2624-00S-GRN/flatlay.jpg`, etc.

For each image, determine:
- Is this a **person shot** (someone wearing the garment) or a **flatlay** (garment on white)?
- What **garment_type** — `upper_body` (shirt/tee/polo/tank_top) or `lower_body` (jeans/pants/short/trouser)?

Uses simple heuristics: filename hints (`flatlay`, `product`), image content (skin detection, aspect ratio), folder name parsing. Outputs an index dict:

```python
index = {
  "POLO/PS2624-00S-GRN": {
    "garment_type": "upper_body",
    "flatlays": ["POLO/PS2624-00S-GRN/flatlay_1.jpg", ...],
    "persons": [
      {"path": "POLO/.../model_3_front.jpg", "kind": "person"},
      {"path": "POLO/.../model_3_side.jpg", "kind": "person"},
      ...
    ],
    "front_persons": [...],   # subset of persons that are frontal shots
  },
  ...
}
```

### Step 02 — face_cluster (`step_02_face_cluster.py`)

For every person shot, compute a face embedding (via `insightface` or similar). Cluster embeddings across all shots. Assign each shot a `model_id`:

```python
for p in info["front_persons"]:
    p["model_id"] = "model_3"    # or "unknown" if face not detected / cluster too small
```

**Critical for cross_outfit**: without `model_id`, we can't identify "same person wearing different outfit" pairs — cross_outfit would either be random pairing (wrong signal) or would collapse to self_recon (weak signal).

`model_id == "unknown"` shots are excluded from cross_outfit generation but can still contribute to self_recon.

### Step 03 — build_manifest (`step_03_build_manifest.py`)

Two functions: `build_self_recon` and `build_cross_outfit`.

**`build_self_recon`** — for each SKU with a flatlay and ≥1 front person, emit one triplet per (person, flatlay) combination:

```python
for p in info["front_persons"]:
    triplets.append({
        "person":       p["path"],
        "garment":      info["flatlay"],       # largest-file flatlay for that SKU
        "target":       p["path"],             # target = person (self_recon)
        "garment_type": info["garment_type"],
        "kind":         "self_recon",
        "sku_person":   sku,
        "sku_garment":  sku,
        "model_id":     p["model_id"],
    })
```

**`build_cross_outfit`** — group by `(garment_type, model_id)` so we only pair within one LoRA's scope + one person's identity. For every ordered pair `(a, b)` where `a != b` and `a["sku"] != b["sku"]`, apply the session-histogram gate. If it passes:

```python
triplets.append({
    "person":       a["person"],
    "garment":      b["flatlay"],
    "target":       b["person"],           # real photo of model in garment b
    "garment_type": gt,
    "kind":         "cross_outfit",
    "sku_person":   a["sku"],
    "sku_garment":  b["sku"],
    "model_id":     mid,
    "session_sim":  round(sim, 4),
})
```

Cap per (garment_type, model_id) via `MAX_CROSS_PAIRS_PER_MODEL_CATEGORY = 15` (or similar) so no one shoot dominates.

**Held-out split** (`build_self_recon(index, heldout_skus)`): a small number of SKUs are held out via SKU-based split (not triplet-based), so we can never leak "training person + eval garment" pairs. Held-out is written to `manifest_heldout.jsonl`.

Final manifest.jsonl entries have the `_` prefix added to metadata fields (`_kind`, `_sku_person`, etc.) so the trainer ignores them.

### Step 04 — contact_sheet (`step_04_contact_sheet.py`) — Gate 1

Render `CONTACT_SHEET_N_ROWS = 8` random triplets as image contact sheets. Save to `artifacts/contact_sheet_<i>.png`. Human eyeballs to confirm pairings are correct — no wrong-model matches, no bad SKU assignments, no obviously bad garments. Human then DROPS bad SKUs from the manifest before proceeding.

### Step 05 — upload (`step_05_upload.py`)

Copy the classified/renamed image files into the target folder structure:

```python
# Take raw file "POLO/PS2624-00S-GRN/model_3.jpg"
# Slug it → "polo_ps2624_grn_<hash>"
# Copy to <local_dataset>/person/polo_ps2624_grn_<hash>.jpg
# Copy same file to <local_dataset>/target/polo_ps2624_grn_<hash>.jpg (target dedup happens here)
# Copy flatlay to <local_dataset>/garment/polo_ps2624_grn_<hash>.jpg
```

Then `modal volume put brand_dataset_v2 leffa-weights:/brand_dataset_v2` uploads everything at once.

### Step 06 — precompute_masks (`step_06_precompute_masks.py`) — Gate 2

For every UNIQUE person shot in the manifest (deduped by person path), run the mask + densepose pipeline described in [`MASKING.md`](MASKING.md). Write `mask/*.png` and `densepose/*.png` to the volume. Update every manifest entry to include the new `mask` and `densepose` fields.

Also renders `artifacts/mask_previews.png` — Gate 2 human eyeball to confirm masks look sane.

---

## The heldout set — how we split

Held-out is at the **SKU level, not the triplet level**. Reasoning:

- If we split at triplet level: same model + same garment could appear in train AND eval → data leakage → eval scores are inflated.
- If we split at SKU level: entire SKU (all triplets involving it) goes to either train OR eval → clean split.

Held-out is **only `self_recon`** for those SKUs. Cross_outfit isn't held out because generating meaningful cross_outfit for held-out SKUs would require training-side models to be in the held-out set too, which defeats the purpose.

Currently: **13 held-out triplets**, all self_recon. Eval renders 5-column grids showing the ground-truth target next to base output and LoRA output.

---

## How the trainer uses this

From [`retrain/train/modal_train_v2.py`](retrain/train/modal_train_v2.py):

```python
trainer = LoRATrainer(app_config=config, train_config=train_cfg)
all_entries = list(trainer.dataset.entries)
# Filter to just one garment_type (train one LoRA per type)
trainer.dataset.entries = [e for e in all_entries if e.garment_type == garment_type]
# Verify every kept entry has mask + densepose (step_06 ran)
missing = [e for e in trainer.dataset.entries if not e.mask or not e.densepose]
if missing: raise RuntimeError(...)
```

- One LoRA per garment_type (currently 2: upper, lower).
- Filter is client-side (no re-manifesting needed) — just skip entries with wrong garment_type.
- Missing mask/densepose = hard failure — refuse to train on incomplete data.

Then each epoch iterates over the filtered entries with joint horizontal flip augmentation (see [`MASKING.md`](MASKING.md) for why flip is joint).

---

## Debug recipes

**Q: How many triplets per garment_type?**
```powershell
python -c "import json, collections; c=collections.Counter(); [c.update([(json.loads(l)['garment_type'], json.loads(l)['_kind'])]) for l in open('brand_dataset_v2/manifest.jsonl')]; print(c)"
```

**Q: A triplet is failing to load — why?**
1. Check the manifest entry: are all 5 paths (`person`, `garment`, `target`, `mask`, `densepose`) present?
2. Do the files actually exist on disk? `ls brand_dataset_v2/<path>` for each
3. Does the mask decode as an "L" mode PIL image with values ∈ {0, 255}? See [`MASKING.md`](MASKING.md) failure modes
4. Check trainer log for the specific error message — the `_safe_flip_getitem` wrapper in modal_train_v2 skips failing entries and prints a warning

**Q: Do the manifest and files agree?**
```powershell
python -c "
import json, os
missing = 0
for line in open('brand_dataset_v2/manifest.jsonl'):
    e = json.loads(line)
    for k in ('person', 'garment', 'target', 'mask', 'densepose'):
        p = os.path.join('brand_dataset_v2', e[k])
        if not os.path.exists(p):
            print(f'MISSING: {p}'); missing += 1
print(f'total missing: {missing}')
"
```

**Q: Cross_outfit pair quality — how do I audit?**
Each cross_outfit entry has `session_sim` field. Higher = closer histograms = more likely same shoot. Sort by session_sim ascending, eyeball the bottom 20:
```python
crosses = [json.loads(l) for l in open('manifest.jsonl') if json.loads(l)['_kind'] == 'cross_outfit']
crosses.sort(key=lambda e: e.get('session_sim', 1.0))
for e in crosses[:20]:
    print(e['session_sim'], e['person'], e['target'])
```
If low-sim pairs look wrong (different lighting/pose), raise the `SESSION_HIST_SIM_THRESHOLD` in step_03 and re-run.

---

## Extending: adding new SKUs

1. Drop new SKU folder into raw catalog (`data set/POLO/PE9999-00X-XYZ/`)
2. Re-run steps 01 → 06 in order. All steps are **idempotent** — existing entries won't be regenerated.
3. Verify new triplets appear in `manifest.jsonl` and their masks exist in `mask/`.
4. Retrain LoRA. Optional: warm-start from the previous checkpoint via `--resume` flag (not currently implemented but easy to add).

**Do NOT skip step 06 for new SKUs.** The trainer will crash on missing mask fields.

---

## Common misconceptions (things that trip people up)

1. **"target should always equal person"** — no, only for `self_recon`. For `cross_outfit`, target is a DIFFERENT photo of the SAME model wearing the GARMENT reference. This is what makes cross_outfit useful.

2. **"more triplets = better model"** — false. Adding more self_recon on already-included SKUs just increases memorization. What helps is: more UNIQUE SKUs, more model_ids, more session diversity.

3. **"mask is just a body silhouette"** — no, it's an AGNOSTIC mask: the region we want to replace. For upper_body it's roughly torso + arms (where the shirt goes). For lower_body it's legs + hip. The mask excludes the parts of the body that should stay fixed (face, hair, hands, feet).

4. **"densepose is optional"** — currently no, it's a required channel of the UNet input. Missing densepose = training crashes.

5. **"you can mix upper_body and lower_body in one LoRA"** — you technically can (v1 did this) but it dilutes the signal. Better to train one LoRA per garment_type (v2 approach). The manifest is one file; the trainer filters per-type at load time.

---

## File map

| Concern | File | Lines |
|---|---|---|
| Classify raw catalog | [`retrain/step_01_classify.py`](retrain/step_01_classify.py) | all |
| Face clustering → model_id | [`retrain/step_02_face_cluster.py`](retrain/step_02_face_cluster.py) | all |
| Build triplet manifest | [`retrain/step_03_build_manifest.py`](retrain/step_03_build_manifest.py) | all |
| `build_self_recon()` | step_03 | 154-203 |
| `build_cross_outfit()` + session gate | step_03 | 205-290 |
| Contact-sheet Gate 1 | [`retrain/step_04_contact_sheet.py`](retrain/step_04_contact_sheet.py) | all |
| Upload + slug convention | [`retrain/step_05_upload.py`](retrain/step_05_upload.py) | `_slug()` fn |
| Mask + densepose (Gate 2) | [`retrain/step_06_precompute_masks.py`](retrain/step_06_precompute_masks.py) | 190-268 |
| Manifest reader (dataset class) | `leffavton/training/dataset.py` (`TripletManifestEntry`) | fields must match manifest keys |
| Trainer filter by garment_type | [`retrain/train/modal_train_v2.py`](retrain/train/modal_train_v2.py) | 210-224 |

---

## History of triplet-building bugs (chronological)

1. **Random cross_outfit pairing** — early versions paired any two shots of the same model. Training data was full of "hey the background changed, and also the shirt" which taught the model to change everything. Fix: session histogram gate.
2. **Model_id collisions** — face clustering set `model_id = "unknown"` for shots where face was cropped/occluded. Those shots then couldn't be cross_outfit-paired at all → dropped signal. Currently accepted as loss; future fix could re-run face detection at higher recall.
3. **SKU-level held-out omitted** — first draft held out random triplets → data leakage across cross_outfit pairs. Fixed by moving to SKU-level split.
4. **Missing mask files** — one person shot failed openpose → no mask → training crashed at that entry. Fixed by adding `_safe_flip_getitem` wrapper that skips-with-log instead of crashing.
5. **Manifest field name drift** — early code used `kind` (no underscore), later versions used `_kind`. Trainer looked for `garment_type` at top level while some old code wrote `kind` field. Fixed by standardizing on underscore-prefix for metadata.

All five bugs share a theme: **the triplet is a contract — five paths + one garment_type + valid metadata — and everything downstream assumes the contract holds.** Break any part of it (missing field, wrong slug, unbuilt mask) and training either crashes or silently teaches the wrong thing.
