# IDM-VTON LoRA Fine-Tuning on Modal.com

## Overview
This guide deploys the prepared shorts fine-tuning dataset to Modal.com and runs LoRA training in your **fitcheckml** workspace.

**Status:** Dataset preparation complete
- ✅ 95 shorts triplets (76 train, 19 val)
- ✅ 95 base triplets (76 train, 19 val) for 50/50 replay
- ✅ 190 total mixed triplets ready for training
- ✅ Modal training script ready

**Next:** Upload dataset & run training on Modal GPU

---

## Prerequisites

1. **Modal Account & CLI**
   ```bash
   pip install modal
   modal token new  # Authenticate with Modal
   ```

2. **fitcheckml Workspace**
   - Ensure base IDM-VTON model is available in fitcheckml
   - Workspace location: https://modal.com/workspaces/fitcheckml

3. **Modal Volumes** (create if missing)
   ```bash
   modal volume create idm-vton-datasets
   modal volume create idm-vton-checkpoints
   ```

---

## Step 1: Prepare Dataset for Upload

Copy prepared datasets to a local directory:

```bash
# From your fine-tuning folder:
cd "d:\modal.com env\IDM-VTON\fine-tuning"

# Create upload directory
mkdir -p upload_to_modal

# Copy datasets
cp mixed_replay_dataset.json upload_to_modal/
cp shorts_triplets.json upload_to_modal/
cp base_triplets.json upload_to_modal/

# Copy sample triplet report (for reference)
cp triplet_dataset_report.txt upload_to_modal/
cp base_triplet_dataset_report.txt upload_to_modal/
cp replay_dataset_report.txt upload_to_modal/
```

---

## Step 2: Upload Datasets to Modal Volume

Upload the datasets to Modal's persistent storage:

```bash
# Upload to idm-vton-datasets volume
modal volume put idm-vton-datasets upload_to_modal/mixed_replay_dataset.json /mixed_replay_dataset.json
modal volume put idm-vton-datasets upload_to_modal/shorts_triplets.json /shorts_triplets.json
modal volume put idm-vton-datasets upload_to_modal/base_triplets.json /base_triplets.json

# Verify upload
modal volume ls idm-vton-datasets /
```

**Expected output:**
```
mixed_replay_dataset.json
shorts_triplets.json
base_triplets.json
```

---

## Step 3: Deploy Training Script to Modal

Run the training on Modal GPU infrastructure:

```bash
# From the fine-tuning directory
cd "d:\modal.com env\IDM-VTON\fine-tuning"

# Deploy and run training
modal run modal_train.py::train_lora_shorts
```

**What happens:**
1. Modal allocates an A10G GPU (or specified GPU)
2. Loads base IDM-VTON model from fitcheckml
3. Mounts idm-vton-datasets volume with prepared triplets
4. Trains LoRA adapters for 3 epochs with:
   - 50/50 batch mixing (shorts + base)
   - All 10 weight preservation methods
   - Validation after each epoch
5. Saves checkpoint to idm-vton-checkpoints volume
6. Generates training report

---

## Step 4: Monitor Training

While training runs, you can:

```bash
# Check volume status
modal volume ls idm-vton-checkpoints /

# View training logs
modal logs modal_train.py::train_lora_shorts

# After training completes, download checkpoint
modal volume get idm-vton-checkpoints /lora_shorts_YYYYMMDD_HHMMSS.pt ./checkpoint.pt
```

---

## Step 5: Verify Training Results

After training completes, check the report:

```bash
# Download training report
modal volume get idm-vton-checkpoints /training_report.txt ./training_report.txt

# View report
cat training_report.txt
```

**Expected report shows:**
- Training loss decreasing over 3 epochs
- Validation loss < training loss (no overfitting)
- Best validation loss recorded
- Checkpoint saved to volume

---

## Step 6: Deploy Trained Model to Production

After verification, integrate the LoRA checkpoint into production:

### Option A: Load from Modal Volume (Recommended)

```python
# In your production code
import torch
from pathlib import Path
from modal import Volume

# Mount the checkpoints volume
checkpoints_volume = Volume.from_name("idm-vton-checkpoints")

# Load LoRA checkpoint
checkpoint_path = "/path/to/lora_shorts_YYYYMMDD_HHMMSS.pt"
checkpoint = torch.load(checkpoint_path)

# Load into your model
model.load_lora_adapters(checkpoint['lora_adapters'])
```

### Option B: Export to Hugging Face

```bash
# Download checkpoint
modal volume get idm-vton-checkpoints /lora_shorts_*.pt ./checkpoint.pt

# Convert to HF format and upload
# (See production deployment guide)
```

---

## Weight Preservation Verification

After training, verify that existing capabilities (shirts/pants) are preserved:

```python
# Validation check (in your test suite)
def test_weight_preservation():
    """Verify shorts accuracy improved without hurting shirts/pants."""
    
    # Load trained LoRA checkpoint
    checkpoint = torch.load("checkpoint.pt")
    lora_adapters = checkpoint['lora_adapters']
    
    # Test results should show:
    # - Shorts accuracy: +20-30% improvement
    # - Shirts/pants/dresses: Maintained or slightly improved
    # - Total weight change: <10% (LoRA isolation + 50/50 replay)
    
    shorts_accuracy_delta = metric_shorts_new - metric_shorts_base
    shirts_accuracy_delta = metric_shirts_new - metric_shirts_base
    
    assert shorts_accuracy_delta > 0.15, "Shorts should improve by 15%+"
    assert shirts_accuracy_delta >= -0.05, "Shirts should not degrade by >5%"
```

---

## Troubleshooting

### Issue: Volume not found
```bash
# Create volumes if missing
modal volume create idm-vton-datasets
modal volume create idm-vton-checkpoints
```

### Issue: Out of memory
```bash
# Reduce batch size in modal_train.py
config['batch_size'] = 2  # From 4 to 2
```

### Issue: Base model not loading
```bash
# Ensure fitcheckml has the model and update modal_train.py to load from correct path
# Example: Load from HuggingFace if fitcheckml path not available
```

### Issue: Dataset not found in training
```bash
# Verify upload to volume:
modal volume ls idm-vton-datasets /

# Re-upload if missing:
modal volume put idm-vton-datasets mixed_replay_dataset.json /mixed_replay_dataset.json
```

---

## Dataset Summary

**Mixed Replay Dataset (for 50/50 training):**
- Total triplets: 190 (152 train, 38 val)
- Shorts: 95 triplets (76 train, 19 val)
  - 19 SKUs × 5 shuffles = 95
  - Cross-outfit training (shuffled persons)
  - Prevents overfitting to specific person-garment pairs
- Base (replay): 95 triplets (76 train, 19 val)
  - Shirts/pants/dresses for knowledge replay
  - Same persons (no distribution shift)
  - Prevents catastrophic forgetting

**Training Strategy:**
- Batch size: 4 (2 shorts, 2 base)
- Each batch: 50% shorts + 50% base
- This ensures model learns shorts while replaying base knowledge
- 3 epochs = ~114 gradient updates per epoch

**Weight Preservation Methods:**
1. **LoRA isolation** — Only LoRA layers trained
2. **50/50 replay** — Balanced batch mixing
3. **EMA regularization** — Weight drift monitoring
4. **Weight decay (L2)** — Regularization on LoRA
5. **Low LR** — Conservative 2e-4
6. **Dropout** — 5% in LoRA layers
7. **Gradient clipping** — Norm 1.0
8. **Validation monitoring** — Early stopping criteria
9. **Checkpointing** — Save best model
10. **Task-specific adapters** — Extensible for future tasks

---

## Next Steps

1. ✅ Dataset preparation complete
2. → **Upload dataset to Modal (Step 2 above)**
3. → **Run training on Modal (Step 3 above)**
4. → **Verify results & deploy to production**

---

## Support

For issues with Modal deployment:
- Modal docs: https://modal.com/docs
- Modal community: https://modal.com/community
- fitcheckml workspace: Check with your Modal admin

For IDM-VTON fine-tuning issues:
- Review dataset report: `replay_dataset_report.txt`
- Check training logs during execution
- Verify base model is loaded correctly
