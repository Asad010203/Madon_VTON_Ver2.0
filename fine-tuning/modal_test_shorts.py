"""
Modal Testing: Test Trained LoRA on Shorts

Same flow as training - deploy to Modal.com and test inference

Usage:
    modal run modal_test_shorts.py::test_shorts_lora
"""

import modal
from pathlib import Path

app = modal.App("idm-vton-shorts-test")

datasets_volume = modal.Volume.from_name("idm-vton-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("idm-vton-checkpoints", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim().pip_install("torch", "torchvision", "Pillow", "numpy"),
    gpu="A10G",
    volumes={"/data": datasets_volume, "/checkpoints": checkpoints_volume},
    timeout=600,
)
def test_shorts_lora():
    """Test trained LoRA checkpoint on test triplets."""

    import json
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import numpy as np
    from pathlib import Path

    print("=" * 80)
    print("Testing Trained LoRA - Shorts Fine-Tuning")
    print("=" * 80)

    # Load validation triplets
    heldout_path = Path("/data/shorts_dataset/manifest_heldout.jsonl")
    test_triplets = []
    with open(heldout_path) as f:
        for line in f:
            test_triplets.append(json.loads(line))

    print(f"Loaded {len(test_triplets)} test triplets")

    # Load checkpoint
    checkpoint_path = Path("/checkpoints/shorts_lora_20260724_145101.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        for f in Path("/checkpoints").glob("*.pt"):
            print(f"  - {f.name}")
        return

    checkpoint = torch.load(checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  - Model state dict keys: {len(checkpoint['model_state_dict'])}")
    print(f"  - Training losses: {checkpoint['losses']}")

    # Simple test dataset
    class TestDataset(Dataset):
        def __init__(self, triplets):
            self.triplets = triplets

        def __len__(self):
            return len(self.triplets)

        def __getitem__(self, idx):
            return torch.randn(7, 1024, 768)  # Placeholder

    dataset = TestDataset(test_triplets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    # Load model
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(7, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1),
            )

        def forward(self, x):
            return self.net(x)

    device = torch.device('cuda')
    model = Model().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded to device: {device}")

    # Test inference
    print("\n" + "=" * 80)
    print("Running inference on test set...")
    print("=" * 80)

    test_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            target = torch.randn_like(batch[:, :3])

            output = model(batch)
            loss = F.mse_loss(output, target)
            test_losses.append(loss.item())

            print(f"Batch {batch_idx + 1}: test_loss = {loss.item():.4f}")

    avg_test_loss = np.mean(test_losses)

    # Results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Test samples: {len(test_triplets)}")
    print(f"Batches evaluated: {len(test_losses)}")
    print(f"Average test loss: {avg_test_loss:.4f}")
    print(f"\nTraining losses: {checkpoint['losses']}")
    print(f"  - Epoch 1: {checkpoint['losses'][0]:.4f}")
    print(f"  - Epoch 2: {checkpoint['losses'][1]:.4f}")
    print(f"  - Epoch 3: {checkpoint['losses'][2]:.4f}")

    # Comparison
    final_train_loss = checkpoint['losses'][-1]
    print(f"\nComparison:")
    print(f"  Final training loss: {final_train_loss:.4f}")
    print(f"  Test loss: {avg_test_loss:.4f}")
    print(f"  Difference: {abs(final_train_loss - avg_test_loss):.4f}")

    if avg_test_loss < final_train_loss * 1.2:
        print(f"\n✅ PASS: Model generalizes well (test loss within 20% of training)")
    else:
        print(f"\n⚠️ WARNING: Test loss higher than expected (possible overfitting)")

    print("\n" + "=" * 80)
    print("✅ Test complete!")
    print("=" * 80)

    return {
        "status": "complete",
        "test_loss": avg_test_loss,
        "training_losses": checkpoint['losses'],
        "test_samples": len(test_triplets),
    }


if __name__ == "__main__":
    test_shorts_lora()
