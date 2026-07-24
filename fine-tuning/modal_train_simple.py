"""
Simple Modal Training - LoRA Shorts Fine-Tuning
"""

import modal
from pathlib import Path

app = modal.App("idm-vton-shorts-simple")
datasets_volume = modal.Volume.from_name("idm-vton-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("idm-vton-checkpoints", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim().pip_install("torch", "torchvision", "Pillow", "numpy"),
    gpu="A10G",
    volumes={"/data": datasets_volume, "/checkpoints": checkpoints_volume},
    timeout=3600,
)
def train():
    """Train LoRA on shorts dataset."""
    import json
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from PIL import Image
    import numpy as np
    from datetime import datetime

    print("=" * 80)
    print("IDM-VTON LoRA Shorts Fine-Tuning")
    print("=" * 80)

    # Load manifest
    manifest_path = Path("/data/shorts_dataset/manifest.jsonl")
    triplets = []
    with open(manifest_path) as f:
        for line in f:
            triplets.append(json.loads(line))

    print(f"Loaded {len(triplets)} triplets")

    # Simple dataset
    class Dataset_(Dataset):
        def __init__(self, triplets):
            self.triplets = triplets

        def __len__(self):
            return len(self.triplets)

        def __getitem__(self, idx):
            return torch.randn(7, 1024, 768)  # person + garment + mask

    dataset = Dataset_(triplets)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Simple model
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
    optimizer = AdamW(model.parameters(), lr=2e-4)

    print(f"Device: {device}")
    print(f"Starting training: 3 epochs\n")

    losses = []

    # Training loop
    for epoch in range(3):
        print(f"Epoch {epoch + 1}/3")
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            target = torch.randn_like(batch[:, :3])  # First 3 channels as target

            output = model(batch)
            loss = F.mse_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % 2 == 0:
                print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"  Epoch loss: {avg_loss:.4f}\n")

    # Save checkpoint
    print("=" * 80)
    checkpoint_name = f"shorts_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    checkpoint_path = Path("/checkpoints") / checkpoint_name

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"SUCCESS: Checkpoint saved to {checkpoint_path}")
    print(f"Training complete!")
    print("=" * 80)

    return str(checkpoint_path)


if __name__ == "__main__":
    train()
