"""
Modal Inference: Generate Virtual Try-On Output with Trained LoRA

Usage:
    modal run modal_inference_shorts.py::inference_shorts_lora

This generates visual outputs showing how the trained model performs.
"""

import modal
from pathlib import Path

app = modal.App("idm-vton-shorts-inference")

datasets_volume = modal.Volume.from_name("idm-vton-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("idm-vton-checkpoints", create_if_missing=True)
outputs_volume = modal.Volume.from_name("idm-vton-outputs", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim().pip_install("torch", "torchvision", "Pillow", "numpy", "opencv-python"),
    gpu="A10G",
    volumes={"/data": datasets_volume, "/checkpoints": checkpoints_volume, "/outputs": outputs_volume},
    timeout=600,
)
def inference_shorts_lora():
    """Generate try-on outputs using trained LoRA model."""

    import json
    import torch
    import torch.nn as nn
    from PIL import Image
    import numpy as np
    from pathlib import Path
    import cv2

    print("=" * 80)
    print("IDM-VTON LoRA Shorts Inference - Virtual Try-On")
    print("=" * 80)

    # Load checkpoint
    checkpoint_path = Path("/checkpoints/shorts_lora_20260724_145101.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        for f in Path("/checkpoints").glob("*.pt"):
            print(f"  - {f.name}")
        return

    checkpoint = torch.load(checkpoint_path)
    print(f"✅ Loaded checkpoint: {checkpoint_path}")

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
    print(f"✅ Model loaded to device: {device}")

    # Load all triplets from dataset
    manifest_path = Path("/data/shorts_dataset/manifest.jsonl")
    triplets = []
    with open(manifest_path) as f:
        for line in f:
            triplets.append(json.loads(line))

    print(f"✅ Loaded {len(triplets)} triplets from dataset")

    # Load images function
    def load_image(path, size=(1024, 768)):
        """Load and resize image to (height, width)."""
        img = Image.open(path).convert('RGB')
        img = img.resize((size[1], size[0]), Image.LANCZOS)  # PIL resize takes (W, H)
        return np.array(img) / 255.0

    # Save outputs
    output_dir = Path("/outputs/shorts_inference")
    output_dir.mkdir(exist_ok=True)

    print(f"\n" + "=" * 80)
    print("Generating try-on outputs...")
    print("=" * 80)

    results = []

    # Run inference on all triplets
    for idx, triplet in enumerate(triplets, 1):
        print(f"\nSample {idx}/{len(triplets)}: {triplet['person'].split('/')[-1]} + {triplet['garment'].split('/')[-1]}")

        person_path = Path(f"/data/shorts_dataset/{triplet['person']}")
        garment_path = Path(f"/data/shorts_dataset/{triplet['garment']}")
        mask_path = Path(f"/data/shorts_dataset/{triplet['mask']}")

        if not person_path.exists() or not garment_path.exists():
            print(f"  ⚠️ Skipped (missing files)")
            continue

        person_img = load_image(person_path)
        garment_img = load_image(garment_path)
        mask_img = load_image(mask_path)

        # Stack: person (3) + garment (3) + mask (1) = 7 channels
        input_tensor = np.concatenate([
            person_img,
            garment_img,
            mask_img[:, :, :1]
        ], axis=2)

        input_batch = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0).float()
        input_batch = input_batch.to(device)

        # Inference
        with torch.no_grad():
            output = model(input_batch)

        output_img = output[0].cpu().permute(1, 2, 0).numpy()
        output_img = np.clip(output_img, 0, 1)
        output_img_uint8 = (output_img * 255).astype(np.uint8)

        # Save generated image
        output_path = output_dir / f"sample_{idx:02d}_generated.png"
        Image.fromarray(output_img_uint8).save(output_path)

        # Save comparison montage
        person_uint8 = (person_img * 255).astype(np.uint8)
        garment_uint8 = (garment_img * 255).astype(np.uint8)
        comparison = np.hstack([person_uint8, garment_uint8, output_img_uint8])
        comparison_path = output_dir / f"sample_{idx:02d}_comparison.png"
        Image.fromarray(comparison).save(comparison_path)

        print(f"  ✅ Generated | Output range: [{output_img.min():.4f}, {output_img.max():.4f}]")

        results.append({
            "idx": idx,
            "person": str(person_path),
            "garment": str(garment_path),
            "output": str(output_path),
            "comparison": str(comparison_path),
        })

    # Save summary
    summary_path = output_dir / "inference_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("IDM-VTON SHORTS INFERENCE RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path.name}\n")
        f.write(f"Training losses: {checkpoint['losses']}\n")
        f.write(f"  - Epoch 1: {checkpoint['losses'][0]:.4f}\n")
        f.write(f"  - Epoch 2: {checkpoint['losses'][1]:.4f}\n")
        f.write(f"  - Epoch 3: {checkpoint['losses'][2]:.4f}\n\n")
        f.write(f"Inference results:\n")
        f.write(f"  - Total samples: {len(results)}\n")
        f.write(f"  - Files generated: {len(results) * 2} images\n\n")
        f.write("Generated files:\n")
        for r in results:
            f.write(f"  Sample {r['idx']}:\n")
            f.write(f"    - sample_{r['idx']:02d}_generated.png (output)\n")
            f.write(f"    - sample_{r['idx']:02d}_comparison.png (person | garment | output)\n")

    print(f"\n" + "=" * 80)
    print("✅ INFERENCE COMPLETE!")
    print("=" * 80)
    print(f"\n📁 {len(results)} samples processed")
    print(f"📁 Output files saved to: /outputs/shorts_inference/")

    return {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "generated_image": str(output_path),
        "comparison_image": str(comparison_path),
    }


if __name__ == "__main__":
    inference_shorts_lora()
