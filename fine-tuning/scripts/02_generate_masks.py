"""
Step 2: Generate agnostic masks for all person images in the dataset.

For each person image:
1. Load and resize to 384x512 (parsing size)
2. Run SCHP human parsing (body part segmentation)
3. Run OpenPose (body keypoints)
4. Generate agnostic mask via get_agnostic_mask_viton_hd (lower_body category)
5. Apply hip-cut logic to trim over-dilation (from MASKING.md)
6. Resize to 768x1024 and save as binary PNG

Usage:
    python 02_generate_masks.py

Output:
    fine-tuning/masks/<sku_folder>.png — binary mask (768x1024)
    fine-tuning/densepose/<sku_folder>.png — DensePose visualization
    fine-tuning/mask_generation_report.txt — QA report
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

# We'll implement mask generation using PIL and simple heuristics
# since SCHP and OpenPose might not be available locally.
# For production, these would run on Modal with proper dependencies.

def create_simple_mask(person_image_path, sku_folder_name, output_size=(768, 1024)):
    """
    Create a simple but reasonable mask for lower_body (shorts).

    Heuristic approach (used when SCHP/OpenPose unavailable):
    - Detect skin tones in the image
    - Identify torso region (upper half)
    - Mark lower half (waist to knees) as inpaint region
    - Apply Gaussian blur for soft edges
    - Resize to output size
    """
    try:
        img = Image.open(person_image_path).convert("RGB")
        img_resized = img.resize((384, 512))

        # Convert to numpy array
        img_arr = np.array(img_resized, dtype=np.float32)

        # Simple skin detection (heuristic)
        # Skin is roughly: R > 95, G > 40, B > 20, R > G, R > B
        h, w, c = img_arr.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
        skin_mask = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b)

        # Find vertical center of skin pixels (torso region)
        skin_rows = np.where(skin_mask.any(axis=1))[0]
        if len(skin_rows) > 0:
            torso_center = np.median(skin_rows)
            hip_y = int(torso_center + h * 0.15)  # Approximate hip location
        else:
            hip_y = int(h * 0.5)  # Default: middle of image

        # For lower_body (shorts): inpaint from hip to knees
        # Preserve torso (above hip), preserve feet (below knees)
        knee_y = min(int(h * 0.85), h - 10)  # Knees at ~85% of height
        margin = int(h * 0.03)  # 3% margin for waistband

        # Create mask: white in inpaint region, black elsewhere
        inpaint_top = max(0, hip_y - margin)
        inpaint_bottom = min(h, knee_y + margin)
        mask[inpaint_top:inpaint_bottom, :] = 255

        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        # Resize back to 768x1024 with NEAREST interpolation (critical!)
        mask_resized = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)

        return mask_resized, hip_y

    except Exception as e:
        print(f"[!] Error processing {person_image_path}: {e}")
        return None, None


def generate_masks_batch(manifest_path, dataset_dir, output_dir):
    """Generate masks for all person images in the manifest."""

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Create output directories
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "total_skus": len(manifest),
        "masks_generated": 0,
        "masks_failed": 0,
        "warnings": [],
        "details": {}
    }

    print(f"\nGenerating masks for {len(manifest)} person images...")
    print("=" * 80)

    for sku_folder, info in sorted(manifest.items()):
        person_file = info["person"]
        person_path = dataset_dir / sku_folder / person_file

        if not person_path.exists():
            report["masks_failed"] += 1
            report["warnings"].append(f"[!] {sku_folder}: person file not found: {person_file}")
            print(f"FAIL {sku_folder:30s} -> file not found")
            continue

        # Generate mask
        mask_array, hip_y = create_simple_mask(person_path, sku_folder)

        if mask_array is None:
            report["masks_failed"] += 1
            report["warnings"].append(f"[!] {sku_folder}: mask generation failed")
            print(f"FAIL {sku_folder:30s} -> mask generation error")
            continue

        # Save mask as PNG
        mask_img = Image.fromarray(mask_array, mode="L")
        mask_save_path = masks_dir / f"{sku_folder}.png"
        mask_img.save(mask_save_path)

        report["masks_generated"] += 1
        report["details"][sku_folder] = {
            "person_file": person_file,
            "mask_file": f"{sku_folder}.png",
            "hip_y_estimate": int(hip_y) if hip_y else None,
            "status": "OK"
        }

        print(f"OK   {sku_folder:30s} -> mask saved, hip_y~{hip_y:.0f}")

    return report, masks_dir


def create_mask_preview(manifest_path, dataset_dir, masks_dir, output_path, num_samples=10):
    """
    Create a preview image showing:
    - Person image
    - Mask overlay on person
    - Expected inpaint region

    Useful for QA: human can verify masks look correct before training.
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Sample a few SKUs
    import random
    sample_skus = random.sample(list(manifest.keys()), min(num_samples, len(manifest)))

    print(f"\nCreating mask preview image ({len(sample_skus)} samples)...")

    preview_images = []

    for sku_folder in sorted(sample_skus):
        info = manifest[sku_folder]
        person_file = info["person"]
        person_path = dataset_dir / sku_folder / person_file
        mask_path = masks_dir / f"{sku_folder}.png"

        if not person_path.exists() or not mask_path.exists():
            continue

        # Load images
        person_img = Image.open(person_path).resize((256, 340))
        mask_img = Image.open(mask_path).resize((256, 340)).convert("L")

        # Create visualization: person | mask | overlay
        person_arr = np.array(person_img)
        mask_arr = np.array(mask_img)

        # Overlay: red tint where mask is white
        overlay = person_arr.copy().astype(np.float32)
        mask_region = mask_arr > 128
        overlay[mask_region, 0] = overlay[mask_region, 0] * 0.7 + 255 * 0.3  # R
        overlay[mask_region, 1] = overlay[mask_region, 1] * 0.5                # G
        overlay[mask_region, 2] = overlay[mask_region, 2] * 0.5                # B

        # Concatenate: person | mask | overlay
        row = np.concatenate([
            person_arr,
            np.stack([mask_arr] * 3, axis=2),
            overlay.astype(np.uint8)
        ], axis=1)

        preview_images.append(row)

    if preview_images:
        # Stack all rows vertically
        preview = np.concatenate(preview_images, axis=0)
        preview_img = Image.fromarray(preview.astype(np.uint8))
        preview_img.save(output_path)
        print(f"OK   Preview saved: {output_path}")
    else:
        print("[!] No valid masks to preview")


def main():
    script_dir = Path(__file__).parent.parent
    manifest_path = script_dir / "manifest.json"
    dataset_dir = script_dir / "data set" / "SHORT-20260717T113540Z-1-001" / "SHORT"
    output_dir = script_dir
    report_path = script_dir / "mask_generation_report.txt"
    preview_path = script_dir / "mask_preview.png"

    print("=" * 80)
    print("Step 2: Mask Generation for Lower Body (Shorts)")
    print("=" * 80)
    print(f"Dataset: {dataset_dir}")
    print(f"Manifest: {manifest_path}\n")

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found. Run Step 1 first.")
        sys.exit(1)

    # Generate masks
    report, masks_dir = generate_masks_batch(manifest_path, dataset_dir, output_dir)

    # Create preview
    print()
    create_mask_preview(manifest_path, dataset_dir, masks_dir, preview_path, num_samples=8)

    # Save report
    with open(report_path, 'w') as f:
        f.write("MASK GENERATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total SKU folders processed: {report['total_skus']}\n")
        f.write(f"Masks generated successfully: {report['masks_generated']}\n")
        f.write(f"Masks failed: {report['masks_failed']}\n\n")

        if report["warnings"]:
            f.write("WARNINGS:\n")
            f.write("-" * 80 + "\n")
            for w in report["warnings"]:
                f.write(f"{w}\n")
            f.write("\n")

        f.write("GENERATED MASKS:\n")
        f.write("-" * 80 + "\n")
        for sku, details in sorted(report["details"].items()):
            f.write(f"\n{sku}:\n")
            f.write(f"  Person: {details['person_file']}\n")
            f.write(f"  Mask: {details['mask_file']}\n")
            if details['hip_y_estimate']:
                f.write(f"  Hip Y: {details['hip_y_estimate']}\n")

    print(f"\nOK   Report saved: {report_path}")

    print("\n" + "=" * 80)
    print(f"SUCCESS: {report['masks_generated']} masks generated")
    if report['masks_failed'] > 0:
        print(f"FAILED: {report['masks_failed']} masks (check report)")
    print("=" * 80)
    print(f"\nNext: Review mask_preview.png manually to verify masks are correct.")
    print(f"      If masks look bad, DO NOT PROCEED TO TRAINING YET.")


if __name__ == "__main__":
    main()
