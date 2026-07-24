"""
Step 1: Classify dataset images as PERSON or GARMENT using aspect ratio heuristic.

OpenPose would be ideal, but since it's not available locally, we use:
- Aspect ratio: person photos are typically portrait (H > W)
- File naming: files with UUIDs are likely persons, product codes are garments

Usage:
    python 01_classify_images.py

Output:
    fine-tuning/manifest.json
    fine-tuning/classification_report.txt
"""

import json
import sys
import os
from pathlib import Path
from PIL import Image

def classify_by_aspect_ratio(image_path):
    """
    Heuristic: portrait images (H > W, aspect > 1.2) are usually persons.
    Landscape/square are usually garments.
    """
    try:
        img = Image.open(image_path)
        w, h = img.size
        aspect_ratio = h / w

        if aspect_ratio > 1.15:
            return "PERSON", 0.85
        else:
            return "GARMENT", 0.8
    except Exception as e:
        print(f"[!] Could not read {image_path}: {e}")
        return None, 0.0


def classify_by_filename(filename):
    """
    Heuristic: UUIDs (36 char hex) or numbered files are usually persons.
    Product codes (MS2529, etc.) are garments.
    """
    name = filename.replace(".webp", "").replace(".jpg", "")

    # Check if filename looks like a UUID or number
    if "_" in name and len(name.split("_")[0]) <= 2:  # e.g., "4_uuid"
        return "PERSON", 0.9
    elif name.startswith("MS") or name.startswith("m"):  # Product codes
        return "GARMENT", 0.9
    else:
        return None, 0.5  # Uncertain, will use aspect ratio


def classify_image(image_path):
    """Classify image as PERSON or GARMENT."""
    filename = Path(image_path).name

    # Try filename heuristic first
    label, conf = classify_by_filename(filename)
    if label:
        return label, conf

    # Fall back to aspect ratio
    return classify_by_aspect_ratio(image_path)


def scan_dataset(dataset_dir):
    """Scan dataset and classify images."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_dir}")
        sys.exit(1)

    sku_folders = []
    for root, dirs, files in os.walk(dataset_path):
        webp_files = [f for f in files if f.endswith('.webp') or f.endswith('.jpg')]
        if webp_files:
            sku_folders.append((root, webp_files))

    manifest = {}
    warnings = []

    print(f"Found {len(sku_folders)} SKU folders with images.")
    print("Classifying images...\n")

    for folder_path, image_files in sku_folders:
        folder_name = Path(folder_path).name
        sku = folder_name.split('-')[0]

        # Get aspect ratios for all images to help classify
        image_aspects = []
        for img_file in image_files:
            img_path = Path(folder_path) / img_file
            try:
                img = Image.open(img_path)
                w, h = img.size
                aspect = h / w
                image_aspects.append((img_file, aspect))
            except:
                image_aspects.append((img_file, None))

        # If we have 2 images: portrait one is person, landscape is garment
        if len(image_files) == 2:
            aspects_valid = [(f, a) for f, a in image_aspects if a is not None]
            if len(aspects_valid) == 2:
                # Sort by aspect ratio
                aspects_valid.sort(key=lambda x: x[1])
                portrait_file = aspects_valid[1][0]  # Higher aspect = portrait = person
                landscape_file = aspects_valid[0][0]  # Lower aspect = landscape = garment

                manifest[folder_name] = {
                    "sku": sku,
                    "person": portrait_file,
                    "person_confidence": 0.95,
                    "garment": landscape_file,
                    "garment_confidence": 0.95,
                    "num_persons_detected": 1,
                    "num_garments_detected": 1,
                }

                print(f"OK {folder_name:30s} -> person: {portrait_file:45s} garment: {landscape_file}")
                continue

        # Fallback: use filename + aspect ratio heuristics
        classifications = []
        for img_file in image_files:
            img_path = Path(folder_path) / img_file
            label, confidence = classify_image(img_path)
            if label:
                classifications.append((img_file, label, confidence))

        persons = [(f, c) for f, l, c in classifications if l == "PERSON"]
        garments = [(f, c) for f, l, c in classifications if l == "GARMENT"]

        if len(persons) == 0:
            warnings.append(f"[!] {folder_name}: No person image detected")
            continue

        if len(garments) == 0:
            warnings.append(f"[!] {folder_name}: No garment image detected")
            continue

        if len(garments) > 1:
            warnings.append(f"[*] {folder_name}: Multiple garments ({len(garments)}), using first")

        if len(persons) > 1:
            warnings.append(f"[*] {folder_name}: Multiple persons ({len(persons)}), using highest confidence")
            persons.sort(key=lambda x: x[1], reverse=True)

        best_person = persons[0][0]
        best_garment = garments[0][0]

        manifest[folder_name] = {
            "sku": sku,
            "person": best_person,
            "person_confidence": round(float(persons[0][1]), 2),
            "garment": best_garment,
            "garment_confidence": round(float(garments[0][1]), 2),
            "num_persons_detected": len(persons),
            "num_garments_detected": len(garments),
        }

        print(f"OK {folder_name:30s} -> person: {best_person:45s} garment: {best_garment}")

    return manifest, warnings


def main():
    script_dir = Path(__file__).parent.parent
    dataset_dir = script_dir / "data set"
    manifest_path = script_dir / "manifest.json"
    report_path = script_dir / "classification_report.txt"

    print("="*80)
    print("Step 1: Image Classification (Person vs Garment)")
    print("="*80)
    print(f"Dataset: {dataset_dir}\n")

    manifest, warnings = scan_dataset(dataset_dir)

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nOK Manifest saved: {manifest_path}")

    # Save report
    with open(report_path, "w") as f:
        f.write("IMAGE CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total SKU folders: {len(manifest)}\n")
        f.write(f"Classification method: Filename + Aspect Ratio\n\n")

        if warnings:
            f.write("WARNINGS & NOTES:\n")
            f.write("-"*80 + "\n")
            for w in warnings:
                f.write(f"{w}\n")
            f.write("\n")

        f.write("MANIFEST SUMMARY:\n")
        f.write("-"*80 + "\n")
        for folder_name, info in sorted(manifest.items()):
            f.write(f"\n{folder_name}:\n")
            f.write(f"  SKU: {info['sku']}\n")
            f.write(f"  Person: {info['person']}\n")
            f.write(f"  Garment: {info['garment']}\n")

    print(f"OK Report saved: {report_path}")

    print("\n" + "="*80)
    print(f"SUCCESS: {len(manifest)} SKU folders classified")
    if warnings:
        print(f"Warnings: {len(warnings)}")
    print("="*80)


if __name__ == "__main__":
    main()
