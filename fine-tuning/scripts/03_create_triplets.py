"""
Step 3: Create training triplets (person, garment, target, mask).

Strategy:
  For each garment SKU:
    - Use the garment image from that SKU
    - Use the person from that SKU as the TARGET (expected output)
    - Create N triplets by pairing with OTHER shuffled persons (cross-outfit training)

  Result: ~95 triplets (19 SKUs × 5 shuffled persons per SKU)

  Triplet structure:
    {
      "person": "path/to/shuffled_person.webp",
      "garment": "path/to/garment.webp",
      "target": "path/to/target_wearing_garment.webp",
      "mask": "path/to/mask.png",
      "garment_type": "lower_body"
    }

Usage:
    python 03_create_triplets.py

Output:
    fine-tuning/shorts_triplets.json — full triplet dataset
    fine-tuning/triplet_dataset_report.txt — summary
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def create_triplets(manifest_path, masks_dir, dataset_base_dir, output_shuffles=5):
    """
    Create cross-outfit triplets for shorts training.

    For each SKU:
      - garment = garment image from that SKU
      - target = person from that SKU (wearing the garment)
      - person = shuffled person from OTHER SKUs (trying on the garment)
      - Create output_shuffles triplets per garment
    """

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Extract person and garment paths
    all_skus = sorted(manifest.keys())
    triplets = []

    print(f"Creating triplets: {len(all_skus)} SKUs × {output_shuffles} shuffles = ~{len(all_skus) * output_shuffles} triplets")
    print("=" * 80)

    for sku_idx, sku_folder in enumerate(all_skus):
        info = manifest[sku_folder]

        # This SKU's garment and person (target)
        garment_file = info["garment"]
        target_person_file = info["person"]

        garment_path = dataset_base_dir / sku_folder / garment_file
        target_person_path = dataset_base_dir / sku_folder / target_person_file
        mask_path = masks_dir / f"{sku_folder}.png"

        if not garment_path.exists() or not target_person_path.exists() or not mask_path.exists():
            print(f"SKIP {sku_folder}: missing files")
            continue

        # Get OTHER persons (for shuffling, exclude this SKU's person)
        other_persons = []
        for other_sku in all_skus:
            if other_sku != sku_folder:
                other_info = manifest[other_sku]
                other_person_file = other_info["person"]
                other_person_path = dataset_base_dir / other_sku / other_person_file
                if other_person_path.exists():
                    other_persons.append((other_sku, other_person_file, other_person_path))

        # Create triplets with shuffled persons
        if len(other_persons) == 0:
            print(f"SKIP {sku_folder}: no other persons available")
            continue

        # Sample output_shuffles random persons (with replacement if needed)
        sampled_persons = random.choices(other_persons, k=output_shuffles)

        for shuffle_idx, (source_sku, source_person_file, source_person_path) in enumerate(sampled_persons):
            triplet = {
                "sku": info["sku"],
                "sku_folder": sku_folder,
                "source_person_sku": source_sku,

                # Paths relative to dataset_base_dir
                "person": f"{source_sku}/{source_person_file}",
                "garment": f"{sku_folder}/{garment_file}",
                "target": f"{sku_folder}/{target_person_file}",
                "mask": f"../masks/{sku_folder}.png",

                "garment_type": "lower_body",
                "shuffle_index": shuffle_idx,
            }
            triplets.append(triplet)

        print(f"OK   {sku_folder:30s} -> {output_shuffles} triplets created")

    return triplets


def create_train_val_split(triplets, train_ratio=0.8, seed=42):
    """Split triplets into train/val sets (80/20)."""
    random.seed(seed)
    random.shuffle(triplets)

    split_point = int(len(triplets) * train_ratio)
    train = triplets[:split_point]
    val = triplets[split_point:]

    return train, val


def main():
    script_dir = Path(__file__).parent.parent
    manifest_path = script_dir / "manifest.json"
    masks_dir = script_dir / "masks"
    dataset_base_dir = script_dir / "data set" / "SHORT-20260717T113540Z-1-001" / "SHORT"

    triplets_path = script_dir / "shorts_triplets.json"
    report_path = script_dir / "triplet_dataset_report.txt"

    print("=" * 80)
    print("Step 3: Create Shorts Training Triplets (Cross-Outfit)")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Masks: {masks_dir}")
    print(f"Dataset: {dataset_base_dir}\n")

    if not manifest_path.exists():
        print("ERROR: Manifest not found. Run Steps 1-2 first.")
        return

    # Create triplets
    all_triplets = create_triplets(
        manifest_path,
        masks_dir,
        dataset_base_dir,
        output_shuffles=5
    )

    print(f"\nTotal triplets created: {len(all_triplets)}")

    # Split into train/val
    train_triplets, val_triplets = create_train_val_split(all_triplets, train_ratio=0.8)

    # Prepare output
    output_data = {
        "dataset_info": {
            "total_triplets": len(all_triplets),
            "train_triplets": len(train_triplets),
            "val_triplets": len(val_triplets),
            "garment_type": "lower_body",
            "strategy": "cross-outfit (shuffled persons, fixed garments)",
        },
        "train": train_triplets,
        "val": val_triplets,
    }

    # Save
    with open(triplets_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nOK   Triplets saved: {triplets_path}")

    # Report
    with open(report_path, 'w') as f:
        f.write("SHORTS TRIPLET DATASET REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total triplets: {len(all_triplets)}\n")
        f.write(f"Train split (80%): {len(train_triplets)}\n")
        f.write(f"Validation split (20%): {len(val_triplets)}\n")
        f.write(f"Garment type: lower_body\n")
        f.write(f"Strategy: Cross-outfit (shuffled persons × fixed garments)\n\n")

        f.write("DATASET BREAKDOWN:\n")
        f.write("-" * 80 + "\n")

        # Count by SKU
        sku_counts = defaultdict(int)
        for triplet in all_triplets:
            sku = triplet["sku"]
            sku_counts[sku] += 1

        for sku in sorted(sku_counts.keys()):
            f.write(f"SKU {sku}: {sku_counts[sku]} triplets\n")

        f.write("\nSAMPLE TRIPLETS (first 5 from train):\n")
        f.write("-" * 80 + "\n")
        for i, triplet in enumerate(train_triplets[:5]):
            f.write(f"\nTriplet {i+1}:\n")
            f.write(f"  SKU: {triplet['sku']}\n")
            f.write(f"  Source person (from): {triplet['source_person_sku']}\n")
            f.write(f"  Person: {triplet['person']}\n")
            f.write(f"  Garment: {triplet['garment']}\n")
            f.write(f"  Target (expected output): {triplet['target']}\n")
            f.write(f"  Mask: {triplet['mask']}\n")

    print(f"OK   Report saved: {report_path}")

    print("\n" + "=" * 80)
    print(f"SUCCESS: {len(all_triplets)} shorts triplets created")
    print(f"  Train: {len(train_triplets)} (80%)")
    print(f"  Val: {len(val_triplets)} (20%)")
    print("=" * 80)
    print("\nNext: Generate base triplets (shirts/pants) for 50/50 replay training")


if __name__ == "__main__":
    main()
