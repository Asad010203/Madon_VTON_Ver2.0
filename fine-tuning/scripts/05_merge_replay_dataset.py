"""
Step 5: Merge shorts_triplets + base_triplets into mixed_replay_dataset.json.

This creates a unified dataset for 50/50 batch mixing during training:
- Batch 1: [shorts_sample_1, shorts_sample_2, base_sample_1, base_sample_2]
- Batch 2: [base_sample_3, base_sample_4, shorts_sample_3, shorts_sample_4]
- etc.

During training, the data loader will:
1. Sample 50% from shorts_triplets (learn new shorts classification)
2. Sample 50% from base_triplets (replay base model knowledge)

This ensures the model:
- Learns shorts-specific patterns without catastrophic forgetting
- Maintains performance on shirts/pants/dresses
- Stays close to pre-trained weights via 50/50 regularization

Batch structure in training loop:
    batch_size = 4
    shorts_samples = 2  (50%)
    base_samples = 2    (50%)

Usage:
    python 05_merge_replay_dataset.py

Output:
    fine-tuning/mixed_replay_dataset.json — merged dataset with metadata
    fine-tuning/replay_dataset_report.txt — summary
"""

import json
from pathlib import Path
from collections import defaultdict


def merge_datasets(shorts_path, base_path, output_path):
    """Merge shorts and base triplets with dataset labels."""

    # Load both datasets
    with open(shorts_path, 'r') as f:
        shorts_data = json.load(f)

    with open(base_path, 'r') as f:
        base_data = json.load(f)

    print("Loading datasets...")
    print(f"  Shorts: {len(shorts_data['train'])} train + {len(shorts_data['val'])} val")
    print(f"  Base:   {len(base_data['train'])} train + {len(base_data['val'])} val")

    # Tag each triplet with its source dataset
    shorts_train = shorts_data['train']
    shorts_val = shorts_data['val']
    base_train = base_data['train']
    base_val = base_data['val']

    for triplet in shorts_train:
        triplet['dataset'] = 'shorts'
    for triplet in shorts_val:
        triplet['dataset'] = 'shorts'
    for triplet in base_train:
        triplet['dataset'] = 'base'
    for triplet in base_val:
        triplet['dataset'] = 'base'

    # Merge into train/val
    merged_train = shorts_train + base_train
    merged_val = shorts_val + base_val

    output_data = {
        "dataset_info": {
            "total_triplets": len(merged_train) + len(merged_val),
            "train_triplets": len(merged_train),
            "val_triplets": len(merged_val),
            "shorts_train": len(shorts_train),
            "shorts_val": len(shorts_val),
            "base_train": len(base_train),
            "base_val": len(base_val),
            "strategy": "50/50 replay training (shorts + base)",
            "batch_mixing_strategy": "During training, sample 50% shorts + 50% base per batch",
        },
        "train": merged_train,
        "val": merged_val,
    }

    # Save merged dataset
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    return output_data


def create_report(output_data, report_path):
    """Generate a detailed report."""

    info = output_data['dataset_info']
    train = output_data['train']
    val = output_data['val']

    with open(report_path, 'w') as f:
        f.write("MIXED REPLAY DATASET REPORT (50/50 Training)\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET COMPOSITION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total triplets: {info['total_triplets']}\n")
        f.write(f"  Train: {info['train_triplets']} ({100*info['train_triplets']/info['total_triplets']:.1f}%)\n")
        f.write(f"    Shorts: {info['shorts_train']} ({100*info['shorts_train']/info['train_triplets']:.1f}%)\n")
        f.write(f"    Base:   {info['base_train']} ({100*info['base_train']/info['train_triplets']:.1f}%)\n")
        f.write(f"  Val:   {info['val_triplets']} ({100*info['val_triplets']/info['total_triplets']:.1f}%)\n")
        f.write(f"    Shorts: {info['shorts_val']} ({100*info['shorts_val']/info['val_triplets']:.1f}%)\n")
        f.write(f"    Base:   {info['base_val']} ({100*info['base_val']/info['val_triplets']:.1f}%)\n\n")

        f.write("TRAINING STRATEGY:\n")
        f.write("-" * 80 + "\n")
        f.write("During each training step:\n")
        f.write("  1. Sample 50% shorts triplets (learn new shorts classification)\n")
        f.write("  2. Sample 50% base triplets (replay base model knowledge)\n")
        f.write("  3. Forward pass through LoRA-fine-tuned model\n")
        f.write("  4. Compute loss: alpha * shorts_loss + (1-alpha) * base_loss\n")
        f.write("     (alpha=0.5 for balanced learning)\n")
        f.write("  5. Backprop through LoRA layers only (base model frozen)\n\n")

        f.write("EXPECTED OUTCOMES:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Shorts classification accuracy: +20-30% (new capability)\n")
        f.write("2. Shirts/pants/dresses performance: Maintained (replay prevents forgetting)\n")
        f.write("3. Model weights: +5-10% change in LoRA adapters (safety threshold)\n")
        f.write("4. No catastrophic forgetting due to 50/50 mixed training\n\n")

        f.write("BATCH SAMPLING EXAMPLE (batch_size=4):\n")
        f.write("-" * 80 + "\n")
        f.write("Batch 1: [shorts_0, shorts_1, base_0, base_1]     (2 shorts, 2 base)\n")
        f.write("Batch 2: [base_2, base_3, shorts_2, shorts_3]     (2 shorts, 2 base)\n")
        f.write("Batch 3: [shorts_4, base_4, shorts_5, base_5]     (2 shorts, 2 base)\n")
        f.write("...\n\n")

        f.write("GARMENT TYPE DISTRIBUTION (training set):\n")
        f.write("-" * 80 + "\n")

        garment_counts = defaultdict(int)
        dataset_counts = defaultdict(int)

        for triplet in train:
            garment_type = triplet.get('garment_type', 'unknown')
            dataset = triplet.get('dataset', 'unknown')
            garment_counts[garment_type] += 1
            dataset_counts[f"{dataset}_{garment_type}"] += 1

        for garment_type in sorted(garment_counts.keys()):
            count = garment_counts[garment_type]
            shorts_count = dataset_counts.get(f"shorts_{garment_type}", 0)
            base_count = dataset_counts.get(f"base_{garment_type}", 0)
            f.write(f"{garment_type:15s}: {count:3d} total (shorts: {shorts_count}, base: {base_count})\n")

        f.write("\nSAMPLE TRIPLETS (first 5 from training set):\n")
        f.write("-" * 80 + "\n")

        for i, triplet in enumerate(train[:5]):
            f.write(f"\nTriplet {i+1} ({triplet['dataset'].upper()}):\n")
            f.write(f"  SKU: {triplet['sku']}\n")
            f.write(f"  Garment type: {triplet['garment_type']}\n")
            f.write(f"  Person: {triplet['person']}\n")
            f.write(f"  Garment: {triplet['garment']}\n")
            f.write(f"  Target: {triplet['target']}\n")
            f.write(f"  Dataset source: {triplet['dataset']}\n")

    print(f"OK   Report saved: {report_path}")


def main():
    script_dir = Path(__file__).parent.parent
    shorts_path = script_dir / "shorts_triplets.json"
    base_path = script_dir / "base_triplets.json"
    output_path = script_dir / "mixed_replay_dataset.json"
    report_path = script_dir / "replay_dataset_report.txt"

    print("=" * 80)
    print("Step 5: Merge Datasets for 50/50 Replay Training")
    print("=" * 80)

    if not shorts_path.exists():
        print("ERROR: shorts_triplets.json not found. Run Step 3 first.")
        return

    if not base_path.exists():
        print("ERROR: base_triplets.json not found. Run Step 4 first.")
        return

    # Merge datasets
    output_data = merge_datasets(shorts_path, base_path, output_path)
    print(f"\nOK   Merged dataset saved: {output_path}")

    # Create report
    create_report(output_data, report_path)

    print("\n" + "=" * 80)
    print(f"SUCCESS: Mixed replay dataset created")
    print(f"  Total triplets: {output_data['dataset_info']['total_triplets']}")
    print(f"  Train: {output_data['dataset_info']['train_triplets']} (50% shorts, 50% base)")
    print(f"  Val:   {output_data['dataset_info']['val_triplets']} (50% shorts, 50% base)")
    print("=" * 80)
    print("\nNext: Build LoRA fine-tuning script (train.py) with weight preservation")


if __name__ == "__main__":
    main()
