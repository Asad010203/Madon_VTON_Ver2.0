"""
Add mask and pose field references to manifest.jsonl
"""

import json
from pathlib import Path

dataset_dir = Path(r"d:\modal.com env\IDM-VTON\fine-tuning\data set\artifacts_shorts\dataset")
manifest_path = dataset_dir / "manifest.jsonl"
manifest_heldout_path = dataset_dir / "manifest_heldout.jsonl"

def add_mask_pose_fields(manifest_path):
    """Add mask and pose references to each triplet."""
    triplets = []

    with open(manifest_path, 'r') as f:
        for line in f:
            triplet = json.loads(line)

            # Extract person slug from person path
            person_path = triplet['person']
            person_slug = person_path.split('/')[-1].replace('.jpg', '').replace('.png', '')

            # Add mask field
            triplet['mask'] = f"mask/{person_slug}.png"

            # Add pose field
            triplet['pose'] = f"pose/{person_slug}.png"

            triplets.append(triplet)

    # Write back
    with open(manifest_path, 'w') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet) + "\n")

    print(f"Updated {len(triplets)} triplets in {manifest_path}")
    return len(triplets)

def add_mask_pose_fields_heldout(manifest_heldout_path):
    """Add mask and pose references to held-out triplets."""
    triplets = []

    with open(manifest_heldout_path, 'r') as f:
        for line in f:
            triplet = json.loads(line)

            # Extract person slug from person path
            person_path = triplet['person']
            person_slug = person_path.split('/')[-1].replace('.jpg', '').replace('.png', '')

            # Add mask field
            triplet['mask'] = f"mask/{person_slug}.png"

            # Add pose field
            triplet['pose'] = f"pose/{person_slug}.png"

            triplets.append(triplet)

    # Write back
    with open(manifest_heldout_path, 'w') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet) + "\n")

    print(f"Updated {len(triplets)} triplets in {manifest_heldout_path}")
    return len(triplets)

if __name__ == "__main__":
    print("Adding mask and pose field references to manifest...")
    n_train = add_mask_pose_fields(manifest_path)
    n_heldout = add_mask_pose_fields_heldout(manifest_heldout_path)

    print(f"\nOK   Total updated: {n_train + n_heldout} triplets")

    # Verify
    with open(manifest_path, 'r') as f:
        sample = json.loads(f.readline())
        print(f"\nSample triplet:")
        print(json.dumps(sample, indent=2))
