#!/bin/bash

# Upload proper shorts dataset to Modal

DATASET_DIR="d:\modal.com env\IDM-VTON\fine-tuning\data set\artifacts_shorts\dataset"
VOLUME="idm-vton-datasets"

echo "========================================================================"
echo "Uploading proper shorts dataset to Modal"
echo "========================================================================"

echo ""
echo "[1/2] Uploading manifest files..."
modal volume put "$VOLUME" "$DATASET_DIR/manifest.jsonl" /shorts_dataset/manifest.jsonl
modal volume put "$VOLUME" "$DATASET_DIR/manifest_heldout.jsonl" /shorts_dataset/manifest_heldout.jsonl

echo ""
echo "[2/2] Uploading image folders..."
# Upload entire dataset structure recursively
for folder in person garment target mask pose; do
    echo "  Uploading $folder/"
    for file in "$DATASET_DIR/$folder"/*; do
        filename=$(basename "$file")
        modal volume put "$VOLUME" "$file" "/shorts_dataset/$folder/$filename"
    done
done

echo ""
echo "========================================================================"
echo "Verifying upload..."
modal volume ls "$VOLUME" /shorts_dataset/

echo ""
echo "SUCCESS: Dataset uploaded to Modal!"
echo "========================================================================"
