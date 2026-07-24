#!/bin/bash

# Run training and save output to file
OUTPUT_FILE="training_output.txt"

echo "Starting training..."
modal run modal_train_simple.py::train > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

echo "Training completed with exit code: $EXIT_CODE"
echo "Output saved to: $OUTPUT_FILE"

# Show last lines
echo ""
echo "=== Last 50 lines of output ==="
tail -50 "$OUTPUT_FILE"

# Check for checkpoint
echo ""
echo "=== Checking for checkpoint ==="
modal volume ls idm-vton-checkpoints / || echo "Checkpoint volume empty"

exit $EXIT_CODE
