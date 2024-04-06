#!/bin/bash

# Get the directory of the current script (textcaps.sh)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Paths to Python scripts relative to the current script's location
DATASET_SCRIPT="${SCRIPT_DIR}/textcaps_dataset.py"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/textcaps_preprocess.py"

# Activate your Python environment if necessary
# For example, if using Conda, you might have something like:
# source activate myenv

# Run the dataset script
echo "Running dataset script..."
python3 "${DATASET_SCRIPT}"

# Check if the dataset script ran successfully
if [ $? -eq 0 ]; then
    echo "Dataset script completed successfully. Running preprocess script..."
    # Run the preprocess script
    python3 "${PREPROCESS_SCRIPT}"

    # Check if the preprocess script ran successfully
    if [ $? -eq 0 ]; then
        echo "Preprocess script completed successfully."
    else
        echo "Preprocess script encountered an error."
    fi
else
    echo "Dataset script encountered an error."
fi
