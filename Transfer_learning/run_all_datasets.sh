#!/bin/bash

# List of datasets
datasets=(
    ArrowHead
    ACSF1
    Adiac
)

# Directories (Update these paths according to your setup)
MODELS_DIRECTORY="models"
PRETRAINED_MODELS_DIRECTORY="/home/oumaima/Transfer_learning/LITE"
FEATURES_DIRECTORY="/home/oumaima/Transfer_learning/catch22_features"

# Function to check if models exist for a given target and source dataset pair
check_models_exist() {
    local target_dataset=$1
    local source_dataset=$2
    local model_dir="$MODELS_DIRECTORY/$target_dataset/$source_dataset"
    
    for i in {0..4}; do
        if [ ! -f "$model_dir/best_model_$i.hdf5" ]; then
            return 1
        fi
    done
    
    return 0
}

# Loop over each dataset pair
for target_dataset in "${datasets[@]}"; do
    for source_dataset in "${datasets[@]}"; do
        if [ "$target_dataset" != "$source_dataset" ]; then
            # Check if models already exist
            if check_models_exist "$target_dataset" "$source_dataset"; then
                echo "Models for target $target_dataset from source $source_dataset already exist. Skipping..."
                continue
            fi
            
            echo "Transfer learning from $source_dataset to $target_dataset"
            python3 main.py --models_directory "$MODELS_DIRECTORY" \
                            --pretrained_models_directory "$PRETRAINED_MODELS_DIRECTORY" \
                            --features_directory "$FEATURES_DIRECTORY" \
                            --target_dataset "$target_dataset" \
                            --source_dataset "$source_dataset"
        fi
    done
done

echo "Processing complete."
