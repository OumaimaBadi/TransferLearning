import os
import argparse
from Approaches.TransferLearning import TransferLearning

def main(models_directory, pretrained_models_directory, features_directory, target_dataset, source_dataset):
    # Initialize the TransferLearning class
    TL = TransferLearning(models_directory, pretrained_models_directory, features_directory)
    
    # Perform transfer learning from source dataset to target dataset
    print(f"Starting transfer learning from {source_dataset} to {target_dataset}")
    TL.training(source_dataset, target_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer learning for UCR datasets using Catch22 features.")
    parser.add_argument('--models_directory', type=str, required=True, help='Directory to save trained models')
    parser.add_argument('--pretrained_models_directory', type=str, required=True, help='Directory containing pretrained models')
    parser.add_argument('--features_directory', type=str, required=True, help='Directory containing Catch22 features')
    parser.add_argument('--target_dataset', type=str, required=True, help='Target dataset for transfer learning')
    parser.add_argument('--source_dataset', type=str, required=True, help='Source dataset for transfer learning')
    
    args = parser.parse_args()
    
    main(args.models_directory, args.pretrained_models_directory, args.features_directory, args.target_dataset, args.source_dataset)
