#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the trained CRNN model.

This script loads the best saved CRNN model state and evaluates its performance
on the validation set, generating a classification report and a confusion matrix.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# We must reuse the exact same Dataset and Model classes from the training script
from train_crnn import AudioDataset, CRNN, set_seed

def evaluate_crnn_model(args):
    """Load the best CRNN model and evaluate it."""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare the dataset (re-creating the exact same validation split)
    data_dir = Path(args.data_dir)
    label_map = {
        'belly_pain': 0, 'burping': 1, 'cold_hot': 2, 'discomfort': 3,
        'hungry': 4, 'lonely': 5, 'scared': 6, 'tired': 7, 'unknown': 8
    }
    class_names = list(label_map.keys())
    
    all_files = [str(p) for p in data_dir.glob("**/*.wav")]
    all_labels = [label_map[Path(p).parent.name] for p in all_files]
    
    print(f"Loading {len(all_files)} samples from the test set for evaluation.")
    val_dataset = AudioDataset(all_files, all_labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Load the trained model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
        
    print(f"Loading model state from: {model_path}")
    model = CRNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 3. Run inference and collect predictions
    all_preds = []
    all_true_labels = []
    
    print("Running final evaluation...")
    with torch.no_grad():
        for batch_features, batch_labels in tqdm(val_loader, desc="Evaluating"):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            predictions = torch.argmax(outputs, dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_true_labels.extend(batch_labels.cpu().numpy())

    # 4. Generate and print the classification report
    print("\n" + "="*50)
    print("           Final CRNN Model Classification Report")
    print("="*50)
    report = classification_report(all_true_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    print("="*50)

    # 5. Generate and save the confusion matrix
    cm = confusion_matrix(all_true_labels, all_preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Final CRNN Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "final_crnn_confusion_matrix.png"
    plt.savefig(output_path)
    print(f"\nConfusion matrix saved to: {output_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained CRNN model.")
    parser.add_argument('--data_dir', type=str, default='Data/final_balanced_dataset', help='Directory of the dataset used for training.')
    parser.add_argument('--model_path', type=str, default='crnn_model_results/best_crnn_model.pth', help='Path to the saved .pth model file.')
    parser.add_argument('--output_dir', type=str, default='crnn_model_results', help='Directory to save evaluation results.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (must match training).')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_crnn_model(args)
