#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune a pre-trained CRNN model on a small, labeled dataset.

This script implements the fine-tuning phase of our Transfer Learning plan.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import random

# Reuse the model and dataset classes from our original train_crnn script
from train_crnn import CRNN, AudioDataset, set_seed

def load_pretrained_encoder(model, pretrained_path):
    """Loads pre-trained weights into the model's encoder."""
    try:
        print(f"Loading pre-trained encoder weights from: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path)
        
        # Corrected attribute names to 'conv' and 'lstm'
        model.conv.load_state_dict(pretrained_dict['cnn_state_dict'])
        model.lstm.load_state_dict(pretrained_dict['rnn_state_dict'])
        
        print("✅ Successfully loaded pre-trained weights into CNN and RNN layers.")
    except FileNotFoundError:
        print(f"Warning: Pre-trained weights file not found at '{pretrained_path}'. Training from scratch.")
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}. Training from scratch.")

def finetune(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Dataset and Dataloaders ---
    data_dir = Path(args.data_dir)
    label_map = {
        'belly_pain': 0, 'burping': 1, 'cold_hot': 2, 'discomfort': 3,
        'hungry': 4, 'lonely': 5, 'scared': 6, 'tired': 7, 'unknown': 8
    }
    class_names = list(label_map.keys())

    all_files = [str(p) for p in data_dir.glob("**/*.wav")]
    all_labels = [label_map[p.parent.name] for p in Path(data_dir).glob("**/*.wav")]

    if not all_files:
        print(f"No audio files found in {data_dir}. Aborting.")
        return

    # Split data into training and validation sets, stratifying to handle imbalance
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=args.seed, stratify=all_labels
    )

    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Found {len(all_files)} total samples for fine-tuning.")
    print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")

    # --- 2. Model, Loss, Optimizer ---
    model = CRNN(num_classes=len(class_names)).to(device)
    
    load_pretrained_encoder(model, args.pretrained_path)
    
    loss_fct = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # --- 3. Fine-tuning Loop ---
    print(f"🚀 Starting CRNN fine-tuning for {args.num_epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Fine-tune]"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fct(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(args.output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), output_path)
            print(f"✨ New best model saved to '{output_path}' with validation accuracy: {best_val_acc:.4f}")

    print(f"✅ Finished fine-tuning. Best model saved to '{args.output_path}'.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained CRNN model.")
    parser.add_argument('--data_dir', type=str, default='Data/finetune_data', help='Directory of the fine-tuning dataset.')
    parser.add_argument('--pretrained_path', type=str, default='crnn_pretrained_encoder.pth', help='Path to the pre-trained encoder weights.')
    parser.add_argument('--output_path', type=str, default='crnn_model_results/finetuned_crnn_model.pth', help='Path to save the fine-tuned model.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Fine-tuning learning rate (should be smaller).')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of fine-tuning epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=12, help='DataLoader workers.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    finetune(args)
