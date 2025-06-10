#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a custom CRNN (Convolutional Recurrent Neural Network) model.

This script implements the second part of our Master Plan: building and
training a purpose-built CRNN model on our new, balanced, augmented dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import random
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class AudioDataset(Dataset):
    """Dataset for loading audio and converting to Mel spectrograms."""
    def __init__(self, file_list, label_list, sr=16000, n_mels=128):
        self.file_list = file_list
        self.label_list = label_list
        self.sr = sr
        self.n_mels = n_mels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.label_list[idx]
        try:
            samples, _ = librosa.load(audio_path, sr=self.sr, mono=True)
            
            # Pad or truncate to a fixed length (e.g., 5 seconds)
            target_len = 5 * self.sr
            if len(samples) > target_len:
                samples = samples[:target_len]
            else:
                samples = np.pad(samples, (0, target_len - len(samples)), 'constant')

            # Create Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=samples, sr=self.sr, n_mels=self.n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Add a channel dimension
            log_mel_spec = np.expand_dims(log_mel_spec, axis=0)
            
            return torch.tensor(log_mel_spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"\n[Warning] Error processing {audio_path}: {e}")
            return torch.zeros((1, self.n_mels, 251)), torch.tensor(0, dtype=torch.long) # Return dummy data

class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for Audio Classification."""
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # CNN part
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        # Get shape for RNN
        # Input: (batch, 1, 128, 251) -> after CNN -> (batch, 128, 16, 31)
        self.rnn_input_size = 128 * 16 # channels * features
        
        # RNN part
        self.lstm = nn.LSTM(input_size=self.rnn_input_size, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True)
        
        # Classifier part
        self.fc = nn.Linear(128 * 2, num_classes) # 128 * 2 for bidirectional

    def forward(self, x):
        # x shape: (batch, 1, n_mels, time_steps)
        x = self.conv(x)
        # x shape: (batch, channels, n_mels_reduced, time_steps_reduced)
        
        # Prepare for RNN
        x = x.permute(0, 3, 1, 2) # (batch, time, channels, n_mels)
        x = x.reshape(x.size(0), x.size(1), -1) # (batch, time, channels * n_mels)
        
        x, _ = self.lstm(x)
        
        # Use the output of the last time step
        x = x[:, -1, :]
        
        x = self.fc(x)
        return x

def train_crnn_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    label_map = {
        'belly_pain': 0, 'burping': 1, 'cold_hot': 2, 'discomfort': 3,
        'hungry': 4, 'lonely': 5, 'scared': 6, 'tired': 7, 'unknown': 8
    }
    class_names = list(label_map.keys())
    
    all_files = [str(p) for p in data_dir.glob("**/*.wav")]
    all_labels = [label_map[Path(p).parent.name] for p in all_files]
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=args.seed, stratify=all_labels
    )

    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)
    
    # Since the dataset is balanced, we don't need weighted loss or sampler
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = CRNN(num_classes=len(class_names)).to(device)
    loss_fct = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_val_acc = 0.0
    
    print(f"🚀 Starting CRNN model training for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]"):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = loss_fct(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]"):
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                predictions = torch.argmax(outputs, dim=-1)
                total_correct += (predictions == batch_labels).sum().item()
                total_samples += batch_labels.size(0)
                
        val_accuracy = (total_correct / total_samples) * 100
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_accuracy:.2f}%")
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), Path(args.output_dir) / "best_crnn_model.pth")
            print(f"  💾 Saved new best CRNN model with accuracy: {best_val_acc:.2f}%")
            
    print(f"✅ Finished CRNN training. Best validation accuracy: {best_val_acc:.2f}%")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a CRNN model for cry classification.")
    parser.add_argument('--data_dir', type=str, default='Data/final_balanced_dataset', help='Directory of the balanced, augmented dataset.')
    parser.add_argument('--output_dir', type=str, default='crnn_model_results', help='Directory to save the trained CRNN model.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    Path(args.output_dir).mkdir(exist_ok=True)
    train_crnn_model(args)
