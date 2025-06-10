#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-train the CRNN model on a large, unlabeled (for our task) dataset.
This script implements the pre-training phase of our Transfer Learning plan.
It uses a self-supervised autoencoder approach to learn robust feature
representations from the large AudioSet cry subset.
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
from tqdm import tqdm

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class PretrainAudioDataset(Dataset):
    """Dataset for loading audio and returning it as both input and target."""
    def __init__(self, file_list, sr=16000, n_mels=128, duration=5):
        self.file_list = file_list
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.target_len = self.duration * self.sr
        # Calculate a fixed number of time steps that is divisible by 8 (for 3 max-pooling layers)
        self.target_time_steps = 160 

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        try:
            samples, _ = librosa.load(audio_path, sr=self.sr, mono=True)
            
            if len(samples) > self.target_len:
                samples = samples[:self.target_len]
            else:
                samples = np.pad(samples, (0, self.target_len - len(samples)), 'constant')

            mel_spec = librosa.feature.melspectrogram(y=samples, sr=self.sr, n_mels=self.n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Ensure fixed size for autoencoder input
            n_frames = log_mel_spec.shape[1]
            if n_frames > self.target_time_steps:
                log_mel_spec = log_mel_spec[:, :self.target_time_steps]
            else:
                pad_width = self.target_time_steps - n_frames
                log_mel_spec = np.pad(log_mel_spec, ((0,0), (0, pad_width)), mode='constant')

            log_mel_spec = np.expand_dims(log_mel_spec, axis=0)
            
            return torch.tensor(log_mel_spec, dtype=torch.float32)
        except Exception as e:
            print(f"\n[Warning] Error processing {audio_path}: {e}")
            return torch.zeros((1, self.n_mels, self.target_time_steps))

class CRNNAutoencoder(nn.Module):
    """An Autoencoder using the CRNN's feature extractor as the encoder."""
    def __init__(self):
        super(CRNNAutoencoder, self).__init__()
        # --- Encoder ---
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2)
        )
        # Input: (batch, 1, 128, 160) -> after CNN -> (batch, 128, 16, 20)
        self.encoder_rnn = nn.LSTM(input_size=128 * 16, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        
        # --- Decoder ---
        self.decoder_rnn = nn.LSTM(input_size=128 * 2, hidden_size=128 * 16, num_layers=2, batch_first=True)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        # Encode
        x = self.encoder_cnn(x)
        # Reshape for RNN
        # Input shape: (batch, 128, 16, 20)
        x = x.permute(0, 3, 1, 2) # -> (batch, 20, 128, 16)
        x = x.reshape(x.size(0), x.size(1), -1) # -> (batch, 20, 128 * 16)
        x, _ = self.encoder_rnn(x) # -> (batch, 20, 256)
        
        # Decode
        x, _ = self.decoder_rnn(x) # -> (batch, 20, 128 * 16)
        # Reshape for CNN
        x = x.reshape(x.size(0), x.size(1), 128, 16) # -> (batch, 20, 128, 16)
        x = x.permute(0, 3, 2, 1) # -> (batch, 16, 128, 20) - This was wrong!
        # Correct permute for ConvTranspose2d: (batch, channels, height, width)
        x = x.permute(0, 2, 1, 3) # -> (batch, 128, 16, 20)
        x = self.decoder_cnn(x) # -> (batch, 1, 128, 160)
        
        return x

def pretrain(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    all_files = [str(p) for p in data_dir.glob("**/*.wav")]
    print(f"Found {len(all_files)} audio files for pre-training.")
    
    if not all_files:
        print("No files found. Aborting.")
        return

    dataset = PretrainAudioDataset(all_files)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model = CRNNAutoencoder().to(device)
    loss_fct = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    print(f"🚀 Starting CRNN pre-training for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch_features in tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Pre-train]"):
            batch_features = batch_features.to(device)
            
            optimizer.zero_grad()
            reconstructed_features = model(batch_features)
            loss = loss_fct(reconstructed_features, batch_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Pre-training Loss={avg_train_loss:.4f}")
        
    # Save only the encoder part of the model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    # Create a state dict for only the encoder parts
    encoder_state_dict = {
        'cnn_state_dict': model.encoder_cnn.state_dict(),
        'rnn_state_dict': model.encoder_rnn.state_dict()
    }
    torch.save(encoder_state_dict, output_path)
    print(f"✅ Finished pre-training. Saved pre-trained encoder weights to '{output_path}'.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pre-train a CRNN model using an autoencoder approach.")
    parser.add_argument('--data_dir', type=str, default='external_data/audioset_cries_commercially_usable', help='Directory of the large, unlabeled dataset.')
    parser.add_argument('--output_path', type=str, default='crnn_pretrained_encoder.pth', help='Path to save the pre-trained encoder weights.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of pre-training epochs (should be small).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=12, help='DataLoader workers.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    pretrain(args)
