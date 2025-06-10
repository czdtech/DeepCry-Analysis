#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the custom CRNN model on a JAMS-annotated test set.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import jams
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa

# Import the CRNN model definition from our training script
from train_crnn import CRNN, set_seed

class JamsCRNNDataset(Dataset):
    """A dataset for loading JAMS-annotated audio for our CRNN model."""
    def __init__(self, data_dir, label_map, sr=16000, n_mels=128):
        self.data_dir = Path(data_dir)
        self.label_map = label_map
        self.sr = sr
        self.n_mels = n_mels
        self.audio_files = sorted(list(self.data_dir.glob("*.wav")))
        print(f"Found {len(self.audio_files)} audio files in {data_dir}.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        jams_path = audio_path.with_suffix(".jams")
        
        true_label_str = "unknown"
        try:
            jam = jams.load(str(jams_path))
            for event in jam.annotations['scaper'][0]['data']:
                if event.value['role'] == 'foreground':
                    true_label_str = event.value['label']
                    break
            
            true_label_id = self.label_map.get(true_label_str, self.label_map['unknown'])

            samples, _ = librosa.load(audio_path, sr=self.sr, mono=True)
            target_len = 5 * self.sr
            if len(samples) > target_len:
                samples = samples[:target_len]
            else:
                samples = np.pad(samples, (0, target_len - len(samples)), 'constant')

            mel_spec = librosa.feature.melspectrogram(y=samples, sr=self.sr, n_mels=self.n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            log_mel_spec = np.expand_dims(log_mel_spec, axis=0)
            
            return torch.tensor(log_mel_spec, dtype=torch.float32), torch.tensor(true_label_id, dtype=torch.long)
        except Exception as e:
            print(f"\n[Warning] Error processing {audio_path} or {jams_path}: {e}")
            return torch.zeros((1, self.n_mels, 251)), torch.tensor(self.label_map['unknown'], dtype=torch.long)

def evaluate(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    label_map = {
        'belly_pain': 0, 'burping': 1, 'cold_hot': 2, 'discomfort': 3,
        'hungry': 4, 'lonely': 5, 'scared': 6, 'tired': 7, 'unknown': 8
    }
    class_names = list(label_map.keys())

    # --- Load Model ---
    model = CRNN(num_classes=len(class_names)).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model state from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()

    # --- Dataset and DataLoader ---
    test_dataset = JamsCRNNDataset(args.data_dir, label_map)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Evaluation Loop ---
    all_preds = []
    all_true_labels = []
    print("Running final evaluation...")
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features = features.to(device)
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.numpy())

    # --- Reporting ---
    print("\n" + "="*50)
    print("           Final CRNN Model Classification Report")
    print("="*50)
    report = classification_report(all_true_labels, all_preds, labels=list(label_map.values()), target_names=class_names, digits=4, zero_division=0)
    print(report)
    print("="*50)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a CRNN model on a JAMS-annotated dataset.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of the JAMS-annotated test data.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pth model file.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args)
