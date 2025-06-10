#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Data Preparation Script.

This script implements the data strategy from our OPTIMIZATION_PLAN.md:
1. Unifies data from all specified source directories.
2. Performs targeted offline augmentation to create a large, balanced dataset.
3. Uses a rich set of augmentations (noise, pitch, time stretch) for robustness.
"""

import argparse
from pathlib import Path
import glob
import os
import shutil
from tqdm import tqdm
import librosa
import soundfile as sf
from audiomentations import Compose, AddBackgroundNoise, PitchShift, TimeStretch
import jams
import random
from collections import defaultdict

def prepare_dataset(args):
    """
    Generates the final, balanced, augmented dataset.
    """
    source_dirs = [Path(d) for d in args.source_dirs]
    noise_dir = Path(args.noise_dir)
    output_dir = Path(args.output_dir)
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Warning: Output directory '{output_dir}' is not empty. Files may be overwritten.")
    output_dir.mkdir(exist_ok=True)

    # --- 1. Unify Data from all sources ---
    print("Step 1: Unifying data from all sources...")
    
    label_map = {
        'belly_pain': 0, 'burping': 1, 'cold_hot': 2, 'discomfort': 3,
        'hungry': 4, 'lonely': 5, 'scared': 6, 'tired': 7, 'unknown': 8
    }
    class_names = list(label_map.keys())
    
    data_pool = defaultdict(list)

    # Define mappings for special directories
    mendeley_label_map = {
        'Hungry': 'hungry',
        'Tired': 'tired',
        'Uncomfortable': 'discomfort'
    }

    for source_dir in source_dirs:
        print(f"Processing source directory: {source_dir}")
        if not source_dir.is_dir():
            print(f"Warning: Source directory '{source_dir}' not found. Skipping.")
            continue

        # Handle Mendeley dataset with its specific folder names
        if 'mendeley_dataset' in str(source_dir):
            for mendeley_class, internal_class in mendeley_label_map.items():
                class_dir = source_dir / mendeley_class
                if class_dir.is_dir():
                    for audio_file in class_dir.glob("*.wav"):
                        data_pool[internal_class].append(audio_file)
        
        # Handle JAMS-based dataset
        elif 'synthetic_validation' in str(source_dir):
            for audio_file in source_dir.glob("*.wav"):
                jams_path = audio_file.with_suffix(".jams")
                if jams_path.exists():
                    try:
                        jam = jams.load(str(jams_path))
                        for event in jam.annotations['scaper'][0]['data']:
                            if event.value['role'] == 'foreground':
                                label = event.value['label']
                                if label in class_names:
                                    data_pool[label].append(audio_file)
                                break
                    except Exception:
                        print(f"Skipping file with problematic JAMS: {jams_path}")
        
        # Handle standard directory-based labels (like Data/v2)
        else:
            for class_name in class_names:
                class_dir = source_dir / class_name
                if class_dir.is_dir():
                    for audio_file in class_dir.glob("*.wav"):
                        data_pool[class_name].append(audio_file)


    print("\nOriginal sample counts per class (after unifying all sources):")
    for class_name in class_names:
        print(f"- {class_name}: {len(data_pool[class_name])} samples")

    # --- 2. Targeted Offline Augmentation ---
    print("\nStep 2: Performing targeted offline augmentation...")
    
    noise_files = glob.glob(os.path.join(noise_dir, '**/*.wav'), recursive=True)
    if not noise_files:
        print("Warning: No noise files found. Augmentation will proceed without background noise.")

    augment_pipeline = Compose([
        AddBackgroundNoise(sounds_path=noise_files, min_snr_db=5.0, max_snr_db=25.0, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5)
    ])
    
    TARGET_SR = 16000

    for class_name, file_list in tqdm(data_pool.items(), desc="Augmenting Classes"):
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # First, copy all original files to the new location
        for original_file in file_list:
            shutil.copy(original_file, output_class_dir)
            
        # Now, generate new files until the target is reached
        num_current_files = len(file_list)
        
        # If a class has 0 files, we cannot augment it.
        if num_current_files == 0:
            continue
            
        num_to_generate = args.samples_per_class - num_current_files
        
        if num_to_generate <= 0:
            continue

        pbar = tqdm(total=num_to_generate, desc=f"Generating for {class_name}", leave=False)
        generated_count = 0
        while generated_count < num_to_generate:
            source_file_to_augment = random.choice(file_list)
            
            try:
                samples, sr = librosa.load(source_file_to_augment, sr=TARGET_SR, mono=True)
                
                augmented_samples = augment_pipeline(samples=samples, sample_rate=sr)
                
                output_filename = f"{source_file_to_augment.stem}_aug_{generated_count + 1}.wav"
                output_path = output_class_dir / output_filename
                sf.write(output_path, augmented_samples, sr)
                
                generated_count += 1
                pbar.update(1)
            except Exception as e:
                print(f"\n[Warning] Failed during augmentation for {source_file_to_augment}. Error: {e}")
        pbar.close()

    print(f"\n✅ Final dataset prepared at: {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare the final, balanced, augmented dataset.")
    parser.add_argument('--source_dirs', nargs='+', default=['Data/v2', 'external_data/mendeley_dataset'],
                        help='List of source directories containing data to unify.')
    parser.add_argument('--noise_dir', type=str, default='external_data/musan/noise',
                        help='Directory for background noise files.')
    parser.add_argument('--output_dir', type=str, default='Data/final_balanced_dataset',
                        help='Directory to save the final dataset.')
    parser.add_argument('--samples_per_class', type=int, default=500,
                        help='The target number of samples for each class after augmentation.')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    prepare_dataset(args)
