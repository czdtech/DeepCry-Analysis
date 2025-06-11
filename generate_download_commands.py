#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a Windows batch file (.bat) with yt-dlp commands to download
the infant cry subset of AudioSet.
"""
import csv
from pathlib import Path
from tqdm import tqdm

def generate_download_script():
    """
    Parses AudioSet metadata and generates a batch script of download commands.
    """
    # --- 1. Setup Paths ---
    metadata_dir = Path('external_data/audioset-utils/metadata')
    output_audio_dir = Path('external_data/audioset_cries_commercially_usable')
    output_bat_file = Path('download_commands.bat')

    labels_file = metadata_dir / 'class_labels_indices.csv'
    segments_file = metadata_dir / 'unbalanced_train_segments.csv'

    if not labels_file.exists() or not segments_file.exists():
        print("Error: Metadata CSV files not found. Please ensure they are in the metadata folder.")
        return

    # --- 2. Find Target Label ID ---
    print("Finding target label ID for 'Baby cry, infant cry'...")
    target_label_id = None
    target_label_str = "Baby cry, infant cry"
    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 3 and target_label_str in row[2]:
                    target_label_id = row[1]
                    break
    except Exception as e:
        print(f"Error reading labels file: {e}")
        return

    if target_label_id is None:
        print(f"FATAL: Could not find ID for label '{target_label_str}'.")
        return
    print(f"Found ID for '{target_label_str}': {target_label_id}")

    # --- 3. Generate Download Commands ---
    print(f"Scanning '{segments_file.name}' to find matching audio segments...")
    
    commands = []
    # Note: The segments CSV is large, so we stream it.
    with open(segments_file, 'r', encoding='utf-8') as f:
        # Skip header lines, which start with '#'
        for _ in range(3):
            next(f)
        
        reader = csv.reader(f, skipinitialspace=True)
        for row in tqdm(reader, desc="Parsing segments"):
            if len(row) < 4: continue
            
            youtube_id, start_seconds, end_seconds, positive_labels = row
            
            if target_label_id in positive_labels:
                # Construct the yt-dlp command for Windows
                # We assume yt-dlp.exe is in the project root
                output_template = output_audio_dir / f"{youtube_id}_%(start_time)s.%(ext)s"
                
                # Format for yt-dlp postprocessor args: hh:mm:ss.ms
                start_time_str = str(start_seconds)
                end_time_str = str(end_seconds)

                command = (
                    f'yt-dlp.exe -x --audio-format wav '
                    f'--postprocessor-args "ffmpeg_i:-ss {start_time_str} -to {end_time_str}" '
                    f'"https://www.youtube.com/watch?v={youtube_id}" '
                    f'-o "{output_template}"'
                )
                commands.append(command)

    # --- 4. Write the Batch File ---
    print(f"\nFound {len(commands)} matching audio segments.")
    print(f"Writing download commands to '{output_bat_file}'...")
    
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_bat_file, 'w', encoding='utf-8') as f:
        f.write('@echo off\n')
        f.write('echo Starting download of AudioSet cry subset...\n')
        for command in commands:
            f.write(f'echo Downloading segment from video {command.split("v=")[1].split(" ")[0]}...\n')
            f.write(command + '\n')
        f.write('echo.\n')
        f.write('echo ✅ All download commands executed.\n')
        f.write('pause\n')
        
    print(f"✅ Successfully created '{output_bat_file}'.")
    print("You can now run this file on your Windows host to download the data.")

if __name__ == "__main__":
    generate_download_script()
