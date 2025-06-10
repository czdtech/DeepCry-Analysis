"""
dataset.py

This module provides functionality for downloading audio data
from youtube videos.

Author: Bilal Ahmed
Date: 07-18-2024
Version: 1.0
License: MIT
Dependencies: None

Purpose
-------
Once filtered audioset dataset is downloaded to disk,
this module helps create dataset using the downloaded 
wav files and metadata file.  

Classes:
    AudioDataset: A class to handle audio datasets using metadata and file paths.
	
Functions:
    get_huggingface_dataset(audio_dir: str, metadata_file: str, ) -> datasets.Dataset: 
        Creates a Hugging Face dataset from metadata CSV and audio files.

Change Log
----------
- 07-18-2024: Initial version created by Bilal Ahmed.
"""

import os
import pandas as pd
from datasets import Dataset, Audio, Value, Features, Sequence


def get_huggingface_dataset(data_dir: str, metadata_file: str):
	"""
	This method provides high level functionality so that
	user does not have to go into the details of metadata file
	of create an instance of AudioDataset class. 

	Args:
		data_dir: directory path containing metadata file and sub dir audio_data
		metadata_file: csv file automatically created by the downloader object.

	Returns:
		an object of huggingface dataset.
	"""
	dataset = AudioDataset(data_dir, metadata_file)
	return dataset.get_hf_dataset()



class AudioDataset:
	"""Given the metadata file created by downloader, this creates
	a dataset, that has the functionality to return huggingface dataset 
	as well.
	"""
	def __init__(self, data_dir, metadata_file, sampling_rate=48000):
		# self.metadata_path = os.path.join(data_dir, metadata_file)
		self.data_dir = data_dir
		self.metadata = pd.read_csv(os.path.join(data_dir, metadata_file))
		# adding filepaths column...
		filepaths = self.metadata.apply(self.get_filepath, axis=1)
		self.metadata['filepaths'] = filepaths
		self.sampling_rate = sampling_rate

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, idx):
		"""Returns the filepath, video_id, label, human_label corresponding to idx"""
		row = self.metadata.iloc[idx]
		out = {
			'audio': os.path.join(self.data_dir, row['filepaths']),
			'video_id': row['youtube_ids'],
			'labels': row['labels'].strip("[]").replace("'", "").split(', '),
			'human_labels': row['human_labels'].strip("[]").replace("'", "").split(', '),
		}
		return out

	def generator(self):
		for idx in range(len(self.metadata)):
			yield self[idx]
	
	def get_hf_dataset(self):
		"""Returns an instance of huggingface dataset."""
		features = {
			'audio': Audio(sampling_rate=self.sampling_rate),
			'video_id': Value('string'),
			'labels': Sequence(Value('string'), length=-1),
			'human_labels': Sequence(Value('string'), length=-1),
		}

		dataset = Dataset.from_generator(self.generator, features=Features(features))
		return dataset


	def get_filepath(self, row):
		"""For each example, returns the filepath of the downloaded audio file."""
		filepath =   row['youtube_ids']+f"_{int(float(row['start_times']))}_{int(float(row['end_times']))}.wav"
		return 'audio_data/'+filepath