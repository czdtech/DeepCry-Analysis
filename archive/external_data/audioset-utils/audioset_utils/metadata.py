"""
metadata.py

This module provides functionality to read the metadata files
provided on Audioset official website.

Author: Bilal Ahmed
Date: 07-16-2024
Version: 1.0
License: MIT
Dependencies: None

Purpose
-------
This module is designed to provide functionaly to explore the
types of audios available, access the descriptive human labels
and filter to get subsect of labels. It allows to have a feel 
of what type of audios are in the dataset and what would be 
the size of dataset (number of examples as well as hours of 
recording) for selected labels only.

Classes:
    AudiosetMetadata: A class to explor dataset segment using
	    metadata and class labels file.	

Change Log
----------
- 07-16-2024: Initial version created by Bilal Ahmed.
"""
import os
import csv
import pandas as pd
from pathlib import Path

class AudiosetMetadata:
	"""This provides details about unbalanced_train_segments data of audioset.
	"""
	def __init__(self, data_dir, metadata_filename, class_labels_filename):
		"""Create audioset metadata object that allows to explore 
		types of audios and filter out categories of sounds that are
		not needed before downloading.
		
		Args:
			data_dir: path of directory containing metadata file and file
				of class labels 'class_labels_indices.csv'
			metadata_filename: filename of data segment e.g. audio_train_unbalanced.csv
			class_labels_filename = file with detail of all 527 labels.
		"""
		self.audioset_dir = Path(data_dir)
		self.metadata_filename = metadata_filename
		self.class_labels_filename = class_labels_filename

		self.label_descriptions = self.read_label_descriptions()
		self.metadata = self.read_metadata()
		self.add_human_labels()
		self.excluded_labels = []

	def read_metadata(self):
		filepath = os.path.join(self.audioset_dir, self.metadata_filename)
		youtube_ids = []
		start_times = []
		end_times = []
		labels = []
		with open(filepath, newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
			for ind, row in enumerate(spamreader):
				if ind > 3:
					youtube_ids.append(row[0])
					start_times.append(row[1].strip(' '))
					end_times.append(row[2].strip(' '))
					list_labels = row[3:]
					list_labels = [lab.strip('" ') for lab in list_labels]
					labels.append(list_labels)
		data_dict = {
			'youtube_ids': youtube_ids,
			'start_times': start_times,
			'end_times': end_times,
			'labels': labels, 
		}
		return pd.DataFrame(data_dict)
		
	def read_label_descriptions(self):
		filepath = os.path.join(self.audioset_dir, self.class_labels_filename)
		label_indices = []
		labels = []
		human_labels = []
		with open(filepath, newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
			for ind, row in enumerate(spamreader):
				if ind > 0:
					label_indices.append(row[0])
					labels.append(row[1])
					# for lab in row[2:]:
					descriptive_list = row[2:]
					descriptive_list = [des.strip('"').lower() for des in descriptive_list]
					human_labels.append(descriptive_list)

		label_description = {lab: desc for lab, desc in zip(labels, human_labels)}
		return label_description

	def add_human_labels(self):
		"""Adds human labels to the metadata (dataframe) as a new column."""
		human_labels = []
		for labels in self.metadata['labels']:
			human_labels_row = []
			for label in labels:
				# print(label)
				descriptions = self.get_label_descriptions(label)
				# print(descriptions)
				human_labels_row.extend(descriptions)
			human_labels.append(human_labels_row)
		# adding human labels to metadata
		self.metadata['human_labels'] = human_labels

	def get_all_labels(self):
		"""Returns the list of all 527 labels in the dataset."""
		return list(self.label_descriptions.values())
		

	def get_label_descriptions(self, label):
		"""Returns descriptive labels for label IDs"""
		try:
			label = str(label)
			return self.label_descriptions[label]
		except:
			raise KeyError("Invalid label ID..")

	def get_dataset_size(self):
		return self.metadata.shape[0]


	def filter_func(self, row):
		"""Returns True for rows having no excluded label in human_labels"""
		return not any(label in self.excluded_labels for label in row['human_labels'])

	def filter_dataset(self, excluded_labels=None):
		"""Filters dataset by excluding labels and returns filtered dataset."""
		if excluded_labels is None:
			excluded_labels = ['speech', 'music']
		self.excluded_labels = excluded_labels
		mask = self.metadata.apply(self.filter_func, axis=1)
		filtered_datast = self.metadata[mask]
		return filtered_datast.reset_index(drop=True)

