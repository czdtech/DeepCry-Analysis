
"""
downloader.py

This module provides functionality for downloading audio data
from youtube videos.

Author: Bilal Ahmed
Date: 07-16-2024
Version: 1.0
License: MIT
Dependencies: None

Purpose
-------
This module provides high level access to dataset downloading,
after excluding certain filters. User can specify the 
sampling rate of downloaded dataset and preferred codec of
downloaded files e.g. wav, flac etc.
In addition multiprocessing is also supported to speed up the
download process.

Classes:
    AudiosetDownloader: A class to handle audio datasets using
        metadata and file paths.

    Methods:
        filter_and_download_audioset: method that provides
            labels filtering and downloading functionlity,
            user can specify 'num_proc' argument to enable 
            multiprocessing. See scripts for example usage.  

Change Log
----------
- 07-16-2024: Initial version created by Bilal Ahmed.
"""
import os
from audioset_utils.yt_audio import AudiosetYouTube
from audioset_utils.metadata import AudiosetMetadata
from multiprocessing import Pool



class AudiosetDownloader:
    def __init__(self, data_dir, sampling_rate, prefered_codec, metadata_filename, class_labels_filename):
        """Downloader object that creates instances of AudiosetMetadata and AudiosetYouTube
        and downloads dataset with the given preferences.
        """
        self.data_dir = data_dir
        metadata_config = {
            'data_dir': data_dir,
            'metadata_filename': metadata_filename,
            'class_labels_filename': class_labels_filename,
        }
        ayt_config = {
            'data_dir': data_dir,
            'sampling_rate': sampling_rate,
            'prefered_codec': prefered_codec,
        }

        self.audioset = AudiosetMetadata(**metadata_config)
        self.ayt = AudiosetYouTube(**ayt_config)

    def filter_and_download_audioset(
            self, excluded_labels:list, num_proc:int =1, output_metadata_filename:str =None):
        """Using the audioset metadata, filters out excluded_labels
        and downloads the remaining dataset. Also creates metadata 
        file 'audioset_metadata.csv' for the downlaoded audios, that 
        allows it to be used right away.
        
        Args:
            excluded_labels: list = list of labels to be excluded.
            output_metadata_filename: str = filename to be created with metadata of downloaded examples.
            num_proc = int = Default = 1, number of parallel processes
        
        """
        if output_metadata_filename is None:
            output_metadata_filename = 'audioset_metadata.csv'
        filtered_dataset = self.audioset.filter_dataset(excluded_labels)

        # extracting lists for downloading functions...
        video_ids = list(filtered_dataset['youtube_ids'])
        start_times = list(filtered_dataset['start_times'])
        end_times = list(filtered_dataset['end_times'])

        mask = self.download_all_examples(
            video_ids, start_times, end_times, num_proc=num_proc
            )

        # # for each example (row), download audio clip
        # mask = filtered_dataset.apply(self.download_audio_example, axis=1)
        downloaded_dataset = filtered_dataset[mask].reset_index(drop=True)

        # saving metadata of downloaded dataset to data_dir
        metadata_path = os.path.join(self.data_dir, output_metadata_filename)
        downloaded_dataset.to_csv(metadata_path, index=False)
        print(f"Dataset ready, use {output_metadata_filename} to create dataset.")
     


    ########        multiprocess functionaly ################
    def _download_one_example(self, args):
        """This method expects a tuple containing three fields
        'video_id', 'start_time', 'end_time', as required by
        the 'self.ayt.get_audio_slice_from_youtube' method.
        """
        try:
            filename = self.ayt.get_audio_slice_from_youtube(*args)
            return True
        except: 
            print(f"\n Error downloading video ID={args[0]}.....")
            return False
        
    def download_all_examples(self, video_ids, start_times, end_times, num_proc=1):
        """This method distributes the downloading of all the examples specified by
        three lists video_ids, start_times, end_times among num_processes processes.
        Also returns a list of statuses for each example specifying if the file
        was downloaded successfully of not.

        Args:
            video_ids: list = Video ids to be downloaded.
            start_times: list = Start times.
            end_times: list = End times.
            num_proc: int = num of tasks running in parrallel.

        Returns:
            results: list of boolean entries specifying status of download.
        """
        with Pool(num_proc) as pool:
            results = pool.map(self._download_one_example, zip(video_ids, start_times, end_times))
        return results




    