"""
yt_audio.py

This module provides functionality for downloading audio data
from youtube videos.

Author: Bilal Ahmed
Date: 07-16-2024
Version: 1.0
License: MIT
Dependencies: None

Purpose
-------
This module is designed to provide functionaly of downloading 
audioset (audio data provided by Google) directly from youtube
using video ID, start_time, end_time.  


Change Log
----------
- 07-16-2024: Initial version created by Bilal Ahmed.
"""
import os
from pathlib import Path
from yt_dlp import YoutubeDL
from pydub import AudioSegment


class AudiosetYouTube:
    def __init__(self, data_dir, sampling_rate=48000, prefered_codec=None, prefered_quality=None):
        """Constructor for AudiosetYouTube, it takes optional
        parameters for specifying the format and quality 
        of saved audio files.
        
        Args:
            data_dir: = path of dir where downloaded audios will be saved.
            prefered_codec: str = codec for audio format e.g. mp3
            prefered_quality: str = sampling rate of audio e.g. 192 KHz 
        """
        if prefered_codec is None:
            prefered_codec = 'mp3'
        if prefered_quality is None:
            prefered_quality = '192'
        self.prefered_codec = prefered_codec
        self.prefered_quality = prefered_quality
        self.output_dir = Path(data_dir) / 'audio_data'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # self.output_dir = os.path.join(output_dir, 'audio_data')
        self.sampling_rate = sampling_rate
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': prefered_codec,
                'preferredquality': prefered_quality,   # does not matter for wav
            }],
            'postprocessor_args': [
                '-ar', str(sampling_rate),
                '-ac', '1',
            ],
            'outtmpl': os.path.join(self.output_dir, '%(id)s.%(ext)s'),
            'quiet': True,
        }

    def _get_youtubeURL(self, video_id, start_time=None, end_time=None):
        """Returns youtube URL using the provided specifications."""
        youtube_url = f'https://www.youtube.com/watch?v={video_id}'
        if start_time is not None:
            youtube_url += f'&start={int(float(start_time))}'
        if end_time is not None:
            youtube_url += f'&end={int(float(end_time))}'
        return youtube_url

    def _download_audio(self, youtube_url):
        """Downloads audio from the given youtube url, saves to disk 
        and return the filename."""
        with YoutubeDL(self.ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            audio_file = ydl.prepare_filename(info_dict)
            audio_file = audio_file.replace(
                    '.webm', f'.{self.prefered_codec}'
                    ).replace('.m4a', f'.{self.prefered_codec}')
        return audio_file
    
    def _get_audio_slice(self, audio_file, start_time, end_time):
        """Returns the audio slice from start_time to end_time
        
        Args:
            audio_file= filename of audio to be sliced
            start_time= time in seconds
            end_time= time in seconds
        """
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(self.sampling_rate)
        start_time_ms = float(start_time) * 1000  # Convert to milliseconds
        end_time_ms = float(end_time) * 1000  # Convert to milliseconds
        audio_segment = audio[start_time_ms:end_time_ms]

        file_path = Path(audio_file)
        filename = file_path.stem + f'_{int(float(start_time))}_{int(float(end_time))}.{self.prefered_codec}'
        output_file = file_path.parent / filename
        # Save the extracted segment
        # output_file = os.path.join(self.output_dir, f'audio_{info_dict["id"]}_{start_time}-{end_time}.mp3')
        audio_segment.export(output_file, format=self.prefered_codec)
        # Clean up the downloaded full audio file
        os.remove(audio_file)
        return filename
    
    def get_audio_slice_from_youtube(self, video_id, start_time, end_time):
        """Returns the audio slice from start_time to end_time taken
        from youtube video with video_id.
        
        Args:
            video_id = video ID of youtube video
            audio_file= filename of audio to be sliced
            start_time= time in seconds
            end_time= time in seconds
        """
        filename = video_id + f'_{int(float(start_time))}_{int(float(end_time))}.{self.prefered_codec}'
        filepath = self.output_dir / filename
        if not filepath.exists():
            youtube_url = self._get_youtubeURL(video_id, start_time, end_time)
            audio_file = self._download_audio(youtube_url)
            filename = self._get_audio_slice(audio_file, start_time, end_time)
            print(f"Filename: {filename} Downloaded.")
        else:
            print(f"Filename: {filename} already exists...")
        return filename