# audioset_utils

Audioset<sup>[1]</sup> dataset is more than 2 million sounds clips drawn from youtube videos, it is freely available. For example, hugging face hub has it available where it can be downloaded or used in the streaming mode from here<sup>[2]</sup>. Since the dataset is huge, when we try to use it thru huggingface datasets by creating the dataset object like this;
```python
from datasets import load_dataset
dataset = load_dataset("agkphysics/AudioSet", 'unbalanced',
    trust_remote_code=True, cache_dir='/your/local/cache/dir',
    )
```
the *load_dataset* will download and cache the entire dataset in the cache directory. But this is going to download the entire dataset that is more than 2 TB of data. So make sure to set the *cache_dir* to the directory with enough space.
Entire dataset has more than 500 unique labels, the number of examples for each sound category are listed here<sup>[1]</sup>. 

This repository contains modules for managing and processing audio datasets, including downloading from YouTube, handling metadata, and creating datasets for machine learning.

If you want to use only subject of sound categories, creating this dataset object would require downloading the entire dataset needing a huge amount of disk space. Even if one tries to create dataset in the *streaming* mode (by setting *streaming=True*). In order to filter the dataset, it requires to go through the entire dataset. This requires huge disk space & consumes a lot of time.

In order to solve that problem, I created this repo where I read the metadata file and the file containing class labels details, (both of these can be downloaded from [here](https://research.google.com/audioset/download.html)). Using this, the user can go through the dataset labels, filter out the unwanted labels and download the data directly from youtube links. It saves both the required disk space and processing time. Here is the summary of provided functionality;
- Explore the metadata and see the categories of sound available (before downloading dataset).
- Filter out the unwanted labels and calculate the approximate hours of recording data for the remaining labels.
- Download audio data for the categories of interest, and save the data at the preferred sampling rate and prefered format e.g. *wav*, *flac* etc.

## Installation

To set up this repository locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/bilalhsp/audioset-utils.git
    cd audioset-utils
    ```

2. **Install the package:**
    ```bash
    pip install .
    ```
This will install the package alongwith any dependencies.


## Usage

Below are brief examples of how to use each module:

**AudiosetMetadata:** Create metadata object and combines information from metadata file and class labels file that contains descriptive labels.
```python
from audioset_utils import AudiosetMetadata
data_dir = '/path/to/your/data_dir'
metadata_filename = 'unbalanced_train_segments.csv.crdownload'
class_labels_filename = 'class_labels_indices.csv.crdownload'

metadata_config = {
    'data_dir': data_dir,
    'metadata_filename': metadata_filename,
    'class_labels_filename': class_labels_filename,
}

audioset = AudiosetMetadata(**metadata_config)
human_labels = audioset.get_all_labels()
```

**AudiosetDownloader:** Uses metadata object and other classes to provid high level functionality for downloading dataset. I have used [3] and [4] to download youtube audio and extract 10 second clips respectively. Here is a simple example of how to create an instance of downlaoded.

```python
from audioset_utils import AudiosetDownloader
config = {
    'data_dir': '/path/to/your/data_dir',
    'sampling_rate': 48000,
    'prefered_codec': 'wav',
    'metadata_filename': 'unbalanced_train_segments.csv',
    'class_labels_filename': 'class_labels_indices.csv',
}

downloader = AudiosetDownloader(**config)
```
Once the downloader object is created, here is an example of downloading the dataset excluding *speech* labels, using 4 parallel processes. Once the dataset is downloaded, it created an output metadata file with details of downloaded audios. It is highly recommeded to use multiprocessing feature if possible. I was able to download subset of dataset within 2 hours using num_proc=128 that would otherwise have taken several days using single process.

```python
downloader.filter_and_download_audioset(
    excluded_labels=['speech'],         
    num_proc=4,                         # use for multi-processing
    output_metadata_filename=None,	    # using default file name
    # Default name of ouput metadata filename 'audioset_metadata.csv'
)
```

For detailed example, refer to my [script](./scripts/download_subset_audioset.py) that downloads dataset by excluding all the labels mentioned in the [exclude_labels](./scripts/excluded_labels.yml) file.

**AudioDataset:** As mentioned above, downloader saves a output metadata (.csv) file carrying the details of downloaded audios. This file is directly useful for creating a ready to use dataset. 

```python
from audioset_utils import AudioDataset
data_dir = '/path/to/your/data_dir'
output_metadata_filename = 'audioset_metadata.csv'
dataset = AudioDataset(data_dir, output_metadata_filename)

hf_dataset = dataset.get_hf_dataset()
```
The last line returns an instance of hugging face dataset object, that is ready to use. In order to simply things evern futher, I have provided a function that directly return hugging face dataset. 
```python
from audioset_utils import get_huggingface_dataset

data_dir = '/path/to/your/data_dir'
output_metadata_filename = 'audioset_metadata.csv'
hf_dataset = get_huggingface_dataset(data_dir, output_metadata_filename)
```
This hugging face dataset is ready to use and it would have four features;
- **audio** 
- **youtube video ids**
- **labels**
- **human_labels**

## References

1. [Audioset Dataset](https://research.google.com/audioset/)
2. [Audioset (HuggingFace)](https://huggingface.co/datasets/agkphysics/AudioSet)
3. [yt-dpl](https://github.com/yt-dlp/yt-dlp)
4. [pydub](https://github.com/jiaaro/pydub)


[1]: https://research.google.com/audioset/
[2]: https://huggingface.co/datasets/agkphysics/AudioSet
[3]: https://github.com/yt-dlp/yt-dlp
[4]: https://github.com/jiaaro/pydub





