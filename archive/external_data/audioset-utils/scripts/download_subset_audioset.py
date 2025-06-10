"""This script uses audioset taxonomoy and filters out
the human labels that are not needed before it downloads the 
required filters and saves them to the disk.
"""
import os
import time
import yaml
import argparse
from audioset_utils import AudiosetDownloader


def download_dataset(args):

    data_dir = args.data_dir
    metadata_filename = args.metadata_file
    class_labels_filename = args.labels_file

    # Get the absolute path to the current script
    parent_dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(parent_dir_path, 'excluded_labels.yml'), 'r') as F:
        excluded_labels = yaml.load(F, yaml.FullLoader)
    excluded_labels = excluded_labels['excluded_labels']
    
    config = {
        'data_dir': data_dir,
        'sampling_rate': 48000,
        'prefered_codec': 'wav',
        'metadata_filename': metadata_filename,
        'class_labels_filename': class_labels_filename,
    }

    downloader = AudiosetDownloader(**config)
    downloader.filter_and_download_audioset(
        excluded_labels=excluded_labels,
        num_proc=args.num_proc,
        output_metadata_filename=None,	# using default file name
    )

# ------------------  get parser ----------------------#

def get_parser():
    parser = argparse.ArgumentParser(
        description='This is to filter and download Audioset dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-n','--num_proc', dest='num_proc', type=int, default=1,
        help="Specify the number of processes."
    )
    parser.add_argument(
        '-d','--data_dir', dest='data_dir', type=str,
        help="Directory path containing metadata and label details files."
    )
    parser.add_argument(
        '-m','--metadata', dest='metadata_file', type=str,
        help="Name of metadata file for the segment of data to be downloaded."+
        "e.g. eval_segments.csv"
    )
    parser.add_argument(
        '-l','--labels', dest='labels_file', type=str,
        default='class_labels_indices.csv.crdownload',
        help="Name of class labels file, e.g. class_labels_indices.csv"
    )

    return parser

# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    download_dataset(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")

