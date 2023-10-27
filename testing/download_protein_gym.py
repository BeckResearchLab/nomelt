"""
This script downloads DMS data (as csv files) from the Marks lab Protein Gym dataset.
"""
# system dependencies
import os
import logging
import shutil

# library dependencies
from datasets import DownloadManager
from tqdm import tqdm
import numpy as np
import pandas as pd

# local dependencies

## initialize logger
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

## constants
csv_files = [
    "BLAT_ECOLX_Deng_2012.csv"
] # initalize CSVs
# Base URL for the ProteinGym_substitutions folder (change to indels)
base_url = "https://huggingface.co/datasets/OATML-Markslab/ProteinGym/raw/main/ProteinGym_substitutions/"

# Specify a directory to store the downloaded data
download_dir = "./data/gym/"


if __name__ == "__main__":
    # initialize logger
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    try:
        os.makedirs(download_dir, exist_ok=True)
        logger.info("download directory has been created (or already exists!)")
    except OSError as e:
        logger.info(f'Error creating directory: {e}')


# Initialize the download manager
download_manager = DownloadManager(dataset_name="ProteinGym")

# Attempt to download the csv files again
downloaded_paths = {}

for file_name in csv_files:
    data_url = base_url + file_name
    try:
        downloaded_file_path = download_manager.download(data_url)
        downloaded_paths[file_name] = downloaded_file_path
        logger.info(f"downloaded {data_url}")
        # After downloading
        shutil.move(downloaded_file_path, os.path.join(download_dir, file_name))
        logger.info(f"downloaded file moved from the huggingface cache to {download_dir}")
    except Exception as e:
        downloaded_paths[file_name] = f"Error: {e}"