"""
TODO
"""
# system dependencies
import logging
import os
import sys

# library dependencies
import duckdb as ddb
import numpy as np
import pandas as pd
from datasets import load_dataset
import codecarbon

# local dependencies
import nomelt.model as nomelt

##########################
## environment variables #
##########################
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

################
## constants ###
################
ANALYSIS_DIR = "./data/gym/analysis/"

################
## functions ###
################


if __name__ == "__main__":
    # initialize logger
    # start logger
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    utils_logger = logging.getLogger('nomelt')
    utils_logger.setLevel(getattr(logging, LOGLEVEL))
    utils_logger.addHandler(fh)

    try:
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        logger.info("Analysis directory has been created (or already exists!)")
    except OSError as e:
        logger.info(f'Error creating directory: {e}')

    # initialize codecarbon
    tracker = codecarbon.OfflineEmissionsTracker(
    project_name="estimate_various_mutations",
    output_dir="./data/",
    country_iso_code="USA",
    region="washington")

    tracker.start()
