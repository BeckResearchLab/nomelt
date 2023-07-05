"""Trains and tests a regressor of OGT.
"""
import os
import time
from yaml import safe_load as yaml_load
from yaml import dump as yaml_dump
import json
import pandas as pd
import numpy as np
import duckdb as ddb
import re

import datasets
import sklearn.utils
import codecarbon

if 'SLURM_NTASKS' in os.environ:
    CPU_COUNT = int(os.environ['SLURM_NTASKS'])
else:
    import multiprocessing
    CPU_COUNT = multiprocessing.cpu_count()

import logging
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

def load_data(db_file, min_temp_diff, min_align_cov=0.75):
    # Load data from SQL database
    conn = ddb.connect(db_file, read_only=True)
    query = f"""
    SELECT proteins_m.protein_seq, proteins_t.protein_seq, m.taxid
    FROM protein_pairs
    INNER JOIN proteins AS proteins_m ON (protein_pairs.meso_pid=proteins_m.pid)
    INNER JOIN proteins AS proteins_t ON (protein_pairs.thermo_pid=proteins_t.pid)
    INNER JOIN taxa AS m ON protein_pairs.meso_taxid = m.taxid
    INNER JOIN taxa AS t ON protein_pairs.thermo_taxid = t.taxid
    WHERE ABS(m.temperature - t.temperature) >= {min_temp_diff}
    WHERE (protein_pairs.query_align_cov+protein_pairs.subject_align_cov)/2.0 >= {min_align_cov}
    """
    dataset = datasets.Dataset.from_sql(
        query,
        config_name='test',
        cache_dir='./tmp/hf_cache',
        con=conn)
    conn.close()
    return dataset


if __name__ == '__main__':
    # start logger
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    utils_logger = logging.getLogger('l2tml_utils')
    utils_logger.setLevel(getattr(logging, LOGLEVEL))
    utils_logger.addHandler(fh)
    
    # create dirs
    if not os.path.exists('./data/ogt_protein_regressor/data'):
        os.makedirs('./data/ogt_protein_regressor/data')

    # device settings
    logger.info(f'Using {CPU_COUNT} cpus for multiprocessing')

    # load parameters
    with open("./params.yaml", "r") as stream:
        params = yaml_load(stream)
    logger.info(f"Loaded parameters: {params}")
    
    # start carbon tracker for data processing
    data_tracker = codecarbon.OfflineEmissionsTracker( 
        project_name="data_process_regressor",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    data_tracker.start()
    
    # get data
    ds = load_data('../data/database.ddb', params['min_temp_diff'], params['min_align_cov'])
    # split the data
    data_dict = ds.train_test_split(test_size=0.1)
    logger.info(f"Split data into train and test.")

    # remove unnecessary columns
    if not params['dev_keep_columns']:
        bad_columns = [k for k in data_dict['train'].column_names if k not in['protein_seq', 'labels']]
        data_dict = data_dict.remove_columns(bad_columns)

    logger.info(f'Final datasets: {data_dict}')
    data_dict.cleanup_cache_files()
    data_dict.save_to_disk('./data/dataset/')
    logger.info("Saved data to disk.")
    
    # get co2
    co2 = data_tracker.stop()

    metrics = {
        'data_co2': float(co2),
        'data_n_train': len(data_dict['train']),
        'data_n_test': len(data_dict['test']),
    }

    # save metrics
    with open('./data/ogt_protein_regressor/data_metrics.yaml', "w") as stream:
        yaml_dump(metrics, stream)
