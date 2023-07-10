"""Prepare protein sequence pairs as a HF dataset
"""
import os
import pickle
import time
from yaml import safe_load as yaml_load
from yaml import dump as yaml_dump
import json
import pandas as pd
import numpy as np
import duckdb as ddb
import re
import dvc.api

import datasets
import sklearn.utils
import sklearn.model_selection
import codecarbon

import nomelt.deduplication

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

def load_data(db_file, min_temp_diff, min_thermo_temp, min_align_cov=0.75):
    # check the cache
    if os.path.exists('./tmp/hf_cache/ds_cache_key.pkl'):
        with open('./tmp/hf_cache/ds_cache_key.pkl', 'rb') as f:
            cache_key = pickle.load(f)
        if cache_key == (min_temp_diff, min_thermo_temp, min_align_cov):
            logger.info(f"Loading data from cache.")
            dataset = datasets.load_from_disk('./tmp/hf_cache/ds_save/')
            return dataset

    # Load data from SQL database
    conn = ddb.connect(db_file, read_only=True)
    query = f"""
    SELECT proteins_m.protein_seq AS meso_seq, proteins_t.protein_seq AS thermo_seq, m.taxid
    FROM pairs
    INNER JOIN proteins AS proteins_m ON (pairs.meso_pid=proteins_m.pid)
    INNER JOIN proteins AS proteins_t ON (pairs.thermo_pid=proteins_t.pid)
    INNER JOIN taxa AS m ON pairs.meso_taxid = m.taxid
    INNER JOIN taxa AS t ON pairs.thermo_taxid = t.taxid
    WHERE ABS(m.temperature - t.temperature) >= {min_temp_diff}
    AND t.temperature >= {min_thermo_temp}
    AND ((pairs.query_align_cov+ pairs.subject_align_cov)/2.0)>={min_align_cov}
    """
    dataset = datasets.Dataset.from_sql(
        query,
        config_name='test',
        cache_dir='./tmp/hf_cache',
        con=conn)
    conn.close()
    dataset.save_to_disk('./tmp/hf_cache/ds_save/')
    dataset = datasets.load_from_disk('./tmp/hf_cache/ds_save/')

    # cache the key
    with open('./tmp/hf_cache/ds_cache_key.pkl', 'wb') as f:
        pickle.dump((min_temp_diff, min_thermo_temp, min_align_cov), f)
    return dataset

if __name__ == '__main__':
    # start logger
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    utils_logger = logging.getLogger('nomelt')
    utils_logger.setLevel(getattr(logging, LOGLEVEL))
    utils_logger.addHandler(fh)

    # device settings
    logger.info(f'Using {CPU_COUNT} cpus for multiprocessing')

    # load parameters
    params = dvc.api.params_show(stages="prepare_data")
    logger.info(f"Loaded parameters: {params}")
    
    # start carbon tracker for data processing
    data_tracker = codecarbon.OfflineEmissionsTracker( 
        project_name="data_processr",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    data_tracker.start()
    
    # get data
    ds = load_data('./data/database.ddb', params['data']['min_temp_diff'], params['data']['min_thermo_temp'], params['data']['min_align_cov'])
    logger.info(f"Loaded data from database: {len(ds)} pairs.")
    if params['data']['dev_sample_data']:
        ds = ds.shuffle().select(range(params['data']['dev_sample_data']))
        logger.info(f"Sampled data: {len(ds)} pairs.")

    # add a columnd equal to the index - we need to keep track of original indices
    # because we will be removing duplicates based on index
    def add_index(examples, idxs):
        examples['index'] = idxs
        return examples
    ds = ds.map(add_index, with_indices=True, batched=True)

    # remove similar pairs
    logger.info(f"Removing similar pairs based on meso seq kgrams.")
    duplicate_clusters, cluster_summary, duplicate_indices, extreme_indices = nomelt.deduplication.deduplicate_dataset(
        ds,
        sequence_key='meso_seq',
        jaccard_threshold=params['data']['minhash_threshold'],
        num_perm=params['data']['minhash_num_perm'],
        k=params['data']['kgram'])

    # remove duplicate proteins that are not extremes, if specified
    if params['data']['keep_only_extremes']:
        logger.info(f"Removing non-extreme duplicates.")
        remove_indices = duplicate_indices - extreme_indices
        ds = ds.filter(lambda x, idx: idx not in remove_indices, with_indices=True)
        logger.info(f"Removed {len(remove_indices)} non-extreme duplicates, {len(ds)} pairs remaining.")
    # split data keeping clusters together
    logger.info(f"Splitting data into train and test by cluster")
    def add_cluster_column(example):
        """We want to assign the cluster id here if it is in a cluster, otherwise keep the indexes.
        Note, since indexes start at 0 and increment up, cluster ids will be negative to not overlap.
        """
        if example['index'] in duplicate_indices:
            for cluster_id, cluster_dict in duplicate_clusters.items():
                if example['index'] in cluster_dict['ids']:
                    example['cluster'] = -cluster_id
                    if example['index'] in cluster_dict['extreme_ids']:
                        example['status_in_cluster'] = 'extreme'
                    else:
                        example['status_in_cluster'] = 'duplicate'
                    break
        else:
            example['cluster'] = example['index']
            example['status_in_cluster'] = 'unique'
        return example
    ds = ds.map(add_cluster_column, batched=False, num_proc=CPU_COUNT)
    # group shuffle splitter to keep clusters together
    group_shuffle_splitter = sklearn.model_selection.GroupShuffleSplit(
        n_splits=1, train_size=1-params['data']['test_size'])
    for train_indexes, test_indexes in group_shuffle_splitter.split(X=None, groups=ds['cluster']):
        pass
    train = ds.select(train_indexes).shuffle()
    test = ds.select(test_indexes).shuffle()
    logger.info(f"""SPLITTING REPORT
-------------------
Train size: {len(train)}
Train clusters: {[c for c in set(train['cluster']) if c<0]}
Train fraction of examples in clusters: {sum(np.array(train['cluster']) < 0)/len(train)}
Test size: {len(test)}
Test clusters: {[c for c in set(test['cluster']) if c<0]}
Test fraction of examples in clusters: {sum(np.array(test['cluster']) < 0)/len(test)}
""")
    data_dict = datasets.DatasetDict({'train': train, 'test': test})

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
    with open('./data/data_metrics.yaml', "w") as stream:
        yaml_dump(metrics, stream)
