"""Prepare protein sequence pairs as a HF dataset
"""
import os
import pickle
import time
import subprocess
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
    SELECT 
        proteins_m.protein_seq AS meso_seq,
        proteins_t.protein_seq AS thermo_seq,
        m.taxid,
        pairs.query_align_cov,
        pairs.subject_align_cov,
        pairs.bit_score,
        pairs.scaled_local_symmetric_percent_id,
        LENGTH(proteins_m.protein_seq) AS meso_seq_len,
        LENGTH(proteins_t.protein_seq) AS thermo_seq_len,
        ABS(meso_seq_len - thermo_seq_len) AS seq_len_diff,
        m.temperature AS meso_temp,
        t.temperature AS thermo_temp,
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

def run_cdhit(i, o, c=0.9, n=5, G=1, M=800, T=1, s=0.0, aL=0.0, aS=0.0):
    """
    Run CD-HIT with specified parameters.

    Parameters:
        i (str): Input filename in FASTA format.
        o (str): Output filename.
        c (float, optional): Sequence identity threshold. Default is 0.9.
        n (int, optional): Word length. Default is 5.
        G (int, optional): Use global sequence identity (1) or not (0). Default is 1.
        M (int, optional): Memory limit in MB. Default is 800.
        T (int, optional): Number of threads. Default is 1.
        s (float, optional): Length difference cutoff. Default is 0.0.
        aL (float, optional): Alignment coverage for the longer sequence. Default is 0.0.
        aS (float, optional): Alignment coverage for the shorter sequence. Default is 0.0.
    """
    cmd = [
        'cd-hit',
        '-i', i,
        '-o', o,
        '-c', str(c),
        '-n', str(n),
        '-G', str(G),
        '-M', str(M),
        '-T', str(T),
        '-s', str(s),
        '-aL', str(aL),
        '-aS', str(aS)
    ]
    subprocess.run(cmd)

def parse_clstr(clstr_file):
    cluster_dict = {}
    current_cluster = None

    with open(clstr_file, 'r') as f:
        for line in f:
            if line.startswith(">Cluster"):
                current_cluster = line.strip().split()[-1]
            else:
                seq_id = line.split('>')[1].split('...')[0]
                cluster_dict[int(seq_id)] = int(current_cluster)

    return cluster_dict

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

    # label clusters
    logger.info('Labeling clusters using CD-HIT')
    # Prepare FASTA file from ds
    with open("./tmp/cd_hit_input.fasta", "w") as f:
        for idx, item in enumerate(ds):
            f.write(f">{idx}\n{item['meso_seq']}\n")

    # Run CD-HIT
    t0 = time.time()
    run_cdhit(
        "./tmp/cd_hit_input.fasta",
        "./tmp/cd_hit_output.fasta",
        M=40000,
        T=CPU_COUNT,
        c=params['data']['cd_c'],
        n=params['data']['cd_n'],
        G=params['data']['cd_G'],
        s=params['data']['cd_s'],
        aL=params['data']['cd_aL'],
        aS=params['data']['cd_aS']
    )

    # Parse CD-HIT output
    logger.info('Parsing CD-HIT output')
    clusters = parse_clstr("./tmp/cd_hit_output.fasta.clstr")
    t1 = time.time()
    logger.info(f"Minutes elapsed: {(t1-t0)/60}")

    # Add CD-HIT cluster IDs to ds
    def add_cdhit_cluster(example, idx):
        example['cluster'] = clusters.get(idx, -1)
        return example

    logger.info('Adding CD-HIT cluster IDs to dataset')
    ds = ds.map(add_cdhit_cluster, with_indices=True, batched=False, num_proc=CPU_COUNT)
    
    # group shuffle splitter to keep clusters together
    group_shuffle_splitter = sklearn.model_selection.GroupShuffleSplit(
        n_splits=1, train_size=1-params['data']['test_size'])
    for train_indexes, test_indexes in group_shuffle_splitter.split(X=None, groups=ds['cluster']):
        pass
    train = ds.select(train_indexes).shuffle()
    test = ds.select(test_indexes).shuffle()
    group_shuffle_splitter = sklearn.model_selection.GroupShuffleSplit(
        n_splits=1, train_size=1-params['data']['test_size'])
    for train_indexes, eval_indexes in group_shuffle_splitter.split(X=None, groups=train['cluster']):
        pass
    eval = train.select(eval_indexes).shuffle()
    train = train.select(train_indexes).shuffle()
    logger.info(f"""SPLITTING REPORT
-------------------
Train size: {len(train)}
Train clusters: {[c for c in set(train['cluster']) if c<0]}
Train fraction of examples in clusters: {sum(np.array(train['cluster']) < 0)/len(train)}
Eval size: {len(eval)}
Eval clusters: {[c for c in set(eval['cluster']) if c<0]}
Eval fraction of examples in clusters: {sum(np.array(eval['cluster']) < 0)/len(eval)}
Test size: {len(test)}
Test clusters: {[c for c in set(test['cluster']) if c<0]}
Test fraction of examples in clusters: {sum(np.array(test['cluster']) < 0)/len(test)}
""")
    data_dict = datasets.DatasetDict({'train': train, 'eval': eval, 'test': test})

    logger.info(f'Final datasets: {data_dict}')
    data_dict.cleanup_cache_files()
    data_dict.save_to_disk('./data/dataset/')
    logger.info("Saved data to disk.")
    
    # get co2
    co2 = data_tracker.stop()

    metrics = {
        'data_co2': float(co2),
        'data_n_train': len(data_dict['train']),
        'data_n_eval': len(data_dict['eval']),
        'data_n_test': len(data_dict['test']),
    }

    # save metrics
    with open('./data/data_metrics.yaml', "w") as stream:
        yaml_dump(metrics, stream)
