"""Prepare protein sequence pairs as a HF dataset
"""
import os
import shutil
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

def run_mmseqs_command(cmd):
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"MMseqs2 command failed with error: {e.stderr}")
        raise e


def run_mmseq(input_fasta, output_dir, params):
    # Database creation
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # delete existing database and remake
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    db_name = os.path.join(output_dir, "mmseq_db")
    run_mmseqs_command(["mmseqs", "createdb", input_fasta, db_name])

    # Clustering
    cluster_out = os.path.join(output_dir, "mmseq_clu")
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    run_mmseqs_command(["mmseqs", "cluster", db_name, cluster_out, tmp_dir, 
                        "--min-seq-id", str(params['min-seq-id']),
                        "-c", str(params['coverage']),
                        "--cov-mode", str(0),
                        "--cluster-mode", str(params['cluster-mode']), "--threads", "32"])

    # Convert to TSV
    tsv_out = os.path.join(output_dir, "mmseq_clu.tsv")
    run_mmseqs_command(["mmseqs", "createtsv", db_name, db_name, cluster_out, tsv_out])

    # Extract sequences
    seq_out = os.path.join(output_dir, "mmseq_clu_seq")
    run_mmseqs_command(["mmseqs", "createseqfiledb", db_name, cluster_out, seq_out])

    # Convert results to flat format
    fasta_out = os.path.join(output_dir, "mmseq_clu_seq.fasta")
    run_mmseqs_command(["mmseqs", "result2flat", db_name, db_name, seq_out, fasta_out])
    
    return tsv_out, fasta_out

def parse_mmseq(file):
    results = pd.read_csv(file, sep='\t', header=None)
    results.columns = ['clust', 'member']
    cluster_dict = results.set_index('member')['clust'].to_dict()
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
    logger.info('Labeling clusters using mmseqs')
    # Prepare FASTA file from ds
    with open("./tmp/mmseqs_input.fasta", "w") as f:
        for idx, item in enumerate(ds):
            f.write(f">{idx}\n{item['meso_seq']}\n")

    # Run mmseq
    mmseq_params = params['data']['mmseq_params']
    t0 = time.time()
    tsv_out, fasta_out = run_mmseq(
        "./tmp/mmseqs_input.fasta",
        "./tmp/mmseqs_outs",
        mmseq_params
    )

    # Parse mmseq output
    logger.info('Parsing mmseq output')
    clusters = parse_mmseq(tsv_out)
    t1 = time.time()
    logger.info(f"Minutes elapsed: {(t1-t0)/60}")

    # Add mmseq cluster IDs to ds
    def add_cdhit_cluster(example, idx):
        example['cluster'] = clusters.get(idx, -1)
        return example

    logger.info('Adding mmseq cluster IDs to dataset')
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
Train clusters: {[c for c in set(train['cluster'])]}
Eval size: {len(eval)}
Eval clusters: {[c for c in set(eval['cluster'])]}
Test size: {len(test)}
Test clusters: {[c for c in set(test['cluster'])]}
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
