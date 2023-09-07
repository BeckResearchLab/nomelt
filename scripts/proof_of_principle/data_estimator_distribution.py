"""Use the AF Estimator to ecaluate training examples, ensure thermophilic protein is more stable on average"""

"""Run optimization on the translated ENH1 sequence."""

import os
import json
from yaml import safe_load
import optuna.samplers
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait
from dask.distributed import get_worker
import codecarbon
import torch
import pandas as pd

import nomelt.thermo_estimation
from nomelt.thermo_estimation.rosetta import RosettaMinimizationParameters

import logging
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

N_PAIRS = 30

def main():
    # start logger
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    utils_logger = logging.getLogger('nomelt')
    utils_logger.setLevel(getattr(logging, LOGLEVEL))
    utils_logger.addHandler(fh)

    tracker = codecarbon.OfflineEmissionsTracker(
        project_name="training_data_estimation",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    # start a dask cluster
    N_GPUS = torch.cuda.device_count()
    cluster = LocalCUDACluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    logger.info(f"Starting cluster with config: {cluster.__dict__}")

    # load the parameters from file
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)

    if not os.path.exists('./data/enh/optimize_enh1/'):
        os.makedirs('./data/enh/optimize_enh1/')
    
    # get estimator class
    estimator_class = getattr(nomelt.thermo_estimation, params['optimize']['estimator'])
    estimator_args_class = params['optimize']['estimator'].split('Estimator')[0] + 'Args'
    estimator_args_class = getattr(nomelt.thermo_estimation, estimator_args_class)
    if hasattr(estimator_args_class, 'rosetta_params'):
        estimator_args_class.rosetta_params = RosettaMinimizationParameters(update_pdb=True)
    if params['optimize']['estimator_args'] is not None:
        estimator_args = estimator_args_class(**params['optimize']['estimator_args'])
    else:
        estimator_args = estimator_args_class()
    estimator_args.wdir = './tmp/training_data_estimation/'
    estimator = estimator_class(args=estimator_args)
    logger.info(f"Using estimator {estimator_class.__name__} with args {estimator_args}.")

    def _worker_function(estimator, sequences, ids):
        logger = logging.getLogger('root')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(LOGFILE, mode='a')
        wid = get_worker().name
        formatter = logging.Formatter(f'WORKER {wid} '+'%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        result = estimator.run(sequences, ids)
        return result
    
    # load sequences from dataset
    df = pd.read_csv('./data/nomelt-model/predictions.tsv', header=None, sep='\t')
    meso_sequences = df[0][:N_PAIRS]
    thermo_sequences = df[1][:N_PAIRS]
    trans_sequences = df[2][:N_PAIRS]
    meso_sequences = [''.join(s.split()) for s in meso_sequences]
    thermo_sequences = [''.join(s.split()) for s in thermo_sequences]
    trans_sequences = [''.join(s.split()) for s in trans_sequences]

    ids = list(range(N_PAIRS))
    meso_ids = [f'meso_{i}' for i in ids]
    thermo_ids = [f'thermo_{i}' for i in ids]
    trans_ids = [f'trans_{i}' for i in ids]

    sequences = meso_sequences + thermo_sequences + trans_sequences
    ids = meso_ids + thermo_ids + trans_ids
    logger.info(f"Got sequences: {sequences}")
    logger.info(f"Got ids: {ids}")
    # break them up into even chunks based on number of workers
    chunk_size = (len(sequences) // N_GPUS) + 1
    if chunk_size == 0:
        chunk_size = 1
    sequences = [sequences[i:i+chunk_size] for i in range(0, len(sequences), chunk_size)]
    ids = [ids[i:i+chunk_size] for i in range(0, len(ids), chunk_size)]
    logger.info(f"Got number of bacthes: {len(sequences)} of size {chunk_size}.")

    # map over workers
    futures = client.map(_worker_function, [estimator]*len(sequences), sequences, ids)
    wait(futures)
    results = [f.result() for f in futures]
    logger.info(f"Got results: {results}")
    # this should be a list of dictionaries of ids. combine the dictionaries and save the output
    results = {k: v for d in results for k, v in d.items()}

    logger.info(f"Complete estimator for {N_PAIRS} pairs.")
    # save the results
    with open(f'./data/proof_of_principle/training_data_estimated.json', 'w') as f:
        json.dump(results, f, indent=4)
    tracker.stop()

if __name__ == "__main__":
    main()


