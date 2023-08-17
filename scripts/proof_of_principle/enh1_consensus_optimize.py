"""Run optimization on the translated ENH1 sequence."""

import os
import json
from yaml import safe_load
import optuna.samplers
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import codecarbon
import torch

from nomelt.thermo_estimation.optimizer import MutationSubsetOptimizer, OptimizerArgs, OptTrajSuperimposer
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
        project_name="enh1_consensus_optimize",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    # start a dask cluster
    N_GPUS = torch.cuda.device_count()
    cluster = LocalCUDACluster(n_workers=N_GPUS, threads_per_worker=1)
    client = Client(cluster)
    logger.info(f"Starting cluster with config: {cluster.__dict__}")

    # load the parameters from file
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)

    if not os.path.exists('./tmp/optimize_consensus'):
        os.makedirs('./tmp/optimize_consensus')
    
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
    estimator = estimator_class(args=estimator_args)
    logger.info(f"Using estimator {estimator_class.__name__} with args {estimator_args}.")

    # set up the optimizer
    optimizer_args = OptimizerArgs(
        n_trials=params['optimize']['n_trials'],
        direction=params['optimize']['direction'],
        sampler=getattr(optuna.samplers, params['optimize']['sampler'])(**params['optimize']['sampler_args']),
        cut_tails=params['optimize']['cut_tails'],
        gap_compressed_mutations=params['optimize']['gap_compressed_mutations'],
        matrix=params['optimize']['matrix'],
        gapopen=params['optimize']['gapopen'],
        gapextend=params['optimize']['gapextend'],
        match_score=params['optimize']['match_score'],
        mismatch_score=params['optimize']['mismatch_score'],
        penalize_end_gaps=params['optimize']['penalize_end_gaps'],
        optuna_storage='./tmp/optimize_consensus/optuna.log',
        optuna_overwrite=params['optimize']['optuna_overwrite'],
        measure_initial_structures=False
    )
    logger.info(f"Optimizer args: {optimizer_args}")
    optimizer = MutationSubsetOptimizer(
        "DKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKK",
        "RRKRTTFTKEQLEELEELFEKNRYPSAEEREELAKKLGLTERQVKVWFQNRRAKEKK",
        args=optimizer_args,
        name='enh_vs_consensus',
        wdir='./tmp/optimize_consensus/',
        estimator=estimator)
    
    # run the optimization
    optimizer.run(n_jobs=N_GPUS, client=client)
    logger.info(f"Ran optimizer with best trial {optimizer.best_trial}.")
    # save the results
    outs = {
        'best_score': optimizer.best_trial.value,
        'best_sequence': optimizer.best_trial.user_attrs['variant_seq'],
        'best_structure': optimizer.best_trial.user_attrs['pdb_file'],
    }
    with open(f'./data/proof_of_principle/optimize_enh1_cons_results.json', 'w') as f:
        json.dump(outs, f, indent=4)
    optimizer.study.trials_dataframe().to_csv(f'./data/proof_of_principle/optimize_enh1_cons_trials.csv')
    logger.info("Saved results.")
    tracker.stop()

if __name__ == "__main__":
    main()


