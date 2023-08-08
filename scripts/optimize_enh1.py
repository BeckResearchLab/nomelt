"""Run optimization on the translated ENH1 sequence."""

import os
import json
from yaml import safe_load
import optuna.samplers
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

N_GPUS = torch.cuda.device_count()

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

    # load the parameters from file
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)

    # load the translated ENH1 sequence
    with open('./data/enh/translate_enh1.json', 'r') as f:
        _ = json.load(f)
        enh1_seq = _['original']
        translated_seq = _['generated']

    if not os.path.exists('./data/enh/optimize_enh1'):
        os.makedirs('./data/enh/optimize_enh1')
    
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
        optuna_storage='sqlite:///./data/enh/optimize_enh1/optuna.db',
    )
    optimizer = MutationSubsetOptimizer(
        enh1_seq,
        translated_seq,
        args=optimizer_args,
        name='optimize_enh1',
        wdir='./data/enh/',
        estimator=estimator)
    
    # run the optimization
    optimizer.run(n_jobs=N_GPUS)
    logger.info(f"Ran optimizer with best score {optimizer.best_value} and best sequence {optimizer.best_sequence}.")
    # save the results
    outs = {
        'best_score': optimizer.best_value,
        'best_sequence': optimizer.best_sequence,
        'best_structure': optimizer.optimized_structure_file,
        'wt': optimizer.wt,
        'wt_score': optimizer.initial_targets['wt'],
        'wt_structure': os.path.abspath(f'./tmp/{optimizer.name}/wt.pdb'),
        'translation': optimizer.variant,
        'translation_score': optimizer.initial_targets['variant'],
        'translation_structure': os.path.abspath(f'./tmp/{optimizer.name}/variant.pdb'),
        'trajectory': optimizer.structure_seq_trajectory
    }
    with open(f'./data/enh/optimize_enh1_results.json', 'w') as f:
        json.dump(outs, f, indent=4)
    optimizer.study.trials_dataframe().to_csv(f'./data/enh/optimize_enh1_trials.csv')
    logger.info(f"Saved results to ./data/enh/optimize_enh1_results.json and ./data/enh/optimize_enh1_trials.csv.")
    # make a movie of the optimization

if __name__ == "__main__":
    main()


