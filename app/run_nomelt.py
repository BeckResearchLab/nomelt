import os
import argparse
import yaml
import json
import optuna
import torch
import logging
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None})
from nomelt.model import NOMELTModel

logger = logging.getLogger(__name__)
N_GPUS = torch.cuda.device_count()

def init_optimizer(sequence, trans_seq, output_dir, config):
    from nomelt.thermo_estimation.optimizer import MutationSubsetOptimizer, OptimizerArgs
    import nomelt.thermo_estimation
    from nomelt.thermo_estimation.rosetta import RosettaMinimizationParameters
    estimator_class = getattr(nomelt.thermo_estimation, config['estimator'])
    estimator_args_class = config['estimator'].split('Estimator')[0] + 'Args'
    estimator_args_class = getattr(nomelt.thermo_estimation, estimator_args_class)
    if hasattr(estimator_args_class, 'rosetta_params'):
        estimator_args_class.rosetta_params = RosettaMinimizationParameters(update_pdb=True)
    if config['estimator_args'] is not None: 
        estimator_args = estimator_args_class(**config['estimator_args'])
    else:
        estimator_args = estimator_args_class()
    estimator = estimator_class(args=estimator_args)

    optimizer_args = OptimizerArgs(
        n_trials=config['optimizer_args']['n_trials'],
        direction=config['optimizer_args']['direction'],
        sampler=getattr(optuna.samplers, config['optimizer_args']['sampler'])(**config['optimizer_args']['sampler_args']),
        cut_tails=config['optimizer_args']['cut_tails'],
        gap_compressed_mutations=config['optimizer_args']['gap_compressed_mutations'],
        matrix=config['optimizer_args']['matrix'],
        gapopen=config['optimizer_args']['gapopen'],
        gapextend=config['optimizer_args']['gapextend'],
        match_score=config['optimizer_args']['match_score'],
        mismatch_score=config['optimizer_args']['mismatch_score'],
        penalize_end_gaps=config['optimizer_args']['penalize_end_gaps'],
        optuna_storage=os.path.join(output_dir, "optuna.log"),
        optuna_overwrite=config['optimizer_args']['optuna_overwrite'],
        measure_initial_structures=True
    )
    optimizer = MutationSubsetOptimizer(
        sequence,
        trans_seq,
        args=optimizer_args,
        name=f'optimize_target',
        wdir=output_dir,
        estimator=estimator
    )
    return optimizer

def run_in_silico_optimization(optimizer):
    cluster = LocalCUDACluster(n_workers=N_GPUS, threads_per_worker=1)
    client = Client(cluster)
    optimizer.run(n_jobs=N_GPUS, client=client)
    return optimizer

def main(args):
    # Load the configuration
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    sequence = args.input
    output_dir = args.output_dir
    # check that the output exists and is a dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = args.model_path
    # create a log file in the output directory
    nomelt_logger = logging.getLogger('nomelt')
    log_file = os.path.join(output_dir, "run_nomelt.log")
    logger.setLevel(logging.INFO)
    nomelt_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    nomelt_logger.addHandler(fh)

    model = NOMELTModel(model_path, **config['model']['hyperparams'])

    if config['zero_shot_ranking']['enabled']:
        logger.info('Doing zero shot, ignoring generation options')

        with open(sequence, 'r') as f:
            sequences = f.readlines()
        sequences = [seq.strip() for seq in sequences]
        wild_type = sequences[0]
        sequences = sequences[1:]
        wt_score, variant_scores = model.score_variants(wt=wild_type, variants=sequences, indels=config['zero_shot_ranking']['indels']) # this is just a list of scores
        scores_file = os.path.join(output_dir, "zero_shot_scores.csv")
        logger.info(f"Zero shot scoring complete. Writing to {scores_file}")
        with open(scores_file, 'w') as f:
            f.write("\n".join([str(wt_score)]+[str(score) for score in variant_scores]))
        return
    
    if config['beam_search']['enabled']:
        logger.info("Running beam search translation...")
        out_sequence = model.translate_sequences(
            [sequence],
            generation_max_length=config['beam_search']['generation_max_length'],
            generation_num_beams=config['beam_search']['generation_num_beams'],
        )[0]['sequences'][0]
        seq_out_file = os.path.join(output_dir, "beam_search_sequence.txt")
        logger.info(f"Beam search translation complete. Writing to {seq_out_file}")
        with open(seq_out_file, 'w') as f:
            f.write(out_sequence)

    if config['stochastic']['enabled']:
        sequences = []

        while len(sequences) < config['stochastic']['generation_ensemble_size']:
            sequences.extend(model.translate_sequences(
                    [sequence],
                    generation_max_length=config['beam_search']['generation_max_length'],
                    temperature=config['stochastic']['temperature'],
                    generation_ensemble_size=config['stochastic']['generation_ensemble_size'],
                    generation_num_beams=None
                )[0]['sequences']
            )
        seq_out_file = os.path.join(output_dir, "stochastic_sequences.txt")
        logger.info(f"Stochastic sequence generation complete. Writing to {seq_out_file}")
        with open(seq_out_file, 'w') as f:
            f.write("\n".join(sequences))

    if not config['optimization']['enabled'] and not config['output_library']['enabled']:
        logger.info("No optimization or output library enabled. Exiting.")
        return

    # init the optimizer
    optimizer = init_optimizer(sequence, out_sequence, output_dir, config['optimization'])

    # Placeholder for output permutations library
    if config['output_library']['enabled']:
        logger.info("Outputting library...")
        library = optimizer.all_permutations() # this is an iterator
        library_file = os.path.join(output_dir, "library.txt")
        logger.info(f"Writing library to {library_file}")
        with open(library_file, 'w') as f:
            for seq in library:
                f.write(seq + "\n")

    if config['optimization']['enabled']:
        logger.info("Running in silico optimization...")
        run_in_silico_optimization(optimizer)
        logger.info(f"In silico optimization complete. Writing final sequence and trials to {output_dir}")
        # Output the trials
        trials_file = os.path.join(output_dir, "trials.csv")
        df = optimizer.trials_dataframe
        df.to_csv(trials_file, index=False)
        # Output the final sequence
        final_sequence_file = os.path.join(output_dir, "optimize_results.json")
        results = {
            'optimized_sequence': optimizer.best_trial.user_attrs['variant_seq'],
            'optimized_structure': optimizer.best_trial.user_attrs['pdb_file'],
            'initial_scores': optimizer.initial_targets
        }
        with open(final_sequence_file, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a pipeline based on a sequence and configuration file.")
    parser.add_argument("input", type=str, help="Input sequence (str) or library (text file).")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("model_path", type=str, help="Path to the NOMELT model directory.")
    parser.add_argument("config_file", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()

    main(args)
