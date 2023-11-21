"""
TODO

Things to do:
- [ ] fix slurm job id venv variable for codecarbon
- [x] utilize cuda for model scoring (already in function)
"""
# system dependencies
import io
import os
import sys
import re
import logging
from yaml import safe_load

# library dependencies
import duckdb as ddb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import codecarbon

# local dependencies
from nomelt.model import NOMELTModel

# seaborn settings
sns.set_context("talk")
sns.set_style("ticks")

##########################
## environment variables #
##########################
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'DEBUG'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

# obtain slurm job id for codecarbon/logging
os.environ['SLURM_JOBID'] = os.environ['SLURM_JOB_ID']

################
## constants ###
################
ANALYSIS_DIR = "./data/gym/analysis/"

################
## functions ###
################

def generate_mutated_sequences(wt_sequence: str, mutation_table_path: str) -> list:
    """
    Generate a list of mutated sequences from a mutation table, each with a single mutation applied to the wild-type.

    Parameters:
    - wt_sequence: The wild-type protein sequence.
    - mutation_table_path: Path to the CSV containing the mutation table.

    Returns:
    - List of mutated protein sequences.
    """
    # Load the CSV
    mutation_df = pd.read_csv(mutation_table_path)

    # Extract mutations
    mutations = mutation_df["mutant"]

    # List to store all mutated sequences
    all_mutated_sequences = []

    # Apply each mutation
    for mutation in mutations:
        # Create a fresh copy of wt for each mutation
        mutated_sequence = list(wt_sequence)
        
        match = re.match(r'([A-Z]+)(\d+)([A-Z]+)', mutation)
        if not match:
            continue # Skip any mutations that don't match the expected format
        original_aa, position, new_aa = match.groups() # Return a part of the string where there was a match
        position = int(position) - 1 # Convert to 0-based index

        # Check if the current amino acid matches the expected amino acid from the mutation table
        if mutated_sequence[position] != original_aa:
            print(f"Discrepancy at position {position+1}: Expected {original_aa}, Found {mutated_sequence[position]}, Mutation Table suggests changing to {new_aa}.")
            continue

        # Apply the mutation
        mutated_sequence[position] = new_aa

        # Append to the list of all mutated sequences
        all_mutated_sequences.append(''.join(mutated_sequence))

    return all_mutated_sequences


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
    output_dir=ANALYSIS_DIR,
    country_iso_code="USA",
    region="washington")

    tracker.start()

    # code
    logger.info("Designating WT sequence...")
    wt = "MKFVKRRIIALVTILMLSVTSLFALQPSAKAAEHNPVVMVHGIGGASFNFAGIKSYLVSQGWSRDKLYAVDFWDKTGTNYNNGPVLSRFVQKVLDETGAKKVDIVAHSMGGANTLYYIKNLDGGNKVANVVTLGGANRLTTGKALPGTDPNQKILYTSIYSSADMIVMNYLSRLDGARNVQIHGVGHIGLLYSSQVNSLIKEGLNGGGQNTN"

    # generate mutated sequences
    logger.info("Generating mutated sequences") 
    mutated_sequences_list = generate_mutated_sequences(wt, "./data/gym/ESTA_BACSU_Nutschel_2020.csv")
    logger.info(f"Generated {len(mutated_sequences_list)} mutated sequences")
    logger.debug(f"Sample mutated sequences: {mutated_sequences_list[:3]}")

    # initialize model
    logger.info("Initializing model")

    # load params
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)
    
    # obtain model hyperparams
    hyperparams = params['model']['model_hyperparams']
    logger.debug(f"Model hyperparams: {hyperparams}")
    
    logger.info("Loading model...")
    model = NOMELTModel('./data/nomelt-model/model', **hyperparams)
    logger.info("Model loaded!")
    
    # make into an iterable array
    all_mutated_sequences = pd.Series(mutated_sequences_list).values
    logger.debug("Mutated sequences converted to iterable array")

    # score mutated sequences
    logger.info("Scoring mutated sequences")
    wt_score, variant_scores = model.score_variants(wt, all_mutated_sequences, batch_size=5, indels=False)
    logger.info("model scores computed!")

    # stop codecarbon
    tracker.stop()
    logger.info("CodeCarbon stopped!")

    # plotting
    logger.info("Plotting...")
    data = pd.read_csv("./data/gym/ESTA_BACSU_Nutschel_2020.csv")

    # set data
    x_data = data['DMS_score']
    y_data = variant_scores

    # set figure
    fig, ax= plt.subplots(figsize=(10, 10))

    # plot
    sns.regplot(x=x_data, y=y_data)
    ax.set_title('Figure 1: model vs DMS (thermostability))')
    ax.set_xlabel('DMS score (exp)')
    ax.set_ylabel('Model score')
    plt.savefig("./data/gym/analysis/figure1.png")


