"""
Script from Evan's Claude
"""
import io
import os
import sys
import json
import re
import logging
import itertools
import tqdm
from yaml import safe_load

 

import duckdb as ddb
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import codecarbon

from nomelt.model import NOMELTModel


# Seaborn settings
sns.set_context("talk")
sns.set_style("whitegrid")
 

# Environment variables and logger setup
logger = logging.getLogger(__name__)
LOGLEVEL = os.getenv('LOGLEVEL', 'DEBUG')
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'
os.environ['SLURM_JOBID'] = os.environ.get('SLURM_JOB_ID', '')


# Constants
ANALYSIS_DIR = "./data/gym/analysis/"
 

# Functions definitions
def generate_mutated_sequences(wt_sequence: str, mutation_df: pd.DataFrame) -> list:
    """
    Generate a list of mutated sequences from a mutation table DataFrame, each with a single mutation applied to the wild-type.
    Parameters:
    - wt_sequence: The wild-type protein sequence.
    - mutation_df: DataFrame containing the mutation table.

    Returns:
    - List of mutated protein sequences.
    """
    # Extract mutations
    mutations = mutation_df["mutation_shifted"]

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


def pairwise_t_test(df, alpha=0.05):
    """
    Perform pairwise t-tests on a dataframe of measurements.

    Parameters:
    - df: Dataframe of measurements.
    - alpha: Significance level.

    Returns:
    - List of tuples of significant pairs and their differences.
    """
    # Extract measurements
    measurements = df[['dms_mean', 'dms_std']]

    # Create combinations of all pairs
    pairs = list(itertools.combinations(measurements.index, 2))

    # Store significant pairs
    significant_pairs = []

    for i, j in tqdm.tqdm(pairs, desc="Performing pairwise t-tests"):
        mean_i, std_i = measurements.loc[i].values
        mean_j, std_j = measurements.loc[j].values

        # Calculate t-statistic and p-value
        t_stat, p_val = stats.ttest_ind_from_stats(mean1=mean_i, std1=std_i, nobs1=3,
                                                   mean2=mean_j, std2=std_j, nobs2=3)

        # Check if the difference is significant
        if p_val < alpha:

            significant_pairs.append((i, j, mean_i - mean_j))
    return significant_pairs

 

# Adjusting the 'Variants of BsLipA' to match the 'mutant' format by adding 31 to the numeric part
# Function to adjust the numeric part of the variant names
# this is because the authors reported the mutations starting from a specific region
# of LipA instead of the start of the uniprot seq
def adjust_variant(variant, offset=31):
    parts = re.search(r'([A-Za-z]+)(\d+)([A-Za-z]+)', variant)
    if parts:
        return f"{parts.group(1)}{int(parts.group(2)) + offset}{parts.group(3)}"
    else:
        return variant

 

def load_and_process_data(file_path, sheet_name=0):
    """Load and process data from an Excel file."""
    data = pd.read_excel(file_path, sheet_name=sheet_name, usecols='A:C')
    data.columns = ['mutation', 'dms_mean', 'dms_std']
    data['mutation_shifted'] = data['mutation'].apply(adjust_variant)
    return data

 

def evaluate_model(model, wt, data, benchmark_name):
    """Evaluate the model on a given dataset and return metrics."""
    mutated_sequences_list = generate_mutated_sequences(wt, data)
    data['mutated_sequence'] = mutated_sequences_list

    wt_score, variant_scores = model.score_variants(wt, mutated_sequences_list, batch_size=5, indels=False)
    data['nomelt_score'] = variant_scores

    sig_pairs = pairwise_t_test(data)
    mask = []
    for i, j, i_sub_j_dms in sig_pairs:
        i_sub_j_nomelt = data.loc[i, 'nomelt_score'] - data.loc[j, 'nomelt_score']
        mask.append(np.sign(i_sub_j_nomelt) == np.sign(i_sub_j_dms))

    x_data = data['dms_mean']
    y_data = data['nomelt_score']
    pearson_coef, pearson_p = pearsonr(x_data, y_data)
    spearman_coef, spearman_p = spearmanr(x_data, y_data)

    plot_results(x_data, y_data, pearson_coef, pearson_p, spearman_coef, spearman_p, f'./data/plots/{benchmark_name}.png')

    return {
        'pearson_coef': pearson_coef,
        'pearson_p': pearson_p,
        'spearman_coef': spearman_coef,
        'spearman_p': spearman_p,
        'frac_qualitative_sig': sum(mask)/len(sig_pairs)
    }

 

def plot_results(x_data, y_data, pearson_coef, pearson_p, spearman_coef, spearman_p, filename):
    """Plot and save the results."""
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.kdeplot(x=x_data, y=y_data, fill=True, bw_adjust=0.5, cmap="Blues", ax=ax)
    ax.set_xlabel('DMS score [C]')
    ax.set_ylabel('NOMELT score')
    plt.text(0.05, 0.95, f'Pearson: {pearson_coef:.2f} (p={pearson_p:.2e})\nSpearman: {spearman_coef:.2f} (p={spearman_p:.2e})',
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.5), transform=plt.gca().transAxes)
    plt.savefig(filename, dpi=300, bbox_inches='tight')

 

if __name__ == "__main__":
    # Logger initialization
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    utils_logger = logging.getLogger('nomelt')
    utils_logger.setLevel(getattr(logging, LOGLEVEL))
    utils_logger.addHandler(fh)

 

    # initialize codecarbon
    tracker = codecarbon.OfflineEmissionsTracker(
        project_name="estimate_gym",
        output_dir='./data',
        country_iso_code="USA",
        region="washington")

    tracker.start()

    # Data reading and processing 
    # Load data for both benchmarks
    data_lipa = load_and_process_data("./data/gym/ci9b00954_si_002.xlsx")
    # data_second = load_and_process_data("./path/to/second_benchmark.xlsx")  # Replace with actual path

    # Designating WT sequence (assuming it's the same for both benchmarks, adjust if needed)
    wt = "MKFVKRRIIALVTILMLSVTSLFALQPSAKAAEHNPVVMVHGIGGASFNFAGIKSYLVSQGWSRDKLYAVDFWDKTGTNYNNGPVLSRFVQKVLDETGAKKVDIVAHSMGGANTLYYIKNLDGGNKVANVVTLGGANRLTTGKALPGTDPNQKILYTSIYSSADMIVMNYLSRLDGARNVQIHGVGHIGLLYSSQVNSLIKEGLNGGGQNTN"

 

    # Model initialization for new and original weights
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)

    hyperparams = params['model']['model_hyperparams']
    new_model = NOMELTModel('./data/nomelt-model-full/model', **hyperparams)
    original_model = NOMELTModel(params['model']['pretrained_model'], **hyperparams)

    # Evaluate both models on both datasets
    results = {}

    for model_name, model in [("new", new_model), ("original", original_model)]:
        results[model_name] = {}
        for benchmark_name, data in [("lipa", data_lipa)]: # , ("second_benchmark", data_second)
            results[model_name][benchmark_name] = evaluate_model(model, wt, data, f"{model_name}_{benchmark_name}")

    # Save results
    with open('./data/nomelt-model-full/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
 
    # CodeCarbon tracker stop
    tracker.stop()