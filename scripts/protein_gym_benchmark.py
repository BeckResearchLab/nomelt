"""
Protein Gym original data source for LipA T50:
https://pubs.acs.org/doi/10.1021/acs.jcim.9b00954

"""
# System dependencies
import io
import os
import sys
import json
import re
import logging
import itertools
import tqdm
from yaml import safe_load

# Library dependencies
import duckdb as ddb
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import codecarbon

# Local dependencies
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
    logger.info("Reading data...")
    data_stds = pd.read_excel("./data/gym/ci9b00954_si_002.xlsx", usecols='A:C')
    data_stds.columns = ['mutation', 'dms_mean', 'dms_std']
    data_stds['mutation_shifted'] = data_stds['mutation'].apply(adjust_variant)

    # Designating WT sequence
    wt = "MKFVKRRIIALVTILMLSVTSLFALQPSAKAAEHNPVVMVHGIGGASFNFAGIKSYLVSQGWSRDKLYAVDFWDKTGTNYNNGPVLSRFVQKVLDETGAKKVDIVAHSMGGANTLYYIKNLDGGNKVANVVTLGGANRLTTGKALPGTDPNQKILYTSIYSSADMIVMNYLSRLDGARNVQIHGVGHIGLLYSSQVNSLIKEGLNGGGQNTN"

    # Generate mutated sequences
    logger.info("Generating mutated sequences")
    mutated_sequences_list = generate_mutated_sequences(wt, data_stds)
    data_stds['mutated_sequence'] = mutated_sequences_list
    logger.info(f"Generated {len(mutated_sequences_list)} mutated sequences")

    # Model initialization
    logger.info("Initializing model")
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)
    hyperparams = params['model']['model_hyperparams']
    model = NOMELTModel('./data/nomelt-model-full/model', **hyperparams)
    
    # Score mutated sequences
    logger.info("Scoring mutated sequences")
    wt_score, variant_scores = model.score_variants(wt, mutated_sequences_list, batch_size=5, indels=False)
    data_stds['nomelt_score'] = variant_scores
    logger.info("model scores computed!")

    # Analysis
    logger.info("Analyzing scores...")
    sig_pairs = pairwise_t_test(data_stds)
    logger.info(f"Found {len(sig_pairs)} significant pairs out of {len(data_stds)**2}")

    # get boolean mask for whenever nomelt properly predicts the effect of a mutation wualitatively
    # only for the significant pairs
    mask = []
    for i, j, i_sub_j_dms in sig_pairs:
        i_sub_j_nomelt = data_stds.loc[i, 'nomelt_score'] - data_stds.loc[j, 'nomelt_score']
        mask.append(np.sign(i_sub_j_nomelt) == np.sign(i_sub_j_dms))
    logger.info(f"Found {sum(mask)} out of {len(sig_pairs)} significant pairs where NOMELT qualitatively predicts the effect of the mutation")

    # plotting
    x_data = data_stds['dms_mean']
    y_data = data_stds['nomelt_score']
    # Calculate Pearson and Spearman coefficients and p-values
    pearson_coef, pearson_p = pearsonr(x_data, y_data)
    spearman_coef, spearman_p = spearmanr(x_data, y_data)

    logger.info(f"Pearson correlation coefficient: {pearson_coef}, p-value: {pearson_p}")
    logger.info(f"Spearman correlation coefficient: {spearman_coef}, p-value: {spearman_p}")

    fig, ax= plt.subplots(figsize=(5, 5))
    # create a density plot
    sns.kdeplot(x=x_data, y=y_data, fill=True, bw_adjust=0.5, cmap="Blues", ax=ax)
    # adding a regression line
    m, b = np.polyfit(x_data, y_data, 1)
    x_grid = np.linspace(x_data.min(), x_data.max(), 10)
    # ax.plot(x_grid, m*x_grid + b, color='red', linestyle='--', label='Regression line')
    ax.set_xlabel('DMS score [C]')
    ax.set_ylabel('NOMELT score')

    # annotate with Pearson and Spearman coefficients and p-values
    plt.text(0.05, 0.95, f'Pearson: {pearson_coef:.2f} (p={pearson_p:.2e})\nSpearman: {spearman_coef:.2f} (p={spearman_p:.2e})', 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.5), transform=plt.gca().transAxes)
    plt.savefig('./data/plots/lipa_gym.png', dpi=300, bbox_inches='tight')

    with open('./data/nomelt-model-full/lipa_gym_zero_shot.json', 'w') as f:
        json.dump({'pearson_coef': pearson_coef, 'pearson_p': pearson_p, 'spearman_coef': spearman_coef, 'spearman_p': spearman_p, 'frac_qualitative_sig': sum(mask)/len(sig_pairs)
         }, f)
    # CodeCarbon tracker stop
    tracker.stop()


