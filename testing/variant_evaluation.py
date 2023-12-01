"""
TODO


Things to do:
- [ ] fix slurm job id venv variable for codecarbon
- [x] utilize cuda for model scoring (already in function)

References:
- Nutschel, C., et al. (2020). https://pubs.acs.org/doi/10.1021/acs.jcim.9b00954
"""
# system dependencies
import io
import os
import sys
import re
import logging
import itertools
from yaml import safe_load

# library dependencies
import duckdb as ddb
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
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
    measurements = df[['Mapped_DMS', 'STDEV']]
    
    # Create combinations of all pairs
    pairs = list(itertools.combinations(measurements.index, 2))

    # Store significant pairs
    significant_pairs = []

    for i, j in pairs:
        mean_i, std_i = df.loc[i, ['Mapped_DMS', 'STDEV']]
        mean_j, std_j = df.loc[j, ['Mapped_DMS', 'STDEV']]
        # Calculate t-statistic and p-value
        t_stat, p_val = stats.ttest_ind_from_stats(mean1=mean_i, std1=std_i, nobs1=3,
                                                   mean2=mean_j, std2=std_j, nobs2=3)

        # Check if the difference is significant
        if p_val < alpha:
            significant_pairs.append((i, j, mean_i - mean_j))

    return significant_pairs

# Adjusting the 'Variants of BsLipA' to match the 'mutant' format by adding 31 to the numeric part

# Function to adjust the numeric part of the variant names
def adjust_variant(variant, offset=31):
    parts = re.search(r'([A-Za-z]+)(\d+)([A-Za-z]+)', variant)
    if parts:
        return f"{parts.group(1)}{int(parts.group(2)) + offset}{parts.group(3)}"
    else:
        return variant


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

    logger.info("Reading data...")
    data_dms = pd.read_csv("./data/gym/ESTA_BACSU_Nutschel_2020.csv")
    data_stds = pd.read_excel("./data/gym/ci9b00954_si_002.xlsx")

    logger.info('Mapping DMS scores with STDEVs...')
    # Mapping 'T50' to 'STDEV' in Excel
    t50_to_stdev_mapping = data_stds.set_index('T50')['STDEV'].to_dict()

    # Apply this function to 'Variants of BsLipA'
    data_stds['mutant'] = data_stds['Variants of BsLipA'].apply(lambda x: adjust_variant(x))

    # Now map 'Adjusted_Variant' to 'mutant' and then map 'STDEV'
    data_stds['Mapped_DMS'] = data_stds['mutant'].map(data_dms.set_index('mutant')['DMS_score'].to_dict())
    # data_stds['Mapped_STDEV'] = data_stds['T50'].map(t50_to_stdev_mapping)

    # Display the first few rows of the Excel dataset with the newly mapped columns
    data_stds[['Variants of BsLipA', 'mutant', 'Mapped_DMS', 'STDEV']].head()

    logger.debug('DMS scores mapped with STDEVs!')
    logger.debug(f"Sample data: {data_stds.head()}")

    # generate mutated sequences
    logger.info("Generating mutated sequences") 
    mutated_sequences_list = generate_mutated_sequences(wt, data_stds)
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
    model = NOMELTModel('./data/nomelt-model-full/model', **hyperparams)
    logger.info("Model loaded!")
    
    # make into an iterable array
    all_mutated_sequences = pd.Series(mutated_sequences_list).values
    logger.debug("Mutated sequences converted to iterable array")

    # score mutated sequences
    logger.info("Scoring mutated sequences")
    wt_score, variant_scores = model.score_variants(wt, all_mutated_sequences, batch_size=5, indels=False)
    logger.info("model scores computed!")

    # analyzing scores
    logger.info("Analyzing scores...")


    logger.info('Calculating mean and std of DMS scores...')
    # Calculate mean and std of DMS scores
    sig_pairs = pairwise_t_test(data_stds)
    logger.debug(f"Significant pairs: {sig_pairs}")

    # stop codecarbon
    tracker.stop()
    logger.info("CodeCarbon stopped!")

    # plotting
    logger.info("Plotting...")
    data = pd.read_csv("./data/gym/ESTA_BACSU_Nutschel_2020.csv")

    logger.info("Plotting figure 1, a scatterplot of model score vs DMS score")
    # set data
    x_data = data['DMS_score']
    y_data = variant_scores

    # Calculate Pearson and Spearman coefficients and p-values
    pearson_coef, pearson_p = pearsonr(x_data, y_data)
    spearman_coef, spearman_p = spearmanr(x_data, y_data)

    logger.info(f"Pearson correlation coefficient: {pearson_coef}, p-value: {pearson_p}")
    logger.info(f"Spearman correlation coefficient: {spearman_coef}, p-value: {spearman_p}")

    # set figure
    fig, ax= plt.subplots(figsize=(10, 10))

    # plot
    sns.regplot(x=x_data, y=y_data, ax=ax)
    ax.set_title('Figure 1: model vs DMS (thermostability)')
    ax.set_xlabel('DMS score (exp)')
    ax.set_ylabel('Model score')
    # annotate with Pearson and Spearman coefficients
    ax.text(0.05, 0.95, f'Pearson: {pearson_coef:.2f}, p: {pearson_p:.2g}\nSpearman: {spearman_coef:.2f}, p: {spearman_p:.2g}', 
        transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.5))

    plt.savefig("./data/gym/analysis/figure1.png")
    logger.info("Figure 1 saved!")

    logger.info("Figure 2: density plot")
    # convert to dataframe
    protein_data = pd.DataFrame({'DMS_score': x_data, 'model_score': y_data})

    # For single-variable density plot
    plt.figure(figsize=(10, 10))  # Create a new figure
    sns.kdeplot(data=protein_data['DMS_score'], color="red", fill=True, bw_adjust=0.5)
    plt.title('Density Plot of DMS Scores')
    plt.xlabel('DMS score (exp)')
    plt.savefig("./data/gym/analysis/figure2a.png")

    plt.figure(figsize=(10, 10))  # Create a new figure
    sns.kdeplot(data=protein_data['model_score'], color="blue", fill=True, bw_adjust=0.5)
    plt.title('Density Plot of Model Scores')
    plt.xlabel('Model score')
    plt.savefig("./data/gym/analysis/figure2b.png")
    logger.info("Figure 2a and 2b saved!")

    # For two-variable density plot
    logger.info("Plotting figure 3, a density plot of model score vs DMS score")
    plt.figure(figsize=(10, 10))  # Create a new figure
    
    # create a density plot
    sns.kdeplot(data=protein_data, x="DMS_score", y="model_score", fill=True, bw_adjust=0.5, cmap="Blues")
    # adding a regression line
    sns.regplot(data=protein_data, x='DMS_score', y='model_score', scatter=False, color='red')

    # annotate with Pearson and Spearman coefficients and p-values
    plt.text(0.05, 0.95, f'Pearson: {pearson_coef:.2f} (p={pearson_p:.2e})\nSpearman: {spearman_coef:.2f} (p={spearman_p:.2e})', 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.5), transform=plt.gca().transAxes)

    plt.title('Density Plot of DMS vs Model Scores with Regression Line')
    plt.xlabel('DMS score (exp)')
    plt.ylabel('Model score')
    plt.savefig("./data/gym/analysis/figure3.png")
    logger.info("Figure 3 saved!")


