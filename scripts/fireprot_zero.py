import os
import requests
import json
import logging
import io
import re
from typing import List, Tuple

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from yaml import safe_load
import codecarbon

from nomelt.model import NOMELTModel

# Seaborn settings
sns.set_context("talk")
sns.set_style("whitegrid")

# Logger setup
logger = logging.getLogger(__name__)
LOGLEVEL = os.getenv('LOGLEVEL', 'INFO')
LOGFILE = f'./logs/fireprotdb_benchmark.log'

def setup_logging():
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def download_fireprotdb_data(url: str, output_file: str) -> pd.DataFrame:
    if os.path.exists(output_file):
        logger.info(f"Loading FireProtDB data from file: {output_file}")
        return pd.read_csv(output_file)
    
    logger.info(f"Downloading FireProtDB data from {url}")
    payload = {"searchData": {"type": "expr", "key": "all", "value": " ", "checkOptions": []}, "filter": {"filterKey": "ddG", "order": "asc"}}
    response = requests.post(url, json=payload)
    content = response.content.decode('utf-8')
    df = pd.read_csv(io.StringIO(content))
    df.to_csv(output_file, index=False)
    return df

def process_fireprotdb_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wt_data = {}
    variant_data = []

    for _, row in df.iterrows():
        uniprot_id = row['uniprot_id']
        if uniprot_id not in wt_data:
            wt_data[uniprot_id] = {
                'sequence': row['sequence'],
                'tm': row['tm'] - row['dTm']  # Calculate wildtype Tm
            }
        
        if row['sequence'][row['position']-1] == row['wild_type']:
            variant_data.append({
                'uniprot_id': uniprot_id,
                'wildtype_sequence': row['sequence'],
                'mutation': f"{row['wild_type']}{row['position']}{row['mutation']}",
                'delta_tm': row['dTm']
            })

    wt_df = pd.DataFrame.from_dict(wt_data, orient='index')
    wt_df.reset_index(inplace=True)
    wt_df.rename(columns={'index': 'uniprot_id'}, inplace=True)

    variant_df = pd.DataFrame(variant_data)
    
    return wt_df, variant_df

def apply_mutation(sequence: str, mutation: str) -> str:
    match = re.match(r'([A-Z])(\d+)([A-Z])', mutation)
    if not match:
        raise ValueError(f"Invalid mutation format: {mutation}")
    wt, pos, mut = match.groups()
    pos = int(pos) - 1
    if sequence[pos] != wt:
        raise ValueError(f"Mutation {mutation} does not match sequence at position {pos+1}")
    return sequence[:pos] + mut + sequence[pos+1:]

def evaluate_model(model: NOMELTModel, wt_df: pd.DataFrame, variant_df: pd.DataFrame) -> dict:
    results = []
    for _, wt_row in tqdm(wt_df.iterrows(), total=len(wt_df), desc="Evaluating proteins"):
        wt_sequence = wt_row['sequence']
        variants = variant_df[variant_df['uniprot_id'] == wt_row['uniprot_id']]
        
        if len(variants) == 0:
            continue

        mutated_sequences = [apply_mutation(wt_sequence, mut) for mut in variants['mutation']]
        wt_score, variant_scores = model.score_variants(wt_sequence, mutated_sequences, batch_size=5, indels=False)
        
        for _, variant, score in zip(variants.itertuples(), variants['delta_tm'], variant_scores):
            results.append({
                'uniprot_id': variant.uniprot_id,
                'mutation': variant.mutation,
                'delta_tm': variant.delta_tm,
                'nomelt_score': score
            })

    results_df = pd.DataFrame(results)
    
    x_data = results_df['delta_tm']
    y_data = results_df['nomelt_score']
    
    spearman_coef, spearman_p = spearmanr(x_data, y_data)
    auroc = roc_auc_score((x_data > 0).astype(int), y_data)

    return {
        'spearman_coef': spearman_coef,
        'spearman_p': spearman_p,
        'auroc': auroc,
        'results_df': results_df
    }

def plot_results(results: dict, model_name: str, output_dir: str):
    x_data = results['results_df']['delta_tm']
    y_data = results['results_df']['nomelt_score']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.kdeplot(x=x_data, y=y_data, fill=True, cmap="YlGnBu", ax=ax)
    ax.set_xlabel('ΔTm [°C]')
    ax.set_ylabel('NOMELT score')
    ax.set_title(f'{model_name} Model Performance on FireProtDB')
    
    textstr = f"Spearman: {results['spearman_coef']:.3f} (p={results['spearman_p']:.2e})\nAUROC: {results['auroc']:.3f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fireprotdb_{model_name.lower()}_performance.png'), dpi=300)
    plt.close()

def main():
    setup_logging()

    # Initialize codecarbon
    tracker = codecarbon.OfflineEmissionsTracker(
        project_name="fireprotdb_benchmark",
        output_dir='./data',
        country_iso_code="USA",
        region="washington")
    tracker.start()

    # Load parameters
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)

    hyperparams = params['model']['model_hyperparams']

    # Download and process data
    url = "https://loschmidt.chemi.muni.cz/fireprotdb/v1/export?searchType=advanced&type=csv"
    output_file = "./data/fireprotdb_data.csv"
    df = download_fireprotdb_data(url, output_file)
    wt_df, variant_df = process_fireprotdb_data(df)

    # Initialize models
    new_model = NOMELTModel('./data/nomelt-model-full/model', **hyperparams)
    original_model = NOMELTModel(params['model']['pretrained_model'], **hyperparams)

    # Evaluate models
    results = {}
    for model_name, model in [("New", new_model), ("Original", original_model)]:
        logger.info(f"Evaluating {model_name} model")
        results[model_name] = evaluate_model(model, wt_df, variant_df)
        plot_results(results[model_name], model_name, './data/plots')

    # Save results
    with open('./data/nomelt-model-full/fireprotdb_benchmark_results.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'results_df'} for k, v in results.items()}, f, indent=2)

    # Save detailed results
    for model_name, result in results.items():
        result['results_df'].to_csv(f'./data/nomelt-model-full/fireprotdb_{model_name.lower()}_detailed_results.csv', index=False)

    logger.info("Benchmark completed. Results saved.")
    tracker.stop()

if __name__ == "__main__":
    main()
