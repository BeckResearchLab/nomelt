import os
import io
import requests
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import spearmanr, pearsonr
from yaml import safe_load
from nomelt.model import NOMELTModel

def download_and_read_csv(url, csv_name, cache_dir='./tmp/'):
    # Create cache directory if it doesn't exist
    cache_file = os.path.join(cache_dir, csv_name)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if os.path.exists(cache_file):
        print(f"Loading data from cache: {cache_file}")
        return pd.read_csv(cache_file)
    else:
        print(f"Downloading data from {url}")
        response = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        df = pd.read_csv(z.open(csv_name))
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        print(f"Data cached to {cache_file}")
        
        return df

def evaluate_model(model, sequences, temperatures):
    scores = model.score_wts(sequences)
    spearman_corr, spearman_p = spearmanr(temperatures, scores)
    pearson_corr, pearson_p = pearsonr(temperatures, scores)
    return scores, spearman_corr, spearman_p, pearson_corr, pearson_p

def plot_results(temperatures, scores, spearman_corr, pearson_corr, model_name):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=temperatures, y=scores)
    plt.xlabel('Melting Temperature (°C)')
    plt.ylabel(f'{model_name} NOMELT Score')
    plt.title(f'{model_name} Model: NOMELT Score vs Melting Temperature\n'
              f'Spearman R: {spearman_corr:.3f}, Pearson R: {pearson_corr:.3f}')
    plt.savefig(f'./data/plots/{model_name.lower()}_meltome_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load parameters
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)

    # Download and read data
    url = "https://github.com/J-SNACKKB/FLIP/raw/main/splits/meltome/splits.zip"
    df = download_and_read_csv(url, "splits/mixed_split.csv")

    # Initialize models
    hyperparams = params['model']['model_hyperparams']
    original_model = NOMELTModel(params['model']['pretrained_model'], **hyperparams)
    new_model = NOMELTModel('./data/nomelt-model-full/model', **hyperparams)

    sequences = df['sequence'].tolist()
    temperatures = df['target'].tolist()

    results = {}
    for model, model_name in [(original_model, "Original"), (new_model, "New")]:
        scores, spearman_corr, spearman_p, pearson_corr, pearson_p = evaluate_model(model, sequences, temperatures)
        
        results[model_name] = {
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p)
        }
        
        plot_results(temperatures, scores, spearman_corr, pearson_corr, model_name)

    # Save results
    os.makedirs('./data/nomelt-model-full', exist_ok=True)
    with open('./data/nomelt-model-full/meltome_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()