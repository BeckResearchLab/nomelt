import codecarbon
import multiprocessing as mp
import logging
import numpy as np
import pandas as pd
import os
import pprint
import json
from tqdm import tqdm
from joblib import Parallel, delayed

from evaluate import load

import logging
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

# define metrics to compute
def compute_metrics(inputs):
    """We use:
    - ter: translation edit rate
    - Rouge 2: F1 score for bigrams
    - Rouge L: score for longest common subsequences
    - google_bleu: single sentence BLEU like score, minimum of recall and precision on 1, 2, 3, and 4 grams
    """
    predictions, labels = inputs
    # outputs encoded
    length_differences = []
    for p, l in zip(predictions, labels):
        length_differences.append((len(p.split()) - len(l.split()))/len(l.split()))
    # outputs are list of strings, with spaces ## CHECK
    out_metrics = {'len_diff_frac': np.mean(length_differences)}
    out_metrics['len_diff_frac_abs']= np.mean(np.abs(length_differences))
    out_metrics['len_diff_frac_std']=np.std(length_differences)
    # tranlsation error rate
    ter_metric = load('ter')
    out_metrics.update(ter_metric.compute(predictions=predictions, references=labels, normalized=True, case_sensitive=False))
    # rouge
    # expects tokens sperated by spaces
    rouge_metric = load('rouge')
    out_metrics.update(rouge_metric.compute(predictions=predictions, references=labels, rouge_types=['rouge2', 'rougeL'], use_stemmer=True, use_aggregator=True))
    # google bleu
    bleu_metric = load('google_bleu')
    out_metrics.update(bleu_metric.compute(predictions=predictions, references=labels, max_len=4))
    return out_metrics, len(predictions)

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
        project_name="score",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    df = pd.read_csv('./data/nomelt-model/predictions.tsv', sep='\t')
    df = df.applymap(lambda x: ' '.join(list(x)))
    logger.info(f"Loaded predictions.  Computing metrics.")

    # chunk up the dataframe and use multiprocessing to compute metrics
    predictions = df['prediction'].tolist()
    labels = df['label'].tolist()
    inputs = [(predictions[i:i+10], labels[i:i+10]) for i in range(0, len(predictions), 10)]

    # compute metrics
    results = Parallel(n_jobs=-1)(delayed(compute_metrics)(ins) for ins in inputs)
    logger.info(f"Computed metrics.  Aggregating results.")
    metrics_aggregator = {}
    for result in results:
        metrics, n = result
        for k, v in metrics.items():
            if k not in metrics_aggregator:
                metrics_aggregator[k] = 0
            metrics_aggregator[k] += float(v * n)
    # average
    for k, v in metrics_aggregator.items():
        metrics_aggregator[k] /= len(predictions)
    # convert all to float
    metrics = {k: float(v) for k, v in metrics.items()}

    # Log the final metrics
    logger.info(f"Average Metrics over Test Set: {pprint.pformat(metrics)}")

    # save metrics to dvc yaml file
    with open('./data/nomelt-model/test_scores.json', 'w') as f:
        json.dump(metrics, f)
    try:
        tracker.stop()
    except:
        pass

if __name__ == "__main__":
    main()


    