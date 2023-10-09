"""Align test generated sequences to the thermo ones."""
import os
import pandas as pd
import numpy as np
import duckdb as ddb
import json
import subprocess
import codecarbon
from Bio import pairwise2
from Bio.Align import substitution_matrices
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

def global_alignment(seq1, seq2):
    matrix = substitution_matrices.load("BLOSUM62")
    alignments = pairwise2.align.globaldx(seq1, seq2, matrix)
    best_alignment = alignments[0]
    return best_alignment

def calculate_identity(seq1, seq2):
    identical = sum(el1 == el2 for el1, el2 in zip(seq1, seq2))
    return (identical / ((len(seq1) + len(seq2))/2))

def calculate_bpr(score, length):
    return score / length

def compute_alignment_metrics(seq1, seq2):
    aligned_seq1, aligned_seq2, score, begin, end = global_alignment(seq1, seq2)
    identity = calculate_identity(aligned_seq1, aligned_seq2)
    alignment_length = end - begin
    bpr = calculate_bpr(score, alignment_length)
    
    return {
        'seq1': aligned_seq1,
        'seq2': aligned_seq2,
        'score': score,
        'begin': begin,
        'end': end,
        'identity': identity,
        'bpr': bpr
    }

if __name__ == '__main__':
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
        project_name="compare_sequence_alignment",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    # get the saved predicted sequences
    predictions = pd.read_csv('./data/nomelt-model/predictions.tsv', sep='\t')

    gen_seqs = predictions['prediction'].values
    thermo_seqs = predictions['label'].values
    meso_seqs = predictions['input'].values

    aligned_results = []
    for meso, gen, thermo in tqdm(zip(meso_seqs, gen_seqs, thermo_seqs), total=len(gen_seqs)):
        result = {}
        
        result['tg'] = compute_alignment_metrics(gen, thermo)
        result['mt'] = compute_alignment_metrics(meso, thermo)
        result['mg'] = compute_alignment_metrics(meso, gen)
        
        aligned_results.append(result)

    # save to json
    with open('./data/nomelt-model/test_predictions_aligned_results.json', 'w') as f:
        json.dump(aligned_results, f)

    tracker.stop()