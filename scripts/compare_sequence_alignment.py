"""Align test generated sequences to the thermo ones."""
import os
import pandas as pd
import numpy as np
import duckdb as ddb
import json
import subprocess
import codecarbon
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo
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
    matrix = MatrixInfo.blosum62
    alignments = pairwise2.align.globaldx(seq1, seq2, matrix)
    best_alignment = alignments[0]
    return best_alignment

def calculate_identity(seq1, seq2):
    identical = sum(el1 == el2 for el1, el2 in zip(seq1, seq2))
    return (identical / min(len(seq1), len(seq2))) * 100

def calculate_bpr(score, length):
    return score / length

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
    predictions = pd.read_csv('./data/nomelt-model/predictions.tsv', sep='\t', header=None)
    predictions.columns = ['meso', 'gen', 'thermo']
    # remove spaces in all values in teh dataframe
    predictions = predictions.applymap(lambda x: ''.join(str(x).split()))

    gen_seqs = predictions['gen'].values
    thermo_seqs = predictions['thermo'].values

    aligned_results = []
    for gen, thermo in tqdm(zip(gen_seqs, thermo_seqs), total=len(gen_seqs)):
        best_alignment = global_alignment(gen, thermo)
        aligned_seq1, aligned_seq2, score, begin, end = best_alignment
        identity = calculate_identity(aligned_seq1, aligned_seq2)
        
        alignment_length = end - begin
        bpr = calculate_bpr(score, alignment_length)

        result = {
            'gen': aligned_seq1,
            'thermo': aligned_seq2,
            'score': score,
            'begin': begin,
            'end': end,
            'identity': identity,
            'bpr': bpr
        }
        aligned_results.append(result)

    aligned_df = pd.DataFrame(aligned_results)
    aligned_df.to_csv('./data/nomelt-model/sequence_alignment.csv', index=False)

    tracker.stop()