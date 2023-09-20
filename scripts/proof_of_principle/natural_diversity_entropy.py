"""Compute the token wise entropy of the natural diversity for each thermophilic test sequence.

Each sequence is jackhammered against all thermophilic sequences, then the MSA is used
to compute column wise amino acid frequencies. The entropy of each column is then computed
and averaged over all residues and all sequences.
"""
import os
from Bio import AlignIO
import pandas as pd
import numpy as np
import datasets
import duckdb as ddb
import json
import subprocess

import logging
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

def write_fasta_db(seqs):
    with open("./tmp/database.fasta", "w") as f:
        for idx, row in seqs.iterrows():
            f.write(f">{row['pid']}\n{row['protein_seq']}\n")
    logger.info(f"Wrote {seqs.shape[0]} sequences to database file")

def write_query_file(seqs):
    with open("./tmp/query.fasta", "w") as f:
        for i, (idx, row) in enumerate(seqs.iterrows()):
            f.write(f">{row['pid']}\n{row['protein_seq']}\n")
    logger.info(f"Wrote {i+1} sequences to query file")

def run_jackhmmer(query_fasta, database_fasta, output_file):
    # skip if output file exists
    if os.path.exists(output_file):
        return

    command = [
        'jackhmmer',
        '-o', '/dev/null',  
        '-A', output_file,  
        '--noali', 
        '--F1', '0.0005',  
        '--F2', '0.00005',  
        '--F3', '0.0000005',
        '--incE', '0.0001',  
        '-E', '0.0001',  
        '--cpu', '32',  
        '-N', '1',  
        query_fasta,
        database_fasta
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(f'Jackhmmer failed: {process.stderr.decode()}')

def cross_entropy(ps):
    return -np.sum(np.log(ps)), len(ps)

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

    ds = datasets.load_from_disk('./data/dataset')['test']
    ds = ds.filter(lambda x: x['status_in_cluster'] in ['extreme', 'unique'])

    conn = ddb.connect('./data/database.ddb', read_only=True)
    seqs = conn.execute('SELECT pid, protein_seq FROM proteins INNER JOIN taxa ON (taxa.taxid=proteins.taxid) WHERE taxa.temperature>=65.0').df()
    conn.close()

    write_fasta_db(seqs)
    query_df = pd.DataFrame({'pid': ds['index'], 'protein_seq': ds['thermo_seq']})
    logger.info(f"Loaded {query_df.shape[0]} sequences from test set")
    query_df.drop_duplicates(subset=['protein_seq'], inplace=True)
    logger.info(f"Removed {len(ds['index']) - query_df.shape[0]} duplicates")
    write_query_file(query_df)

    # run jackhammer
    logger.info("Running jackhammer on {} sequences".format(query_df.shape[0]))
    run_jackhmmer('./tmp/query.fasta', './tmp/database.fasta', './tmp/thermo_jackhammer.sto')
    logger.info("Done hmmering")

    # open up the jackhammer output and compute cross entrop for each one
    alignments = AlignIO.parse("./tmp/thermo_jackhammer.sto", "stockholm")

    # Convert the alignment object to a Pandas DataFrame
    cross_entropy_running_sum = 0
    total_residue_count = 0
    for alignment in alignments:
        align_dict = {}
        for i, record in enumerate(alignment):
            if i == 0:
                reference = record.id
            else:
                # skip self, cause it will also be in the DB
                if record.id.startswith(reference):
                    continue
            align_dict[record.id] = list(record.seq)
        logger.info(f'Alignment with {len(align_dict[record.id])} sequences')
        df = pd.DataFrame.from_dict(align_dict, orient='index')
        df = df.replace('-', None)
        amino_acid_likelihood = df.apply(lambda x: x.value_counts(normalize=True)).fillna(1e-14)
        ref_columns = df.iloc[0]
        probs = [amino_acid_likelihood.loc[v,k] for k, v in ref_columns.items() if v is not None]
        entropy, residue_count = cross_entropy(probs)
        cross_entropy_running_sum += entropy
        total_residue_count += residue_count

    logger.info(f"Cross entropy: {cross_entropy_running_sum / total_residue_count}")
    
    # save the loss to a file
    with open('./data/proof_of_principle/natural_diversity_entropy.json', 'w') as f:
        json.dump({'cross_entropy': cross_entropy_running_sum / total_residue_count}, f)


