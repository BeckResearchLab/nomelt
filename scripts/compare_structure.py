"""Get structures/predicted structures, compare the ss distribution of a given thermophilic sequence
and a structural alignment to its generated counterparts."""
import os
import pandas as pd
import numpy as np
import duckdb as ddb
import json
import subprocess
import multiprocessing as mp
import torch
import requests
import esm
import codecarbon
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import pydssp
from collections import Counter
import time
import tqdm

import logging
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

MODEL = esm.pretrained.esmfold_v1().eval().cuda()

def esm_one_struc(i, sequence):
    if not os.path.exists('./tmp/esmfold_predicts/'):
        os.makedirs('./tmp/esmfold_predicts/')

    if os.path.exists(f"./tmp/esmfold_predicts/structure_{i}.pdb"):
        logger.info(f"Structure {i} already exists, skipping")
    else:
        with torch.no_grad():
            output = MODEL.infer_pdb(sequence)
        
        with open(f"./tmp/esmfold_predicts/structure_{i}.pdb", "w") as f:
            f.write(output)
    return f"./tmp/esmfold_predicts/structure_{i}.pdb"

def process_pdbs_dssp(pdbs, device=None, output_file=None):
    results = []
    if device == 'cuda':
        assert torch.cuda.is_available(), "CUDA is not available"
    elif device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fh = open(output_file, 'w') if output_file is not None else None

    for pdb in tqdm.tqdm(pdbs):
        coord = torch.tensor(pydssp.read_pdbtext(open(pdb, 'r').read()))
        coord = coord.to(device)
        dsspline = ''.join(pydssp.assign(coord))
        results.append((dsspline, pdb))

        if fh:
            fh.write(f"{dsspline} {pdb}\n")
            
    if fh:
        fh.close()

    return results

def js_divergence(p, q):
    """
    Compute the JS divergence between two distributions for secondary structure
    categories
    """
    assert len(p) == len(q)
    assert set(p.keys()) == set(q.keys())
    # compute the mean distribution
    m = {}
    for k in p.keys():
        m[k] = (p[k] + q[k]) / 2
    # remove 0.0 values
    m = {k: v for k, v in m.items() if v != 0.0}

    # compute the KL divergences
    kl_pm = sum([p[k] * np.log2(p[k] / m[k]) for k in m.keys()])
    kl_qm = sum([q[k] * np.log2(q[k] / m[k]) for k in m.keys()])
    return (kl_pm + kl_qm) / 2
    

def _run_fatcat1(ins):
    i, pdb1, pdb2 = ins
    if os.path.exists(f'./tmp/fatcat/fatcat_{i}.txt'):
        logger.info(f"Fatcat file {i} already exists, skipping")
        with open(f'./tmp/fatcat/fatcat_{i}.txt', 'r') as f:
            pdb1, pdb2, p_value = f.read().strip().split()
        return p_value
    command = [
        'FATCAT',
        '-p1', os.path.basename(pdb1),
        '-p2', os.path.basename(pdb2),
        '-i1', os.path.dirname(pdb1),
        '-i2', os.path.dirname(pdb2),
        '-q'
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.stdout.decode(), process.stderr.decode()
    
    # Find the line containing the p-value
    p_value_line = next(line for line in stdout.split('\n') if line.startswith("P-value"))

    # Extract the p-value
    p_value = float(p_value_line.split()[1])
    assert p_value >= 0 and p_value <= 1, f"Invalid p-value {p_value} for files {pdb1} and {pdb2}"
    logger.info("P-value for files %s and %s is %s", pdb1, pdb2, p_value)
    with open(f'./tmp/fatcat/fatcat_{i}.txt', 'w') as f:
        f.write(f"{pdb1} {pdb2} {p_value}\n")
    return p_value

def run_fatcat(pdb_list1, pdb_list2):
    """
    Run FATCAT on the structures in the dataframe
    :param pdb_list1: List of paths to the first set of structures
    :param pdb_list2: List of paths to the second set of structures
    Note the above parameters should be lists of the same length, each index makes a pair
    """
    if not os.path.exists('./tmp/fatcat/'):
        os.makedirs('./tmp/fatcat/')
    inputs = list(zip(range(len(pdb_list1)), pdb_list1, pdb_list2))

    pool = mp.Pool(mp.cpu_count())
    outs = pool.map(_run_fatcat1, inputs)
    return outs

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
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(getattr(logging, LOGLEVEL))
    transformers_logger.addHandler(fh)

    tracker = codecarbon.OfflineEmissionsTracker(
        project_name="structure",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    # get the saved predicted sequences
    predictions = pd.read_csv('./data/nomelt-model/predictions.tsv', sep='\t')

    # get the pids for thermophilic sequences from the dataset
    logger.info("Getting pids for thermophilic sequences from duckdb")
    conn = ddb.connect('./data/database.ddb', read_only=True)
    conn.execute("CREATE INDEX seqs ON proteins(protein_seq)")
    seqs = predictions['label'].unique().tolist()
    placeholders = ', '.join(['?' for _ in seqs])
    query = f"SELECT pid, protein_seq FROM proteins WHERE protein_seq IN ({placeholders})"
    thermo_pids_map = conn.execute(query, parameters=seqs).df()
    thermo_pids_map = thermo_pids_map.drop_duplicates(subset=['protein_seq']).set_index('protein_seq')['pid'].to_dict()
    conn.close()

    logger.info(f"Have pids for thermophilic sequences. Found {len(thermo_pids_map)} out of {len(seqs)}")

    # get rid of examples that we couldn't find structures
    predictions = predictions[predictions['label'].isin(thermo_pids_map.keys())]

    # query alphafold for pdb structures for each thermo sequence
    # download the structures to file
    if not os.path.exists('./tmp/alphafold_downloads/'):
        os.makedirs('./tmp/alphafold_downloads/')
    for thermo_pid in thermo_pids_map.values():
        url = f'https://alphafold.ebi.ac.uk/files/AF-{thermo_pid}-F1-model_v4.pdb'
        filename = f'./tmp/alphafold_downloads/{thermo_pid}.pdb'
        if os.path.exists(filename):
            logger.info(f"File {filename} already exists, skipping download")
            continue
        response = requests.get(url)
        if response.ok:
            with open(filename, 'wb') as f:
                f.write(response.content)
        else:
            logger.warning(f"Could not download {filename}")
        logger.info(f"Done with {filename}")
        time.sleep(1)
    thermo_structures_list = [f'./tmp/alphafold_downloads/{thermo_pids_map[seq]}.pdb' for seq in predictions['label']]
    predictions['thermo_pdbs'] = thermo_structures_list
    # make sure each file exists, if not, remove the row
    predictions = predictions[predictions['thermo_pdbs'].apply(lambda x: os.path.exists(x))]

    logger.info(f"Have thermo structures for {len(predictions)} examples")

    # predict structures for generated sequences with esmfold
    logger.info("Predicting structures for generated sequences with esmfold")
    
    predicted_structures_list = []
    for i, row in predictions.iterrows():
        predicted_structures_list.append(esm_one_struc(i, row['prediction']))

    # for each pair, comapre secondary structure distributions and FATCAT alignment scores
    logger.info("Comparing secondary structure distributions and FATCAT alignment scores")
    assert len(thermo_structures_list) == len(predicted_structures_list)

    # do dssp on all structures
    logger.info("Running dssp on all structures")
    if os.path.exists('./tmp/thermo_structures_dssp.txt'):
        logger.info("DSSP file already exists, skipping")
        thermo_results = list(pd.read_csv('./tmp/thermo_structures_dssp.txt', sep=' ', header=None).values)
    else:
        thermo_results = process_pdbs_dssp(thermo_structures_list, output_file='./tmp/thermo_structures_dssp.txt')
    if os.path.exists('./tmp/predicted_structures_list_dssp.txt'):
        logger.info("DSSP file already exists, skipping")
        predicted_results = list(pd.read_csv('./tmp/predicted_structures_list_dssp.txt', sep=' ', header=None).values)
    else:
        predicted_results = process_pdbs_dssp(predicted_structures_list, output_file='./tmp/predicted_structures_list_dssp.txt')

    # compare secondary structure distributions
    # each line for each result is a length two iterable, first is the dssp string, second is the pdb file name
    thermo_ss = [x[0] for x in thermo_results]
    predicted_ss = [x[0] for x in predicted_results]
    assert len(thermo_ss) == len(predicted_ss)
    # convert to distributuion
    # we have three letters: H, E, -. Get a count of each and normalize
    jss = []
    for t_ss, p_ss in zip(thermo_ss, predicted_ss):
        t_ss = Counter(t_ss)
        p_ss = Counter(p_ss)
        t_ss = {k: v / sum(t_ss.values()) for k, v in t_ss.items()}
        p_ss = {k: v / sum(p_ss.values()) for k, v in p_ss.items()}
        # make sure each dict has each letter
        for letter in ['H', 'E', '-']:
            if letter not in t_ss:
                t_ss[letter] = 0
            if letter not in p_ss:
                p_ss[letter] = 0
        # compute jenson shannon divergence
        js_ss = js_divergence(t_ss, p_ss)
        logger.info(f"JS divergence for secondary structure: {js_ss}")
        jss.append(js_ss)
    
    metrics = {
        'js_ss': (np.mean(jss), np.std(jss))
    }

    # now fatcat align the structures
    logger.info("Running fatcat on all structures")

    fat_scores = run_fatcat(thermo_structures_list, predicted_structures_list)
    metrics['fatcat'] = (np.mean(fat_scores), np.std(fat_scores))

    # save metrics
    with open('./data/nomelt-model/structure_metrics.json', 'w') as f:
        json.dump(metrics, f)
    tracker.stop()



