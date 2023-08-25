"""Substitute a homeodomain variant in"""
import nomelt.haddock
import json
import os
import pandas as pd

import codecarbon

import logging
logger = logging.getLogger('root')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('./logs/enh1_dna_binding_score.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

COMPLEX_FILE = os.path.abspath('./data/pdbs/enh1_dna_complex.pdb')
WDIR = os.path.abspath('./tmp/enh1_vs_cons_binding/')

def do_one(pdb, name):
    nomelt.haddock.prepare_one_complex(COMPLEX_FILE, pdb, output_dir=WDIR)
    nomelt.haddock.run_haddock(
        protein_pdb=os.path.join(WDIR, 'protein.pdb'),
        dna_pdb=os.path.join(WDIR, 'dna.pdb'),
        run_dir=os.path.join(WDIR, name)
    )
    # load the score from file
    scores = pd.read_csv(
        os.path.join(WDIR, name, 'run', 'analysis', '4_emref_analysis', 'capri_ss.tsv'), sep='\t')
    # take top 5
    top5 = scores.iloc[:5]
    # get average score
    avg_score = top5['score'].mean()
    # get top structure file
    best_struct = os.path.basename(top5.loc[0, 'model'])
    return avg_score, os.path.join(WDIR, name, 'run', '4_emref', best_struct)

if __name__ == '__main__':
    
    tracker = codecarbon.OfflineEmissionsTracker(
        project_name="enh1_consensus_docking",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    ENH1_PDB = './data/pdbs/wt.pdb'
    CONSENSUS_PDB = './data/pdbs/consensus_homeodomain.pdb'

    enh_score, enh_complex = do_one(ENH1_PDB, 'enh1')
    cons_score, cons_complex = do_one(CONSENSUS_PDB, "consensus")

    with open('./data/enh/enh1_vs_consensus_dna_binding.json', 'w') as f:
        json.dump({
            'enh1': {
                'score': enh_score,
                'complex': enh_complex
            },
            'consensus': {
                'score': cons_score,
                'complex': cons_complex
            }
        }, f, indent=2)
    tracker.stop()

    