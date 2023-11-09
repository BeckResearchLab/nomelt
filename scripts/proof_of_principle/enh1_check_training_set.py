"""Script to ensure that enh is not in the training set."""

from nomelt.blast import run_blast_search
import json
from datasets import load_from_disk, concatenate_datasets

ENH1 = "DKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKK"


if __name__ == '__main__':

    dataset = load_from_disk('./data/dataset/')
    dataset = concatenate_datasets([dataset['train'], dataset['eval'], dataset['test']])
    dataset.save_to_disk('./tmp/dataset-all')
    
    records = run_blast_search(ENH1, dataset, './tmp/blast/')

    r = records[0]
    alignments = r.alignments
    for a in alignments:
        hsp = a.hsps[0]
        e = hsp.expect
        s_cov = (hsp.sbjct_end - hsp.sbjct_start) / a.length
        q_cov = (hsp.query_end - hsp.query_start) / r.query_length
        break

    with open('./data/enh/training_data_homologs.json', 'w') as f:
        json.dump({'e': e, 's_cov': s_cov, 'q_cov': q_cov}, f)
