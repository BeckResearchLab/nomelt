"""Script to ensure that enh is not in the training set."""

from nomelt.blast import run_blast_search
import json

ENH1 = "DKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKK"


if __name__ == '__main__':
    records = run_blast_search(ENH1, './data/dataset', './tmp/blast')

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
