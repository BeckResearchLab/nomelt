"""Blast the test set against the train set in order to evaluate the effectiveness of cluster splitting."""
import subprocess
import os
import json
from Bio.Blast.NCBIXML import parse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')
import datasets
import numpy as np
import pandas as pd


if __name__ == '__main__':
    if not os.path.exists('./data/plots'):
        os.makedirs('./data/plots')

    # Load dataset
    ds = datasets.load_from_disk('./data/dataset')

    # select one seq from each cluster
    # keep only a single sequence from each cluster
    # first mark the indexes of each
    cluster_dict = {}
    for i, clust in enumerate(ds['test']['cluster']):
        cluster_dict[clust] = i
    keep_indexes = set(cluster_dict.values())
    ds['test'] = ds['test'].filter(lambda x, idx: idx in keep_indexes, with_indices=True)
    print(len(ds['test']))

    # Write query sequence to file
    query_file_path = f'./tmp/blast_query.fasta'
    with open(query_file_path, 'w') as f:
        for i, seq in enumerate(ds['test']['meso_seq']):
            f.write(f">{i}\n{seq}\n")

    sbjct_file_path = f'./tmp/blast_sub.fasta'
    with open(sbjct_file_path, 'w') as f:
        for i, seq in enumerate(ds['train']['meso_seq']):
            f.write(f">{i}\n{seq}\n")

    # make db
    cmd = ['makeblastdb', '-in', sbjct_file_path, '-dbtype', 'prot']
    subprocess.run(cmd, check=True)

    # Run BLAST
    output_path = f'./tmp/blast_out.tsv'
    cmd = [
        'blastp', '-db', sbjct_file_path, '-query', query_file_path,
        '-evalue', '.001', '-outfmt', '5', '-out', output_path,
        '-num_threads', '32', '-word_size', '3',
        '-matrix', 'BLOSUM62', '-qcov_hsp_perc', '80'
    ]
    subprocess.run(cmd, check=True)

    # Parse and return alignments
    es = []
    perc_identities = []
    records_iter = parse(open(output_path, 'r'))

    for record in records_iter:
        # get only the best alignment and hsp for each query
        # (there should only be one)
        try:
            alignment = record.alignments[0]
        except IndexError:
            es.append(None)
            perc_identities.append(None)
            continue
        hsp = alignment.hsps[0]
        e = hsp.expect

        # check that the first hsp of the first align is indeed the best
        for align in record.alignments:
            for hsp in align.hsps:
                if hsp.expect < e:
                    raise ValueError('Not the best hsp')

        es.append(e)
        perc_identities.append(hsp.identities / ((alignment.length + record.query_length)/2))

    num_aligned = len([e for e in es if e is not None])
    # remove nans
    es = [e for e in es if e is not None]
    perc_identities = [p for p in perc_identities if p is not None]

    # make a plot of e-values and perc identities
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(np.log10(es), bins=100)
    ax[0].set_xlabel('log10(e-value)')
    ax[0].set_ylabel('Count')
    ax[1].hist(perc_identities, bins=100)
    ax[1].set_xlabel('Percent identity')
    fig.savefig('./data/plots/test_train_blast_hist.png', dpi=300, bbox_inches='tight')

    # compute quantiles of both and save metrics to json
    metrics = pd.DataFrame({'e': es, 'perc_id': perc_identities}).describe().to_dict()
    metrics['n_aligned'] = num_aligned
    metrics['n_total'] = len(ds['test'])
    with open('./data/test_train_blast_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)




    



    