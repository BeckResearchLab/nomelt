"""Run mAF with length normalization on homologs with exp melting temp, see if it still works.
https://www.sciencedirect.com/science/article/pii/S0022283698917600
"""
from nomelt.thermo_estimation import mAFminDGEstimator, mAFminDGArgs, AlphaFoldArgs
import json
import os
import codecarbon
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')

import logging
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('./logs/mAF_test.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
utils_logger = logging.getLogger('nomelt')
utils_logger.setLevel(getattr(logging, 'INFO'))
utils_logger.addHandler(file_handler)

estimator_args=mAFminDGArgs(
    af_params='./.config/af_singularity_config.yaml',
    wdir='./tmp/mAF_norm_test/',
    use_relaxed=False,
    num_replicates=25,
    fix_msas=True,
    residue_length_norm=True
)
if __name__ == '__main__':
    tracker = codecarbon.OfflineEmissionsTracker(
        project_name="mAF_test",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()
    estimator = mAFminDGEstimator(args=estimator_args)
    full_seqs2 = {
        'Sa':  "DVSGTVCLSALPPEATDTLNLIASDGPFPYSQDGVVFQNRESVLPTQSYGYYHEYTVITPGARTRGTRRIITGEATQEDYYTGDHYATFSLIDQTC",
        'Sa2': "ADPALDVCRTKLPSQAQDTLALIAKNGPYPYNRDGVVFENRESRLPKKGNGYYHEFTVVTPGSNERGTRRVVTGGYGEQYESPDHYATFQEIDPRC",
        'Sa3': "ASVKAVGRVCYSALPSQAHDTLDLIDEGGPFPYSQDGVVFQNREGLLPAHSTGYYHEYTVITPGSPTRGARRIITGQQWQEDYYTADHYASFRRVDFAC",
        'Ba':  "AQVINTFDGVADYLQTYHKLPNDYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR",
        'T1':  "ACDYTCGSNCYSSSDVSTAQAAGYQLHEDGETVGSNSYPHKYNNYEGFDFSVSSPYYEWPILSSGDVYSGGSPGADRVVFNENNQLAGVITHTGASGNNFVECT",
        'A': "KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV"
    }
    sequences = list(full_seqs2.values())
    ids = list(full_seqs2.keys())
    temps = [48.4, 41.1, 47.2, 53.2, 51.6, 62.8]

    outs = estimator.run(sequences=sequences, ids=ids)
    # get the scores from the outs
    scores = [outs[id_][0] for id_ in ids]

    # make a plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=temps[:-1], y=scores[:-1], ax=ax)
    ax.set_xlabel('Experimental melting temperature (C)')
    ax.set_ylabel('Score')
    plt.savefig('./data/plots/mAF_length_diff_test.png', dpi=300, bbox_inches='tight')
    # compute spearmans
    rho, p = spearmanr(temps, scores)

    metrics = {'score': outs, 'spearman': rho, 'p': p}
    # the 62 degree variant is way underestimated by the model, drop it as a outlier and then compute spear
    rho, p = spearmanr(temps[:-1], scores[:-1])
    metrics['spearman_no_outlier'] = rho
    metrics['p_no_outlier'] = p

    tracker.stop()
    with open('./data/proof_of_principle/mAF_length_diff_test.json', 'w') as f:
        json.dump(metrics, f)