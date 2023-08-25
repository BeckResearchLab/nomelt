"""Determine the effect of various mutations to the tail of enh

See if adding helix will always report a better score
"""

from nomelt.thermo_estimation import mAFminDGEstimator, mAFminDGArgs, AlphaFoldArgs
import json
import os
import codecarbon

import logging
logger = logging.getLogger('root')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('./logs/estimate_trans_energy_enh1.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

estimator_args=mAFminDGArgs(
    af_params='./.config/af_singularity_config.yaml',
    wdir='./data/proof_of_principle/vary_mutations/',
    use_relaxed=False,
    num_replicates=25,
    fix_msas=True,
    residue_length_norm=False
)
if __name__ == '__main__':
    tracker = codecarbon.OfflineEmissionsTracker(
        project_name="estimate_various_mutations",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    # load the translations
    ENH1 = "DKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKK"
    data = {
        'wt': ENH1,
        'add_good_helix': ENH1 + 'AAL',
        'add_many_helix': ENH1 + 'AALAALAALAALAALAALAAL',
        'add_bad_helix': ENH1 + 'PGP',
    }

    estimator = mAFminDGEstimator(args=estimator_args)
    sequences = list(data.values())
    ids = list(data.keys())
    outs = estimator.run(sequences=sequences, ids=ids)
    tracker.stop()
    with open('./data/proof_of_principle/vary_mutations.json', 'w') as f:
        json.dump(outs, f)