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
    wdir='./data/enh/initial_estimate/',
    use_relaxed=False,
    num_replicates=25,
    fix_msas=True,
    residue_length_norm=True
)
if __name__ == '__main__':
    tracker = codecarbon.OfflineEmissionsTracker(
        project_name="estimate_trans_energy_enh1",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    # load the translations
    with open('./data/enh/translate_enh1.json', 'r') as f:
        _ = json.load(f)
        ENH1_TRANSLATED = _['generated']

    estimator = mAFminDGEstimator(args=estimator_args)
    sequences = [
        ENH1_TRANSLATED
    ]
    ids = ["trans"]
    outs = estimator.run(sequences=sequences, ids=ids)
    tracker.stop()
    with open('./data/enh/translated_energy_enh1.json', 'w') as f:
        json.dump(outs, f)