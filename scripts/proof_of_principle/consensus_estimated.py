from nomelt.thermo_estimation import mAFminDGEstimator, mAFminDGArgs, AlphaFoldArgs
import json
import os
import codecarbon

import logging
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('./logs/enh1_vs_consensus_in_silico_estimator.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

estimator_args=mAFminDGArgs(
    af_params='./.config/af_singularity_config.yaml',
    wdir='./tmp/af_dg/',
    use_relaxed=False,
    num_replicates=25,
    fix_msas=True
)
if __name__ == '__main__':
    tracker = codecarbon.OfflineEmissionsTracker(
        project_name="consensus_in_silico_estimator",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()
    estimator = mAFminDGEstimator(args=estimator_args)
    sequences = [
        'RRKRTTFTKEQLEELEELFEKNRYPSAEEREELAKKLGLTERQVKVWFQNRRAKEKK'
    ]
    ids = ["consensus-hd"]
    outs = estimator.run(sequences=sequences, ids=ids)
    tracker.stop()
    with open('./data/proof_of_principle/consensus_estimated.json', 'w') as f:
        json.dump(outs, f)