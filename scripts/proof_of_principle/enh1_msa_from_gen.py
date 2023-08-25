"""Generate a multiple sequence alignment by generating many sequences stochastically."""

import nomelt.translate
from yaml import safe_load
import json

if __name__ == '__main__':

    ENH1 = "DKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKK"

    with open('./params.yaml', 'r') as f:
        params = safe_load(f)
    hparams = params['model']['model_hyperparams']

    ensemble = []
    while len(ensemble) < 100:
        ensemble_ = nomelt.translate.translate_sequences(
            [ENH1],
            model_path='./data/nomelt-model/model/',
            model_hyperparams=hparams,
            generation_num_beams=None,
            generation_ensemble_size=50,
            temperature=0.5
        )
        ensemble.extend([s[0] for s in ensemble_ if abs(len(s[0]) - len(ENH1))/len(ENH1) < 0.1])
    alignment = nomelt.translate.perform_alignment(ensemble, './tmp/translation_ensemble_aligned.fasta')
    consensus_sequence = nomelt.translate.get_consensus_sequence(alignment)

    with open('./data/proof_of_principle/ensemble_trans_enh.json', 'w') as f:
        json.dump({
            'consensus': consensus_sequence,
            'generated': ensemble,
            'align_file': './data/translation_ensemble_aligned.fasta'
        }, f)

