from nomelt.translate import translate_sequences
from yaml import safe_load
import json
import os

def main(sequence):
    # Load parameters from params.yaml
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)

    generated_sequences = translate_sequences(
        [sequence],
        './data/nomelt-model/model/',
        generation_max_length=params['model']['generation_max_length'],
        generation_num_beams=params['model']['generation_num_beams'],
        model_hyperparams=params['model']['model_hyperparams']
    )
    return generated_sequences[0]

if __name__ == "__main__":
    ENH1 = "DKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIKK"
    if not os.path.exists('./data/enh'):
        os.makedirs('./data/enh', exist_ok=True)
    generated_sequence = main(ENH1)[0]
    outs = {
        'original': ENH1,
        'generated': generated_sequence,
        'length_diff': len(generated_sequence) - len(ENH1)
    }
    with open('./data/enh/translate_enh1.json', 'w') as f:
        json.dump(outs, f)

