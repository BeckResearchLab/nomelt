from nomelt.translate import translate_sequences
from yaml import safe_load
import argparse

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
    with open('./data/translations.tsv', 'w') as f:
        f.write('original\ttranslation\n')
        f.write(sequence+'\t'+generated_sequences[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a sequence using a pretrained model.")
    parser.add_argument("sequence", action='store', type=str, help="Input sequence for generation.")
    args = parser.parse_args()
    main(args.sequence)
