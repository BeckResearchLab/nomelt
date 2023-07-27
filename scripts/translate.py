import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, AutoConfig
from yaml import safe_load
import re
import argparse

def main(sequence):
    # Load parameters from params.yaml
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model_config = AutoConfig.from_pretrained('./data/nomelt-model/model')
    for model_hyperparam in params['model']['model_hyperparams']:
        setattr(model_config, model_hyperparam, params['model']['model_hyperparams'][model_hyperparam])
    setattr(model_config, 'max_length', params['model']['generation_max_length'])
    model = AutoModelForSeq2SeqLM.from_pretrained('./data/nomelt-model/model', config=model_config)
    model.to(device)

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('./data/nomelt-model/model')
    except:
        tokenizer = T5Tokenizer.from_pretrained('./data/nomelt-model/model')

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenize and process the sequence
    def prepare_string(string):
        out = ' '.join(string)
        out = re.sub(r"[UZOB]", "X", out)
        return out
    sequence = prepare_string(sequence)
    input_tensor = tokenizer.encode(sequence, return_tensors="pt").to(device)

    # Generate output using the model
    with torch.no_grad():
        output_tensor = model.generate(input_tensor, max_length=params['model']['generation_max_length'], num_beams=params['model']['generation_num_beams'])

    # Decode and display the generated sequence
    output_tensor = torch.where(output_tensor != -100, output_tensor, tokenizer.pad_token_id)
    generated_sequence = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)[0]
    print(f"Generated Sequence: {''.join(generated_sequence.split())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a sequence using a pretrained model.")
    parser.add_argument("sequence", action='store', type=str, help="Input sequence for generation.")
    args = parser.parse_args()
    main(args.sequence)
