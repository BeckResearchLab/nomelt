import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, AutoConfig
import re
import numpy as np
from collections import Counter
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO
import tempfile


def prepare_model_and_tokenizer(model_path, model_hyperparams=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_config = AutoConfig.from_pretrained(model_path)
    if model_hyperparams is not None:
        for key, value in model_hyperparams.items():
            setattr(model_config, key, value)
    else:
        pass
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=model_config)
    model.to(device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def prepare_string(string):
    out = ' '.join(string)
    out = re.sub(r"[UZOB]", "X", out)
    return out

def _decode_tensor(tensor, tokenizer):
    output_tensor = torch.where(tensor != -100, tensor, tokenizer.pad_token_id)
    generated_sequences = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)
    translated_sequences = [''.join(generated_sequence.split()) for generated_sequence in generated_sequences]
    return np.array(translated_sequences).reshape(-1,1)

def translate_sequences(
    sequences, 
    model_path,
    model_hyperparams: dict=None,
    generation_max_length: int=250,
    generation_num_beams: int=10,
    generation_ensemble_size: int=None,
    temperature: float=1.0,
):
    if generation_ensemble_size is not None:
        assert generation_ensemble_size > 1, "generation_ensemble_size must be greater than 1"
        assert generation_num_beams is None, "generation_num_beams must be None if generation_ensemble_size is not None"

    model, tokenizer = prepare_model_and_tokenizer(model_path, model_hyperparams)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sequences = [prepare_string(s) for s in sequences]
    input_tensor = tokenizer(sequences, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        if generation_ensemble_size is None:
            output_tensor = model.generate(**input_tensor, max_length=generation_max_length, num_beams=generation_num_beams)
        else:
            output_tensor = model.generate(
                **input_tensor,
                max_length=generation_max_length,
                num_beams=1,
                temperature=temperature,
                num_return_sequences=generation_ensemble_size,
                do_sample=True,
            )
        translated_sequences = _decode_tensor(output_tensor, tokenizer)

    del output_tensor
    del model
    torch.cuda.empty_cache()
    
    return translated_sequences

def perform_alignment(sequences, output_file="aligned.fasta"):
    # Write sequences to a temporary file
    with tempfile.NamedTemporaryFile(dir='./tmp', mode='w+t', delete=False) as file:
        for idx, seq in enumerate(sequences):
            print(f">seq{idx}\n{seq}\n")
            file.write(f">seq{idx}\n{seq}\n")
    
    # Perform alignment using Clustal Omega
    clustalomega_cline = ClustalOmegaCommandline(infile=file.name, outfile=output_file, verbose=True, auto=True, force=True)
    clustalomega_cline()
    
    # Read the aligned sequences
    alignment = AlignIO.read(output_file, "fasta")
    return alignment

def get_consensus_sequence(alignment):
    consensus_sequence = ''
    
    for col in range(len(alignment[0])):
        # Count the occurrence of each amino acid at this position
        amino_acid_counts = Counter(alignment[:, col])
        
        # Find the most frequent amino acid
        most_common_amino_acid = amino_acid_counts.most_common(1)[0][0]

        # if it is a gap, skip it
        if most_common_amino_acid == '-':
            continue
        else:
            consensus_sequence += most_common_amino_acid

    return consensus_sequence