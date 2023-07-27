import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, AutoConfig
import re

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

def translate_sequences(
        sequences, 
        model_path,
        model_hyperparams: dict=None,
        generation_max_length: int=250,
        generation_num_beams: int=10
    ):
    
    model, tokenizer = prepare_model_and_tokenizer(model_path, model_hyperparams)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sequences = [prepare_string(s) for s in sequences]
    input_tensor = tokenizer(sequences, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        output_tensor = model.generate(**input_tensor, max_length=generation_max_length, num_beams=generation_num_beams)

    output_tensor = torch.where(output_tensor != -100, output_tensor, tokenizer.pad_token_id)
    generated_sequences = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)
    translated_sequences = [''.join(generated_sequence.split()) for generated_sequence in generated_sequences]
    del output_tensor
    del model
    torch.cuda.empty_cache()
    
    return translated_sequences