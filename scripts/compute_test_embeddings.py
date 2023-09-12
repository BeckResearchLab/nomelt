"""Extract embeddings and attentions for test set, save to disk."""
import codecarbon
import logging
import numpy as np
import pandas as pd
import os
import re
import torch
import pprint
from yaml import safe_load, dump
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_from_disk
import transformers.integrations
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    DataCollatorForSeq2Seq
)

import logging
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

def main():
    # start logger
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    utils_logger = logging.getLogger('nomelt')
    utils_logger.setLevel(getattr(logging, LOGLEVEL))
    utils_logger.addHandler(fh)
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(getattr(logging, LOGLEVEL))
    transformers_logger.addHandler(fh)


    # Load parameters from DVC
    # retry until successful
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)

    # start carbon tracker
    tracker = codecarbon.OfflineEmissionsTracker(
        project_name="generate_embeddings",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    # load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(params['model']['pretrained_model'])
    except:
        tokenizer = T5Tokenizer.from_pretrained(params['model']['pretrained_model'])

    # Load and sample as necessary
    dataset = load_from_disk('./data/dataset/')['test']
    logger.info(f"Loaded dataset.  {dataset}")

    # keep only extremes in dataset to get a better uniform score
    dataset = dataset.filter(lambda x: x['status_in_cluster'] in ['extreme', 'unique'])#.select(range(1000))
    logger.info(f"Keeping only extreme cluster and unique sequences. New size: {dataset}")

    # preprocess data with tokenizer
    def prepare_string(string):
        out = ' '.join(string)
        out = re.sub(r"[UZOB]", "X", out)
        return out
    def process_batch(batch):
        in_col, out_col = 'meso_seq', 'thermo_seq'

        in_seqs = [prepare_string(seq) for seq in batch[in_col]]
        out_seqs = [prepare_string(seq) for seq in batch[out_col]]

        # tokenize inputs and outputs
        model_inputs = tokenizer(in_seqs, max_length=params['model']['generation_max_length'], padding='longest', truncation=True)
        decoder_input_ids = torch.tensor(tokenizer(out_seqs, max_length=params['model']['generation_max_length'], padding='longest', truncation=True)['input_ids'])
        model_inputs['decoder_input_ids'] = decoder_input_ids

        input_ids = torch.tensor(model_inputs['input_ids']).cuda()
        attention_mask = torch.tensor(model_inputs['attention_mask']).cuda()
        decoder_input_ids = torch.tensor(model_inputs['decoder_input_ids']).cuda()

        with torch.no_grad():
            outputs = MODEL(
                input_ids= input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True)

        batch["encoder_embeddings"] = outputs.encoder_hidden_states[-1].cpu().numpy()
        batch["decoder_embeddings"] = outputs.decoder_hidden_states[-1].cpu().numpy()
        batch["encoder_attentions"] = outputs.encoder_attentions[-1].cpu().numpy()
        batch["decoder_attentions"] = outputs.decoder_attentions[-1].cpu().numpy()
        batch["cross_attentions"] = outputs.cross_attentions[-1].cpu().numpy()
        batch['logits'] = outputs.logits.cpu().numpy()

        return batch
    dataset.set_format(type='torch')

    # load the model with potentially custom config
    model_config = AutoConfig.from_pretrained('./data/nomelt-model/model')
    for model_hyperparam in params['model']['model_hyperparams']:
        setattr(model_config, model_hyperparam, params['model']['model_hyperparams'][model_hyperparam])
    setattr(model_config, 'max_length', params['model']['generation_max_length'])
    MODEL = AutoModelForSeq2SeqLM.from_pretrained('./data/nomelt-model/model', config=model_config)

    # collate data into batches for generation
    MODEL = MODEL.cuda()
        
    first_batch_dict = dataset.select(range(1)).to_dict()
    output_dict = process_batch(first_batch_dict)

    # generate embeddings and attentions
    logger.info(f"Generating embeddings and attentions.")
    dataset = dataset.map(process_batch, batched=True, batch_size=1, load_from_cache_file=True, desc='Generating embeddings and attentions')

    # save to new disk location
    logger.info(f"Saving to disk.")
    dataset.save_to_disk('./data/nomelt-model/test_embeddings')
        
    try:
        tracker.stop()
    except:
        pass

if __name__ == "__main__":
    main()


    