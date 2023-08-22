import codecarbon
from accelerate import Accelerator, find_executable_batch_size
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
from evaluate import load
import transformers.integrations
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    DataCollatorForSeq2Seq
)

import nomelt.dvclive

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

    # start accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # start carbon tracker
    if accelerator.is_local_main_process:
        tracker = codecarbon.OfflineEmissionsTracker(
            project_name="predict",
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

    with accelerator.main_process_first():
        # Load and sample as necessary
        dataset = load_from_disk('./data/dataset/')['test']
        logger.info(f"Loaded dataset.  {dataset}")

        # keep only extremes in dataset to get a better uniform score
        dataset = dataset.filter(lambda x: x['status_in_cluster'] in ['extreme', 'unique']).select(range(1000))
        logger.info(f"Keeping only extreme cluster and unique sequences. New size: {dataset}")

        # preprocess data with tokenizer
        def prepare_string(string):
            out = ' '.join(string)
            out = re.sub(r"[UZOB]", "X", out)
            return out
        def preprocess_dataset_to_model_inputs(examples):
            if params['model']['task'] == 'translation':
                in_col, out_col = 'meso_seq', 'thermo_seq'
            elif params['model']['task'] == 'reconstruction':
                if examples['index'][0] % 2 == 0:
                    in_col, out_col = 'meso_seq', 'meso_seq'
                else:
                    in_col, out_col = 'thermo_seq', 'thermo_seq'
            else:
                raise ValueError(f"Task {params['model']['task']} not recognized. Must be 'translation' or 'reconstruction'.")

            in_seqs = [prepare_string(seq) for seq in examples[in_col]]
            out_seqs = [prepare_string(seq) for seq in examples[out_col]]

            # tokenize inputs and outputs
            model_inputs = tokenizer(in_seqs, max_length=params['model']['generation_max_length'], padding='max_length', truncation=True)
            labels = torch.tensor(tokenizer(out_seqs, max_length=params['model']['generation_max_length'], padding='max_length', truncation=True)['input_ids'])
            # fill -100s to be ignored in loss
            labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
            model_inputs['labels'] = labels
            return model_inputs
        initial_columns = dataset.column_names
        print(initial_columns)
        logger.info(f"Tokenizing and preparing model inputs. Using task '{params['model']['task']}'. 'tranlsation' is meso to thermo, 'reconstruction' is meso to meso or thermo to thermo.")
        dataset = dataset.map(preprocess_dataset_to_model_inputs, batched=True, remove_columns=initial_columns, load_from_cache_file=True, desc='Tokenizing and preparing model inputs')
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # load the model with potentially custom config
    model_config = AutoConfig.from_pretrained('./data/nomelt-model/model')
    for model_hyperparam in params['model']['model_hyperparams']:
        setattr(model_config, model_hyperparam, params['model']['model_hyperparams'][model_hyperparam])
    setattr(model_config, 'max_length', params['model']['generation_max_length'])
    MODEL = AutoModelForSeq2SeqLM.from_pretrained('./data/nomelt-model/model', config=model_config)

    @find_executable_batch_size(starting_batch_size=10)
    def inner_loop(batch_size):
        # collate data into batches for generation
        model = MODEL.to(device)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL)

        # Prepare DataLoader for batching
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

        # accelerate prepare teh data and model
        test_dataloader = accelerator.prepare(test_dataloader)

        # Initialize metric accumulators
        prediction_file = open(f'./data/nomelt-model/device_predictions_{str(device)}.tsv', 'w')
        total_loss = torch.tensor(0.0).to(accelerator.device)
        # Loop through each batch and generate predictions
        for batch in tqdm(test_dataloader):
            # batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                # Generate predictions
                outputs = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=params['model']['generation_max_length'],
                    num_beams=params['model']['generation_num_beams'],
                )
                loss = model(**batch).loss
                logger.info(f"Batch loss {loss}")
                total_loss += loss * torch.tensor(len(batch['input_ids'])).to(accelerator.device)
            
            predictions, labels = outputs, batch['labels']
            predictions = torch.where(predictions != -100, predictions, tokenizer.pad_token_id)
            # decode
            predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

            # write predictions to file
            for i, p, l in zip(inputs, predictions, labels):
                prediction_file.write(f"{i}\t{p}\t{l}\n")

        total_loss = accelerator.gather_for_metrics(total_loss)

        prediction_file.close()
        return total_loss
    
    # Run the inner loop
    total_losses = inner_loop()
    logger.info(f"Total losses retrieved: {total_losses}")
    total_loss = float(total_losses.cpu().sum())
    logger.info(f"Total loss: {total_loss}")

    accelerator.wait_for_everyone()

    # now compute metrics
    if accelerator.is_local_main_process:
        logger.info("Completed generation.  Loading predictions.")
        dfs = [pd.read_csv(f'./data/nomelt-model/{f}', sep='\t', header=None) for f in os.listdir('./data/nomelt-model/') if 'device_predictions' in f]
        df = pd.concat(dfs)
        # remove old files
        df.to_csv('./data/nomelt-model/predictions.tsv', sep='\t', header=None, index=False)
        for f in os.listdir('./data/nomelt-model/'):
            if 'device_predictions' in f:
                os.remove(f'./data/nomelt-model/{f}')
        
    try:
        tracker.stop()
    except:
        pass

if __name__ == "__main__":
    main()


    