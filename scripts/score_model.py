import codecarbon
import logging
import numpy as np
import os
import re
import torch
import pprint
from yaml import safe_load, dump
from torch.utils.data import DataLoader

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

def compute_loss_over_dataset(dset, model, tokenizer, device, params):
    """Compute the average loss over a dataset."""
    
    # Prepare DataLoader for batching
    dataloader = DataLoader(dset, batch_size=40, shuffle=False, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model))
    n_batches = int(len(dset)/40)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            total_loss += loss.item() * len(batch["input_ids"])
            total_samples += len(batch["input_ids"])
            logger.info(f"Finished batch {i} out of {n_batches}")
    
    return total_loss / total_samples

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

    # get the gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load parameters from DVC
    # retry until successful
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)

    # load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(params['model']['pretrained_model'])
    except:
        tokenizer = T5Tokenizer.from_pretrained(params['model']['pretrained_model'])

    # start carbon tracker
    tracker = codecarbon.OfflineEmissionsTracker( 
        project_name="score",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    # Load and sample as necessary
    dataset = load_from_disk('./data/dataset/')
    logger.info(f"Loaded dataset.  {dataset}")

    if params['training']['keep_only_extremes']:
        dataset = dataset.filter(lambda x: x['status_in_cluster'] in ['extreme', 'unique'])
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
    initial_columns = dataset['train'].column_names
    print(initial_columns)
    logger.info(f"Tokenizing and preparing model inputs. Using task '{params['model']['task']}'. 'tranlsation' is meso to thermo, 'reconstruction' is meso to meso or thermo to thermo.")
    dataset = dataset.map(preprocess_dataset_to_model_inputs, batched=True, remove_columns=initial_columns, load_from_cache_file=True, desc='Tokenizing and preparing model inputs')
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # load the model with potentially custom config
    model_config = AutoConfig.from_pretrained('./data/nomelt-model/model')
    for model_hyperparam in params['model']['model_hyperparams']:
        setattr(model_config, model_hyperparam, params['model']['model_hyperparams'][model_hyperparam])
    setattr(model_config, 'max_length', params['model']['generation_max_length'])
    model = AutoModelForSeq2SeqLM.from_pretrained('./data/nomelt-model/model', config=model_config)
    model.to(device)

    # define metrics to compute
    def compute_metrics(outputs):
        """We use:
        - ter: translation edit rate
        - Rouge 2: F1 score for bigrams
        - Rouge L: score for longest common subsequences
        - google_bleu: single sentence BLEU like score, minimum of recall and precision on 1, 2, 3, and 4 grams
        """
        # outputs encoded
        predictions, labels = outputs
        predictions = torch.where(predictions != -100, predictions, tokenizer.pad_token_id)
        # decode
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # outputs are list of strings, with spaces ## CHECK
        out_metrics = {}
        # tranlsation error rate
        ter_metric = load('ter')
        out_metrics.update(ter_metric.compute(predictions=predictions, references=labels, normalized=True, case_sensitive=False))
        # rouge
        # expects tokens sperated by spaces
        rouge_metric = load('rouge')
        out_metrics.update(rouge_metric.compute(predictions=predictions, references=labels, rouge_types=['rouge2', 'rougeL'], use_stemmer=True, use_aggregator=True))
        # google bleu
        bleu_metric = load('google_bleu')
        out_metrics.update(bleu_metric.compute(predictions=predictions, references=labels, max_len=5))
        return out_metrics

    # collate data into batches for generation
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # compute train loss
    logger.info("Computing train loss.")
    train_loss = compute_loss_over_dataset(dataset['train'], model, tokenizer, device, params)
    logger.info(f"Train loss {train_loss}")


    # Prepare DataLoader for batching
    test_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=data_collator)

    # Initialize metric accumulators
    metric_accumulators = {'loss': 0.0}
    total_samples = 0

    # Loop through each batch and generate predictions
    n_batches = int(len(dataset)/2)
    for i, batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            # Generate predictions
            outputs = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=params['model']['generation_max_length'],
                num_beams=params['model']['generation_num_beams'],
            )
            # also get loss
            loss = model(**batch).loss
        
        # Compute metrics for this batch
        batch_metrics = compute_metrics((outputs, batch["labels"]))
        
        # Accumulate metrics
        for metric, value in batch_metrics.items():
            if metric not in metric_accumulators:
                metric_accumulators[metric] = 0.0
            metric_accumulators[metric] += value * len(batch["input_ids"])  # Weighted accumulation
        total_samples += len(batch["input_ids"])
        metric_accumulators['loss'] += loss.item() * len(batch["input_ids"])
        logger.info(f"Completed batch {i+1} out of {n_batches}")

    # Compute average metrics over the entire dataset
    average_metrics = {metric: float(value / total_samples) for metric, value in metric_accumulators.items()}
    average_metrics['train_loss'] = float(train_loss)

    # Log the final metrics
    logger.info(f"Average Metrics over Test Set: {pprint.pformat(average_metrics)}")

    # Stop carbon tracker
    tracker.stop()

    # save metrics to dvc yaml file
    with open('./data/nomelt-model/test_metrics.yaml', 'w') as f:
        dump(average_metrics, f)

if __name__ == "__main__":
    main()


    