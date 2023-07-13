import codecarbon
import dvc.api
from dvclive import Live
import dvclive.huggingface
from dvclive.huggingface import DVCLiveCallback
import logging
import numpy as np
import os
import re
import torch

from datasets import load_from_disk
from duckdb import connect as duckdb_connect
from evaluate import load
import transformers.integrations
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
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
    params = dvc.api.params_show(stages='train')

    # start carbon tracker
    tracker = codecarbon.OfflineEmissionsTracker( 
        project_name="train",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    tracker.start()

    # Load and sample as necessary
    dataset = load_from_disk('./data/dataset/')
    logger.info(f"Loaded dataset. Train, test size: {(len(dataset['train']), len(dataset['test']))}")
    if params['training']['keep_only_extremes']:
        dataset = dataset.filter(lambda x: x['status_in_cluster'] in ['extreme', 'unique'])
        logger.info(f"Keeping only extreme cluster and unique sequences. New train, val, test size: {(len(dataset['train']), len(dataset['eval']), len(dataset['test']))}")
    if params['training']['dev_sample_data']:
        if len(dataset['train']) <= params['training']['dev_sample_data']:
            pass
        else:
            dataset['train'] = dataset['train'].select(range(params['training']['dev_sample_data']))
        if len(dataset['eval']) <= params['training']['dev_sample_data']:
            pass
        else:
            dataset['eval'] = dataset['eval'].select(range(params['training']['dev_sample_data']))
        if len(dataset['test']) <= params['training']['dev_sample_data']:
            pass
        else:
            dataset['test'] = dataset['test'].select(range(params['training']['dev_sample_data']))
        logger.info(f"Sampling {params['training']['dev_sample_data']} sequences from train and test sets. New train, test size: {(len(dataset['train']), len(dataset['test']))}")
    if params['training']['max_eval_examples'] < len(dataset['eval']):
        dataset['eval'] = dataset['eval'].select(range(params['training']['max_test_examples']))
        logger.info(f"Sampling {params['training']['max_eval_examples']} sequences from eval set. New eval size: {len(dataset['eval'])}")

    # load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(params['model']['pretrained_model'])
    except:
        tokenizer = T5Tokenizer.from_pretrained(params['model']['pretrained_model'])

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
        model_inputs = tokenizer(in_seqs, max_length=params['training']['max_length'], padding='max_length', truncation=True)
        labels = torch.tensor(tokenizer(out_seqs, max_length=params['training']['max_length'], padding='max_length', truncation=True)['input_ids'])
        # fill -100s to be ignored in loss
        labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
        model_inputs['labels'] = labels
        return model_inputs
    logger.info(f"Tokenizing and preparing model inputs. Using task '{params['model']['task']}'. 'tranlsation' is meso to thermo, 'reconstruction' is meso to meso or thermo to thermo.")
    dataset = dataset.map(preprocess_dataset_to_model_inputs, batched=True, remove_columns=['meso_seq', 'thermo_seq'], load_from_cache_file=False, desc='Tokenizing and preparing model inputs')

    # compute number of steps per save and evalution
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
    else:
        num_devices = 1
    if params['training']['saves_per_epoch'] == None:
        save_strat = 'no'
        steps_per_save = None
        logger.info(f"Using {num_devices} devices. No saves per epoch.")
    else:
        save_strat='steps'
        step_size = params['training']['per_device_batch_size'] * params['training']['gradient_accumulation'] * num_devices
        steps_per_epoch = len(dataset['train']) / step_size
        if params['training']['saves_per_epoch'] == 0:
            params['training']['saves_per_epoch'] = 1
        steps_per_save = (steps_per_epoch) // params['training']['saves_per_epoch']
        if steps_per_save == 0:
            steps_per_save = 1
        logger.info(f"Using {num_devices} devices. Steps per epoch: {steps_per_epoch}. Steps per save: {steps_per_save}")

    # load the model with potentially custom config
    model_config = AutoConfig.from_pretrained(params['model']['pretrained_model'])
    for model_hyperparam in params['model']['model_hyperparams']:
        setattr(model_config, model_hyperparam, params['model']['model_hyperparams'][model_hyperparam])
    model = AutoModelForSeq2SeqLM.from_pretrained(params['model']['pretrained_model'], config=model_config)

    # collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Training arguments
    args = Seq2SeqTrainingArguments(
        output_dir='./data/nomelt-model/model',
        # Trainer meta parameters
        log_level='debug',
        do_train=True,
        do_eval=True,
        evaluation_strategy=save_strat,
        eval_steps=steps_per_save,
        save_strategy=save_strat,
        save_steps=steps_per_save,
        save_total_limit=20,
        logging_strategy='steps',
        logging_steps=1,
        predict_with_generate=True,
        generation_max_length=params['training']['max_length'],
        load_best_model_at_end=True,
        # training parameters
        num_train_epochs=params['training']['epochs'],
        # batches
        per_device_train_batch_size=params['training']['per_device_batch_size'],
        per_device_eval_batch_size=params['training']['per_device_batch_size'],
        gradient_accumulation_steps=params['training']['gradient_accumulation'],
        gradient_checkpointing=params['training']['gradient_checkpointing'],
        auto_find_batch_size=params['training']['auto_find_batch_size'],
        # optimizer
        learning_rate=params['training']['learning_rate'],
        lr_scheduler_type=params['training']['lr_scheduler_type'],
        warmup_ratio=params['training']['warmup_ratio'],
        optim=params['training']['optim'],
        optim_args=params['training']['optim_args'],
        label_smoothing_factor=params['training']['label_smoothing_factor'],
        # precision
        fp16=params['training']['fp16'],
    )

    # define metrics to compute
    def compute_metrics(outputs):
        """We use:
        - ter: translation edit rate
        - Rouge 2: F1 score for bigrams
        - Rouge L: score for longest common subsequences
        - google_bleu: single sentence BLEU like score, minimum of recall and precision on 1, 2, 3, and 4 grams
        """
        # outputs encoded
        predictions, labels = outputs.predictions, outputs.label_ids
        # decode
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
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

    with Live('./data/nomelt-model/live', dvcyaml=False, report='md') as live:
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['eval'],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.add_callback(dvclive.huggingface.DVCLiveCallback(live=live))
        trainer.pop_callback(transformers.integrations.TensorBoardCallback)
        
        # compute initial scores
        initial_eval = trainer.evaluate()
        logger.info(f"Eval scores at begining of training: {initial_eval}")

        # train
        trainer.train()

        # compute final score
        final_eval = trainer.evaluate(dataset['test'], metric_key_prefix='test')
        trainer.save_model()
        logger.info(f"Test scores at end of training: {final_eval}")
    tracker.stop()

if __name__ == "__main__":
    main()
