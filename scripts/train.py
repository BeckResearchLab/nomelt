import accelerate
import codecarbon
from dvclive import Live
import logging
import numpy as np
import os
import re
import torch
from pynvml import *
import pprint
from yaml import safe_load

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

global print_gpu_utilization
def print_gpu_utilization(index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(index)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def get_custom_filter(filter_str):
    """Convert a string into a callable thatr can be used to filter data
    based on columns in the dataset.

    Eg for input string 'status_in_cluster == "extreme"':
    >>> filter = get_custom_filter('{status_in_cluster} == "extreme"')
    >>> dataset = dataset.filter(filter)
    """
    def custom_callable(example):
        # Extract the column names by using format-style string operations
        condition = filter_str.format(**example)
        
        # Use eval to compute the result of the condition
        return eval(condition)
    
    return custom_callable

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
    accelerate_logger = logging.getLogger('accelerate')
    accelerate_logger.setLevel(getattr(logging, LOGLEVEL))
    accelerate_logger.addHandler(fh)
    # connect accelerator just to check the params and to only use one process for data processing
    accelerator = accelerate.Accelerator()

    # Load parameters from DVC
    # retry until successful
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)
    logger.info(f"Loaded params: {params}")

    # load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(params['model']['pretrained_model'])
    except:
        tokenizer = T5Tokenizer.from_pretrained(params['model']['pretrained_model'])

    logger.info(f"Got config from accelerator: {pprint.pformat(accelerator.__dict__, indent=4)}")

    if accelerator.is_main_process:
        # start carbon tracker
        tracker = codecarbon.OfflineEmissionsTracker( 
            project_name="train",
            output_dir="./data/",
            country_iso_code="USA",
            region="washington",
            api_call_interval=20,
        )
        tracker.start()

    # Load and sample as necessary
    with accelerator.main_process_first():
        dataset = load_from_disk('./data/dataset/')
        logger.info(f"Loaded dataset. Train, eval, test size: {(len(dataset['train']), len(dataset['eval']), len(dataset['test']))}")
        if params['training']['keep_only_extremes']:
            dataset = dataset.filter(lambda x: x['status_in_cluster'] in ['extreme', 'unique'])
            logger.info(f"Keeping only extreme cluster and unique sequences. New train, val, test size: {(len(dataset['train']), len(dataset['eval']), len(dataset['test']))}")

        # apply additional filters
        if params['training']['additional_filters']:
            for filter_str in params['training']['additional_filters']:
                dataset = dataset.filter(get_custom_filter(filter_str))
                logger.info(f"Applied additional filter {filter_str}. New train, val, test size: {(len(dataset['train']), len(dataset['eval']), len(dataset['test']))}")

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
            dataset['eval_sample'] = dataset['eval'].select(range(params['training']['max_eval_examples']))
            logger.info(f"Sampling {params['training']['max_eval_examples']} sequences from eval set. New eval size: {len(dataset['eval_sample'])}")

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
        # remove the unnecessary columns
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

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
    setattr(model_config, 'max_length', params['model']['generation_max_length'])
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
        prediction_loss_only=True,
        save_strategy=save_strat,
        save_steps=steps_per_save,
        save_total_limit=8,
        logging_strategy='steps',
        logging_steps=1,
        predict_with_generate=False,
        metric_for_best_model='eval_loss',
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
        bf16=params['training']['bf16'],
    )

    # main process is prepared, all processes and begin
    accelerator.wait_for_everyone()
    del accelerator

    # get trainer going
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval_sample'] if 'eval_sample' in dataset else dataset['eval'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    if trainer.accelerator.is_main_process:
        live = Live('./data/nomelt-model/live', dvcyaml=False, report='md')
        trainer.add_callback(nomelt.dvclive.DVCLiveCallback(live=live, model_file='./data/nomelt-model/model'))
    if params['training']['early_stopping']:
        trainer.add_callback(transformers.EarlyStoppingCallback(
            early_stopping_patience=params['training']['early_stopping_patience'],
            early_stopping_threshold=params['training']['early_stopping_threshold'],))
    trainer.pop_callback(transformers.integrations.TensorBoardCallback)
    trainer.pop_callback(transformers.integrations.CodeCarbonCallback) 
    trainer.accelerator.wait_for_everyone()

    result = trainer.train()

    logger.info(f"Training result: {result}")
    trainer.save_model()
    # compute final score
    if trainer.accelerator.is_main_process:
        live.end()
        tracker.stop()

if __name__ == "__main__":
    main()
