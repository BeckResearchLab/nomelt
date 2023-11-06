import accelerate
import codecarbon
from dvclive import Live
from datetime import timedelta
import logging
import numpy as np
import os
import json
import re
import torch
import tqdm
from pynvml import *
import pprint
from yaml import safe_load

from datasets import load_from_disk, concatenate_datasets
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
class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Adding weights to the loss function.
        """
        # if weights, hijack the function
        if 'weight' in inputs:
            weights = inputs.pop('weight').view(-1,1) # should be shape (batch_size, 1)
        else:
            weights = torch.ones(inputs['labels'].shape[0], 1).to(inputs['labels'].device)
        assert weights.shape == (inputs['labels'].shape[0], 1)
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        # tile weights to match label shape
        weights = weights.repeat(1, labels.shape[1])
        assert weights.shape == labels.shape
        # mask weights 
        weights = torch.where(labels == -100, 0, weights)
        # compute loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100, label_smoothing=self.args.label_smoothing_factor)
        losses = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        # weight losses
        losses = losses * weights.view(-1)
        loss = torch.sum(losses) / torch.sum(weights)

        return (loss, outputs) if return_outputs else loss
    
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if not 'weight' in self._signature_columns:
            self._signature_columns.append('weight')

# preprocess data with tokenizer
def prepare_string(string):
    out = ' '.join(string)
    out = re.sub(r"[UZOB]", "X", out)
    return out

def _preprocess_dataset_to_model_inputs(examples, tokenizer, params):
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
    model_inputs = tokenizer(in_seqs, max_length=params['model']['generation_max_length'], padding='longest', truncation=True)
    labels = torch.tensor(tokenizer(out_seqs, max_length=params['model']['generation_max_length'], padding='longest', truncation=True)['input_ids'])
    # fill -100s to be ignored in loss
    labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
    model_inputs['labels'] = labels
    # if there are weights, we have to 
    if 'weight' in examples:
        model_inputs['weight'] = examples['weight']
    return model_inputs

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
    accelerator = accelerate.Accelerator(kwargs_handlers=[accelerate.InitProcessGroupKwargs(timeout=timedelta(minutes=120))])

    # Load parameters from DVC
    # retry until successful
    with open('./params.yaml', 'r') as f:
        params = safe_load(f)
    logger.info(f"Loaded params: {params}")

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

    # load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(params['model']['pretrained_model'])
    except:
        tokenizer = T5Tokenizer.from_pretrained(params['model']['pretrained_model'])
    # call tokenizer with same params as in trasnform to try to fix caching of tokenization
    # see https://github.com/huggingface/datasets/issues/5985
    _ = tokenizer(['M V Y M M', 'M V Y'], max_length=params['model']['generation_max_length'], padding='longest', truncation=True)

    # make a cache file for dataset operations
    if not os.path.exists('./data/ds_cache'):
        os.makedirs('./data/ds_cache')

    # Load and sample as necessary
    with accelerator.main_process_first():

        # need dataset
        dataset = load_from_disk('./data/dataset/')
        logger.info(f"Loaded dataset. Train, eval, test size: {(len(dataset['train']), len(dataset['eval']), len(dataset['test']))}. Combining")
        og_train_size = len(dataset['train'])
        dataset = concatenate_datasets([dataset['train'], dataset['eval'], dataset['test']])

        if params['training']['dev_sample_data']:
            if len(dataset) <= params['training']['dev_sample_data']:
                pass
            else:
                dataset = dataset.select(range(params['training']['dev_sample_data']))

            logger.info(f"Sampling {params['training']['dev_sample_data']} sequences from train and test sets. New train size: {(len(dataset))}")

        preprocess_dataset_to_model_inputs = lambda examples: _preprocess_dataset_to_model_inputs(examples, tokenizer, params)
        removal_columns = [c for c in dataset.column_names]
        logger.info(f"Tokenizing and preparing model inputs. Using task '{params['model']['task']}'. 'tranlsation' is meso to thermo, 'reconstruction' is meso to meso or thermo to thermo.")
        dataset = dataset.map(
            preprocess_dataset_to_model_inputs,
            batched=True,
            load_from_cache_file=True,
            cache_file_name='./data/ds_cache/full_dataset_tokenized.arrow',
            desc='Tokenizing and preparing model inputs')

        if params['training']['reweight']:
            logger.info('Applying reweighting based on cluster size.')
            # compute cluster sizes in train set and use to compute sample weight. A cluster with 1 sequence will have weight 1.
            # first count the number of sequences in each cluster
            if 'cluster_counts.json' in os.listdir('./data/dataset/'):
                with open('./data/dataset/cluster_counts.json', 'r') as f:
                    cluster_sizes = json.load(f)
            else:
                cluster_sizes = {}
                for ex in tqdm.tqdm(dataset, total=len(dataset)):
                    c = ex['cluster']
                    cluster_sizes[c] = cluster_sizes.get(c, 0) + 1
                with open('./data/dataset/cluster_counts.json', 'w') as f:
                    json.dump(cluster_sizes, f)
            # then compute the weight for each sequence
            def do_one(ex):
                c = ex['cluster']
                ex['weight'] = 1/cluster_sizes[c]
                return ex
            dataset = dataset.map(
                do_one,
                batched=False,
                desc='Computing sample weights',
                num_proc=32,
                load_from_cache_file=True,
                cache_file_name='./data/ds_cache/full_dataset_weighted.arrow',
            )
            logger.info(f"Compute sample weights.")
        
        keep_columns = ['input_ids', 'attention_mask', 'labels']
        if 'weight' in dataset.column_names:
            keep_columns.append('weight')
        dataset.set_format(type='torch', columns=keep_columns)

    logger.info(f"Final dataset format: {dataset}")

    # compute number of steps per save and evalution
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
    else:
        num_devices = 1

    # we need to get the number of epochs the previous model trained and replicate that
    ckpt_files = [f for f in os.listdir('./data/nomelt-model/model') if 'checkpoint' in f]
    latest_ckpt = max([int(f.split('-')[1]) for f in ckpt_files])

    new_steps_to_train = int(latest_ckpt * (len(dataset) / og_train_size)) + 1
    logger.info(f"Training for {new_steps_to_train} steps.")

    # Training arguments
    args = Seq2SeqTrainingArguments(
        output_dir='./data/nomelt-model-full/model',
        # Trainer meta parameters
        log_level='debug',
        do_train=True,
        do_eval=False,
        logging_strategy='steps',
        logging_steps=1,
        # training parameters
        max_steps=new_steps_to_train,
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

    # load the model with potentially custom config
    model_config = AutoConfig.from_pretrained(params['model']['pretrained_model'])
    for model_hyperparam in params['model']['model_hyperparams']:
        setattr(model_config, model_hyperparam, params['model']['model_hyperparams'][model_hyperparam])
    setattr(model_config, 'max_length', params['model']['generation_max_length'])
    model = AutoModelForSeq2SeqLM.from_pretrained(params['model']['pretrained_model'], config=model_config)

    if params['training']['freeze_early_layers']:
        if type(params['training']['freeze_early_layers']) != float:
            raise ValueError("If freezing layers, expecting fraction of layers from bottom to freeze")
        encoder_stack_size = len(model.encoder.block)
        decoder_stack_size = len(model.decoder.block)
        enc_layers_to_freeze = int(encoder_stack_size * params['training']['freeze_early_layers'])
        dec_layers_to_freeze = int(decoder_stack_size * params['training']['freeze_early_layers'])
        logger.info(f"Freezing {enc_layers_to_freeze} encoder layers and {dec_layers_to_freeze} decoder layers")
        for i, t5block in enumerate(model.encoder.block):
            if i < enc_layers_to_freeze:
                for param in t5block.parameters():
                    param.requires_grad = False
        for i, t5block in enumerate(model.decoder.block):
            if i < dec_layers_to_freeze:
                for param in t5block.parameters():
                    param.requires_grad = False

    # collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # main process is prepared, all processes and begin
    accelerator.wait_for_everyone()
    del accelerator
    logger.info("All processes caught up. Beginning training.")

    # get trainer going
    TrainerClass = WeightedSeq2SeqTrainer if params['training']['reweight'] else Seq2SeqTrainer

    trainer = TrainerClass(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    if trainer.accelerator.is_main_process:
        live = Live('./data/nomelt-model-full/live', dvcyaml=False, report='md')
        trainer.add_callback(nomelt.dvclive.DVCLiveCallback(live=live, model_file='./data/nomelt-model-full/model'))

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
