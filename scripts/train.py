from evaluate import load
import numpy as np
import duckdb
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoConfig
import dvc.api
import re

import logging
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'


def evaluate_model(trainer, dataset):
    # Evaluate model
    metrics = trainer.evaluate(eval_dataset=dataset)
    return metrics

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

    # Load parameters from DVC
    params = dvc.api.get_params(stages='train')

    # Load data and tokenizer
    dataset = load_from_disk('./data/dataset/')
    tokenizer = AutoTokenizer.from_pretrained(params['pretrained_model'])

    # preprocess data with tokenizer
    def prepare_string(string):
        out = ' '.join(string)
        out = re.sub(r"[UZOB]", "X", out)
        return out
    def preprocess_dataset_to_model_inputs(examples):
        meso_seqs = [prepare_string(seq) for seq in examples['meso_seq']]
        thermo_seqs = [prepare_string(seq) for seq in examples['thermo_seq']]
        # need eos token at end of thermo seqs
        thermo_seqs = [seq + tokenizer.eos_token for seq in thermo_seqs]

        # tokenize inputs and outputs
        model_inputs = tokenizer(meso_seqs, max_length=params['max_length'], padding='max_length', truncation=True)
        labels = tokenizer(thermo_seqs, max_length=params['max_length'], padding='max_length', truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    dataset = dataset.map(preprocess_dataset_to_model_inputs, batched=True, remove_columns=['meso_seq', 'thermo_seq'])

    # compute number of steps per save and evalution
    # TODO
    steps_per_save = 100

    # load the model with potentially custom config
    model_config = AutoConfig.from_pretrained(params['pretrained_model'])
    for model_hyperparam in params['model_hyperparams']:
        setattr(model_config, model_hyperparam, params['model_hyperparams'][model_hyperparam])
    model = AutoModelForSeq2SeqLM.from_pretrained(params['pretrained_model'], config=model_config)

    # collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Training arguments
    args = Seq2SeqTrainingArguments(
        output_dir='./data/nomelt-model',
        # Trainer meta parameters
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=steps_per_save,
        save_strategy='steps',
        save_steps=steps_per_save,
        save_total_limit=20,
        logging_strategy='steps',
        logging_steps=1,
        predict_with_generate=True,
        load_best_model_at_end=True,
        # training parameters
        num_train_epochs=params['epochs'],
        # batches
        per_device_train_batch_size=params['per_device_batch_size'],
        per_device_eval_batch_size=params['per_device_batch_size'],
        gradient_accumulation_steps=params['gradient_accumulation'],
        gradient_checkpointing=params['gradient_checkpointing'],
        auto_find_batch_size=True,
        # optimizer
        learning_rate=params['learning_rate'],
        lr_scheduler_type=params['lr_scheduler_type'],
        warmup_ratio=params['warmup_ratio'],
        optim=params['optim'],
        optim_args=params['optim_args'],
        # precision
        fp16=params['fp16'],
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
        predictions = [''.join(seq.split()) for seq in predictions]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = [''.join(seq.split()) for seq in labels]
        out_metrics = {}
        # tranlsation error rate
        ter_metric = load('ter')
        out_metrics.update(ter_metric.compute(predictions=predictions, references=labels, normalized=True, case_sensitive=False))
        # rouge
        rouge_metric = load('rouge')
        out_metrics.update(rouge_metric.compute(predictions=predictions, references=labels, rouge_types=['rouge2', 'rougeL'], use_stemmer=True, use_aggregator=True))
        # google bleu
        bleu_metric = load('bleu')
        out_metrics.update(bleu_metric.compute(predictions=predictions, references=labels, max_len=5))
        return out_metrics

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    

if __name__ == "__main__":
    main()
