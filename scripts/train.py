import duckdb
from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
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
    tokenizer = T5Tokenizer.from_pretrained(params['model_name'])

    # preprocess data with tokenizer
    def prepare_string(string):
        out = ' '.join(string)
        out = re.sub(r"[UZOB]", "X", out)
        return out
    def preprocess_dataset(examples):
        meso_seqs = [prepare_string(seq) for seq in examples['meso_seq']]
        thermo_seqs = [prepare_string(seq) for seq in examples['thermo_seq']]

    

if __name__ == "__main__":
    main()
