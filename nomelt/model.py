"""Wrap the NOMELT model up into a predictor class."""
import torch
import re
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    DataCollatorForSeq2Seq
)
import datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from typing import Iterable

class NOMELTModel:
    """Wraps up the T5 Model for easy replication of tasks in this work"""

    def __init__(
        self,
        pretrained_dir: str,
        **hyperparams
    ):
        """Initialize the model."""
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_dir)
        self.config = AutoConfig.from_pretrained(pretrained_dir)
        for k, v in hyperparams.items():
            setattr(self.config, k, v)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_dir, config=self.config).to(self.device)


    @staticmethod
    def _prepare_string(string) -> str:
        """Prepare a string for input to the model."""
        out = ' '.join(string)
        out = re.sub(r"[UZOB]", "X", out)
        return out

    def _preprocess_dataset_to_model_inputs(self, examples):
        in_col, out_col = 'meso_seq', 'thermo_seq'
        in_seqs = [self._prepare_string(seq) for seq in examples[in_col]]

        # tokenize inputs and outputs
        model_inputs = self.tokenizer(in_seqs, padding='longest')

        if out_col in examples:
            out_seqs = [self._prepare_string(seq) for seq in examples[out_col]]
            labels = torch.tensor(self.tokenizer(out_seqs, padding='longest')['input_ids'])

            labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
            model_inputs['labels'] = labels
        return model_inputs

    def _predict(self, batch):
        """Predict on a batch of data."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            # Generate predictions
            outputs = self.model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            return loss, logits

    def _decode_tensor(self, tensor):
        output_tensor = torch.where(tensor != -100, tensor, self.tokenizer.pad_token_id)
        generated_sequences = self.tokenizer.batch_decode(output_tensor, skip_special_tokens=True)
        translated_sequences = [''.join(generated_sequence.split()) for generated_sequence in generated_sequences]
        return np.array(translated_sequences).reshape(-1,1)

    def _generate(self, batch, num_beams=10, max_length=250, num_return_sequences=1, temperature=1.0, do_sample=False):
        """Generate from a batch of data."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            # Generate predictions
            outputs = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                do_sample=do_sample,
            )
            sequences = self._decode_tensor(outputs)
            if num_return_sequences > 1:
                sequences = sequences.reshape(-1, num_return_sequences)
            return sequences
    
    def predict(
        self, sequences: Iterable[str],
        labels: Iterable[str],
        batch_size: int=5
    ):
        dataset = datasets.Dataset.from_dict({'meso_seq': sequences, 'thermo_seq': labels})

        dataset = dataset.map(self._preprocess_dataset_to_model_inputs, batched=True, remove_columns=dataset.column_names, desc='Tokenizing and preparing model inputs')
        dataset.set_format(type='torch')

        def _logit_generator():
            for batch in DataLoader(dataset, batch_size=batch_size):
                loss, logits = self._predict(batch)
                for logit in logits:
                    yield {'loss': torch.tensor(loss).reshape(1,-1), 'logits': logit}
        
        out_data = datasets.Dataset.from_generator(_logit_generator)
        return out_data

    def translate_sequences(
        self,
        sequences: Iterable[str], 
        generation_max_length: int=250,
        generation_num_beams: int=10,
        generation_ensemble_size: int=None,
        temperature: float=1.0,
        batch_size: int=5
    ):
        if generation_ensemble_size is not None:
            assert generation_ensemble_size > 1, "generation_ensemble_size must be greater than 1"
            assert generation_num_beams is None, "generation_num_beams must be None if generation_ensemble_size is not None"

        dataset = datasets.Dataset.from_dict({'meso_seq': sequences})
        dataset = dataset.map(self._preprocess_dataset_to_model_inputs, batched=True, remove_columns=dataset.column_names, desc='Tokenizing and preparing model inputs')
        dataset.set_format(type='torch')

        def _seq_generator():
            with torch.no_grad():
                for batch in DataLoader(dataset, batch_size=batch_size):
                    if generation_ensemble_size is None:
                        outs = self._generate(
                            batch,
                            num_beams=generation_num_beams,
                            max_length=generation_max_length,
                        )
                    else:
                        outs = self._generate(
                            batch,
                            num_beams=1,
                            max_length=generation_max_length,
                            num_return_sequences=generation_ensemble_size,
                            temperature=temperature,
                            do_sample=True,
                        )
                    
                    for seq in outs:
                        yield {'sequences': seq}
        translated_sequences = datasets.Dataset.from_generator(_seq_generator)

        return translated_sequences
    
    @staticmethod
    def get_mutation_positions(wt, seq):
        assert len(wt) == len(seq)
        positions = []
        for i in range(len(wt)):
            if wt[i] != seq[i]:
                positions.append(i)
        return positions

    @staticmethod
    def compute_metric(wt_probs, variant_probs, normalize: bool = False):
        try:
            if normalize:
                return sum(np.log(variant_probs))/len(variant_probs) - sum(np.log(wt_probs))/len(wt_probs)
            else:
                return sum(np.log(variant_probs)) - sum(np.log(wt_probs))
        except:
            return 0.0

    def score_variants(self, wt: str, variants: Iterable[str], batch_size: int = 5, indels: bool = False):
        sequences = [wt] * (1 + len(variants))
        labels = [wt] + list(variants)
        vocab = self.tokenizer.get_vocab()

        # Check length constraints when indels are not allowed
        if not indels:
            if not all(len(variant) == len(wt) for variant in variants):
                raise ValueError("Length of variants must match wt sequence when indels=False")

        logit_data = self.predict(sequences, labels, batch_size=batch_size)  # HuggingFace dataset with 'loss' and 'logits' columns

        wt_logits = logit_data['logits'][0]
        wt_probs_all = torch.softmax(torch.tensor(wt_logits), axis=1)

        # Loop through variants and calculate probabilities
        variant_scores = []
        for i, variant in tqdm(enumerate(variants), total=len(variants), desc="Processing variants"):
            variant_logits = logit_data[i + 1]['logits'] # edited
            variant_probs_all = torch.softmax(torch.tensor(variant_logits), axis=1)

            # Use all positions if indels are present, otherwise get mutation positions
            positions = list(range(len(variant))) if indels else self.get_mutation_positions(wt, variant)

            wt_probs = [float(wt_probs_all[pos][vocab['▁' + wt[pos]]]) for pos in positions]
            variant_probs = [float(variant_probs_all[pos][vocab['▁' + variant[pos]]]) for pos in positions]

            metric = self.compute_metric(wt_probs, variant_probs)  # Implement this method as per your metric calculation
            variant_scores.append(metric)

        # now score the wt directly
        wt_probs = [float(wt_probs_all[i][vocab['▁' + wt[i]]]) for i in range(len(wt))]
        wt_score = self.compute_metric(wt_probs, wt_probs, normalize=indels)
        return wt_score, variant_scores




        

