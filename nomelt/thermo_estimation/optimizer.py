
import optuna
import Bio.pairwise2
from typing import List, Dict, Tuple
from functools import partial
from dataclasses import dataclass

import nomelt.thermo_estimation.estimators

@dataclass
class Mutation:
    position: int
    wt: str
    variant: str

@dataclass
class OptimizerParams:
    n_trials: int = 100
    direction: str = 'minimize'  # or 'maximize'
    sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler()
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

class MutationSubsetOptimizer:
    def __init__(self, wt: str, variant: str, estimator: nomelt.thermo_estimation.estimators.ThermoStabilityEstimator, params: OptimizerParams, estimator_args=None):
        self.wt = wt
        self.variant = variant
        self.estimator = estimator
        self.params = params
        self.estimator_args = estimator_args if estimator_args else {}
        self._set_mutation_set()

    def _set_mutation_set(self):
        alignments = Bio.pairwise2.align.globalxx(self.wt, self.variant)
        alignment = alignments[0]
        self.aligned_wt, match_line, self.aligned_variant = alignment[:3]
        self.mutation_set = {
            f"mut_{i}": Mutation(i, wt=wt, variant=variant)
            for i, (wt, variant) in enumerate(zip(self.aligned_wt, self.aligned_variant)) if wt != variant
        }

    def _get_variant_sequence(self, mutation_subset: List[str]) -> str:
        variant = list(self.aligned_wt)  # Create a mutable copy
        for mutation_key in mutation_subset:
            mutation = self.mutation_set[mutation_key]
            variant[mutation.position] = mutation.variant
        return self.clean_gaps("".join(variant))

    def _call_estimator(self, mutation_subset: List[str]) -> float:
        variant_sequence = self._get_variant_sequence(mutation_subset)
        sequences = [variant_sequence]
        estimator = self.estimator(sequences, [self.hash_mutation_set(mutation_subset)], self.estimator_args)
        return estimator.run()[self.hash_mutation_set(mutation_subset)]
    
    def _objective(self, trial: optuna.trial.Trial) -> float:
        selected_mutations = [
            mutation_key for mutation_key, include in
            ((mutation_key, trial.suggest_categorical(mutation_key, [True, False])) for mutation_key in self.mutation_set.keys())
            if include
        ]
        result = self._call_estimator(selected_mutations)
        return result  # Optuna minimizes by default, so return negative result to maximize

    def run(self):
        if self.params.direction == 'minimize':
            study = optuna.create_study(sampler=self.params.sampler, pruner=self.params.pruner)
        elif self.params.direction == 'maximize':
            study = optuna.create_study(sampler=self.params.sampler, pruner=self.params.pruner, direction='maximize')
        else:
            raise ValueError('Invalid direction for optimization. Please choose "minimize" or "maximize".')
        study.optimize(self._objective, n_trials=self.params.n_trials)
        return study.best_params, study.best_value 

    @staticmethod
    def clean_gaps(sequence: str) -> str:
        return sequence.replace("-", "")

    @staticmethod
    def hash_mutation_set(mutation_set: List[str]) -> str:
        return hash("_".join(sorted(mutation_set)))
