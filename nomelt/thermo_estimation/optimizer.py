
import optuna
import Bio.pairwise2
from Bio.Align import substitution_matrices
from typing import List, Dict, Tuple
from functools import partial
from dataclasses import dataclass

import nomelt.thermo_estimation.estimators

import logging
logger = logging.getLogger(__name__)

from typing import Union

@dataclass
class Mutation:
    positions: Union[int, Tuple[int, int]] #if tuple, then it is multiple amino acids
    wt: str
    variant: str

@dataclass
class OptimizerParams:
    n_trials: int = 100
    direction: str = 'minimize'  # or 'maximize'
    sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler()
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    cut_tails: Union[None, int] = 0 # number of gap spaces to keep om ends of the alignment
    gap_compressed_mutations: bool = False # whether to consider a string of gaps a single mutation
    matrix: str = 'BLOSUM62'
    gapopen: int = -11
    gapextend: int = -1
    penalize_end_gaps: bool = False

class MutationSubsetOptimizer:
    def __init__(self, wt: str, variant: str, estimator: nomelt.thermo_estimation.estimators.ThermoStabilityEstimator, params: OptimizerParams, estimator_args=None):
        self.wt = wt
        self.variant = variant
        self.estimator = estimator
        self.params = params
        self.estimator_args = estimator_args if estimator_args else {}
        self._set_mutation_set()

    def _set_mutation_set(self):
        mat = substitution_matrices.load(self.params.matrix)
        alignment = Bio.pairwise2.align.globalds(
            self.wt, self.variant, mat,
            self.params.gapopen,
            self.params.gapextend,
            one_alignment_only=True,
            penalize_extend_when_opening=False,
            penalize_end_gaps=self.params.penalize_end_gaps)
        
        logger.info("Aligned sequences to determined mutation set:")
        logger.info(Bio.pairwise2.format_alignment(*alignment[0], full_sequences=True))

        self.aligned_wt, match_line, self.aligned_variant = alignment[:3]
        # if we cut tails, we want to remove end gaps of the wt sequence down to a few gap positions specified by params.cut_tails
        if self.params.cut_tails:
            # get number of gap characters on the left and right of wt sequence
            left_gaps = len(self.aligned_wt) - len(self.aligned_wt.lstrip('-'))
            right_gaps = len(self.aligned_wt) - len(self.aligned_wt.rstrip('-'))
            # cut down to the specified number
            if left_gaps > self.params.cut_tails:
                self.aligned_wt = self.aligned_wt[left_gaps-self.params.cut_tails:]
                match_line = match_line[left_gaps-self.params.cut_tails:]
                self.aligned_variant = self.aligned_variant[left_gaps-self.params.cut_tails:]
            if right_gaps > self.params.cut_tails:
                self.aligned_wt = self.aligned_wt[:-right_gaps+self.params.cut_tails]
                match_line = match_line[:-right_gaps+self.params.cut_tails]
                self.aligned_variant = self.aligned_variant[:-right_gaps+self.params.cut_tails]
            logger.info("Aligned sequences after cutting tails:")
            logger.info(Bio.pairwise2.format_alignment(self.aligned_wt, match_line, self.aligned_variant, full_sequences=True))

        # now determine mutations. 
        self.mutation_set = {
            f"mut_{i}": Mutation(i, wt=wt, variant=variant)
            for i, (wt, variant) in enumerate(zip(self.aligned_wt, self.aligned_variant)) if wt != variant
        }
        # if gap compressed, go back through and combine mutations where there is a string of gaps
        original_mutation_set_size = len(self.mutation_set)
        if self.params.gap_compressed_mutations:
            # start with the wild type side
            gap_string = '-'
            aggregate_mutation = None
            for mutation in list(self.mutation_set.values()):
                if mutation.wt == gap_string and aggregate_mutation == None:
                    aggregate_mutation = mutation
                    # rework the mutation to be a tuple of positions
                    aggregate_mutation.positions = [aggregate_mutation.positions, aggregate_mutation.positions]
                elif mutation.wt == gap_string and aggregate_mutation != None:
                    # update the aggregate mutation with the new ones information
                    aggregate_mutation.positions[1] = mutation.positions
                    aggregate_mutation.variant += mutation.variant
                    aggregate_mutation.wt += mutation.wt
                    # remove the single mutation
                    del self.mutation_set[f"mut_{mutation.positions}"]
                else:
                    # not a gap string
                    aggregate_mutation = None

            # now do the same for the variant side
            aggregate_mutation = None
            for mutation in list(self.mutation_set.values()):
                if mutation.variant == gap_string and aggregate_mutation == None:
                    aggregate_mutation = mutation
                    # rework the mutation to be a tuple of positions
                    aggregate_mutation.positions = [aggregate_mutation.positions, aggregate_mutation.positions]
                elif mutation.variant == gap_string and aggregate_mutation != None:
                    # update the aggregate mutation with the new ones information
                    aggregate_mutation.positions[1] = mutation.positions
                    aggregate_mutation.variant += mutation.variant
                    aggregate_mutation.wt += mutation.wt
                    # remove the single mutation
                    del self.mutation_set[f"mut_{mutation.positions}"]
                else:
                    # not a gap string
                    aggregate_mutation = None

        # print out a nice image of the alignment string with the mutations seperated by |
        mutation_positions = []
        for mutation in self.mutation_set.values():
            if isinstance(mutation.positions, list):
                mutation_positions.extend([mutation.positions[0], mutation.positions[1]+1])
            else:
                mutation_positions.extend([mutation.positions, mutation.positions+1])






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
