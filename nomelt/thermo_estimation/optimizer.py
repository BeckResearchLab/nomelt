import os
from contextlib import contextmanager
import multiprocessing
import subprocess
import optuna
import Bio.pairwise2
from Bio.Align import substitution_matrices
from typing import List, Dict, Tuple
from functools import partial
from dataclasses import dataclass
import torch
import MDAnalysis.analysis.align
import pandas as pd

import nomelt.thermo_estimation.estimators

import logging
logger = logging.getLogger(__name__)

from typing import Union

N_GPUS = torch.cuda.device_count()

class GpuQueue:
    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(N_GPUS)) if N_GPUS > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


@dataclass
class Mutation:
    positions: Union[int, Tuple[int, int]] #if tuple, then it is multiple amino acids
    wt: str
    variant: str

@dataclass
class OptimizerArgs:
    n_trials: int = 100
    direction: str = 'minimize'  # or 'maximize'
    sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler()
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    cut_tails: Union[None, int] = None # number of gap spaces to keep om ends of the alignment
    gap_compressed_mutations: bool = False # whether to consider a string of gaps a single mutation
    matrix: str = None
    match_score: int = 1
    mismatch_score: int = -1
    gapopen: int = -2
    gapextend: int = -1
    penalize_end_gaps: bool = False

class MutationSubsetOptimizer:
    def __init__(self, wt: str, variant: str, estimator: nomelt.thermo_estimation.estimators.ThermoStabilityEstimator, name: str = None, args: OptimizerArgs = OptimizerArgs(), estimator_args=None):
        self.wt = wt
        self.variant = variant
        self.estimator = estimator
        self.params = args
        self.estimator_args = {'args':estimator_args} if estimator_args else {}
        self.study=None
        if name is None:
            name = "mutation_optimizer_run"
        self.name = name
        self._set_mutation_set()
        self._trial_seq_pdb = []

    def _set_mutation_set(self):
        if self.params.matrix is not None:
            mat = substitution_matrices.load(self.params.matrix)
            alignment = Bio.pairwise2.align.globalds(
                self.wt, self.variant,
                mat,
                self.params.gapopen,
                self.params.gapextend,
                one_alignment_only=True,
                penalize_extend_when_opening=False,
                penalize_end_gaps=self.params.penalize_end_gaps)
        else:
            alignment = Bio.pairwise2.align.globalms(
                self.wt, self.variant,
                self.params.match_score,
                self.params.mismatch_score,
                self.params.gapopen,
                self.params.gapextend,
                one_alignment_only=True,
                penalize_extend_when_opening=False,
                penalize_end_gaps=self.params.penalize_end_gaps)
        
        logger.info("Aligned sequences to determined mutation set:")
        logger.info('\n'+Bio.pairwise2.format_alignment(*alignment[0], full_sequences=True))

        self.aligned_wt, match_line, self.aligned_variant = Bio.pairwise2.format_alignment(*alignment[0], full_sequences=True).split('\n')[:3]
        # if we cut tails, we want to remove end gaps of the wt sequence down to a few gap positions specified by params.cut_tails
        if self.params.cut_tails is not None:
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
            logger.info("Aligned sequences after cutting tails:\n")
            logger.info('\n'.join([self.aligned_wt, match_line, self.aligned_variant]))

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
            logger.info(f"Compressed gaps where applicable. Orignal mutation set size: {original_mutation_set_size}, new size {len(self.mutation_set)}")

        # print out a nice image of the alignment string with the mutations seperated by |
        mutation_starts = []
        mutation_ends = []
        for mutation in self.mutation_set.values():
            if isinstance(mutation.positions, list):
                mutation_starts.append(mutation.positions[0])
                mutation_ends.append(mutation.positions[1]+1)
            else:
                mutation_starts.append(mutation.positions)
                mutation_ends.append(mutation.positions+1)
        wt_str = ""
        variant_str = ""
        alignment_str = ""
        count = 0
        for i, (wt, match, variant) in enumerate(zip(self.aligned_wt, match_line, self.aligned_variant)):
            if i in mutation_starts:
                wt_str += " |  "
                variant_str += " |  "
                if len(str(count)) == 1:
                    alignment_str += f" {count}  "
                elif len(str(count)) == 2:
                    alignment_str += f" {count} "
                else:
                    alignment_str += f" {count}"
                count +=1
            elif i in mutation_ends:
                wt_str += " | "
                variant_str += " | "
                alignment_str += " | "   
            wt_str += wt
            variant_str += variant
            if wt == variant:
                alignment_str += '='
            else:
                alignment_str += ' '
        # we now need to wrap these three strings so that the three lines get shifted down
        # to new lines when they get to 79 characters
        # but the three lines need to still be aligned so that they are viewed in chunks
        # of 79 characters
        wt_chunks = [wt_str[i:i+79] for i in range(0, len(wt_str), 79)]
        variant_chunks = [variant_str[i:i+79] for i in range(0, len(variant_str), 79)]
        alignment_chunks = [alignment_str[i:i+79] for i in range(0, len(alignment_str), 79)]
        net_string = ""
        for wt_chunk, variant_chunk, alignment_chunk in zip(wt_chunks, variant_chunks, alignment_chunks):
            net_string += '\n'.join([wt_chunk, alignment_chunk, variant_chunk]) + '\n\n'

        logger.info("Aligned sequences with mutations marked:")
        logger.info('\n'+net_string)

    def _get_variant_sequence(self, mutation_subset: List[str]) -> str:
        variant = list(self.aligned_wt)  # Create a mutable copy
        for mutation_key in mutation_subset:
            mutation = self.mutation_set[mutation_key]
            if isinstance(mutation.positions, list):
                for i, position in enumerate(range(mutation.positions[0], mutation.positions[1]+1)):
                    variant[position] = mutation.variant[i]
            else:
                variant[mutation.positions] = mutation.variant
        return self.clean_gaps("".join(variant))

    def _call_estimator(self, mutation_subset: List[str], gpu_i: int=None) -> float:
        logger.debug(f"Calling estimator with mutation subset: {mutation_subset}")
        variant_sequence = self._get_variant_sequence(mutation_subset)
        logger.debug(f"Variant sequence: {variant_sequence}")
        sequences = [variant_sequence]
        estimator = self.estimator(sequences, [self.hash_mutation_set(mutation_subset)], **self.estimator_args)
        # override wome parameters
        estimator.args.gpu_i = gpu_i
        estimator.args.wdir = os.path.abspath(f'./tmp/{self.name}')
        return estimator.run()[self.hash_mutation_set(mutation_subset)], variant_sequence, estimator.pdb_files_dict[self.hash_mutation_set(mutation_subset)]

    def run(self, n_jobs: int =1):
        if self.params.direction == 'minimize':
            study = optuna.create_study(sampler=self.params.sampler, pruner=self.params.pruner, study_name=self.name)
        elif self.params.direction == 'maximize':
            study = optuna.create_study(sampler=self.params.sampler, pruner=self.params.pruner, direction='maximize', study_name=self.name)
        else:
            raise ValueError('Invalid direction for optimization. Please choose "minimize" or "maximize".')
        self.study=study
        study.optimize(OptunaObjective(self, GpuQueue()), n_trials=self.params.n_trials, n_jobs=n_jobs, show_progress_bar=True)
        return study.best_params, study.best_value, study

    @property
    def best_mutations(self):
        if self.study is None:
            raise ValueError("Optimized sequence not yet determined, call `run`")
        else:
            return [k for k in self.mutation_set.keys() if self.study.best_params[k]]

    @property
    def optimized_sequence(self):
        best_mutations = self.best_mutations
        best_variant = self._get_variant_sequence(best_mutations)
        return best_variant

    @property
    def structure_seq_trajectory(self):
        if len(self._trial_seq_pdb) == 0:
            raise ValueError('Must run optimizer before a trajecotry of variants and structures will be produced.')
        else:
            return self._trial_seq_pdb
    
    @staticmethod
    def clean_gaps(sequence: str) -> str:
        return sequence.replace("-", "")

    @staticmethod
    def hash_mutation_set(mutation_set: List[str]) -> str:
        return hash("_".join(sorted(mutation_set)))

class OptunaObjective:
    def __init__(self, optimizer: MutationSubsetOptimizer, gpu_queue: GpuQueue):
        self.gpu_queue = gpu_queue
        self.opt = optimizer

    def __call__(self, trial: optuna.trial.Trial) -> float:
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            selected_mutations = [
                mutation_key for mutation_key, include in
                ((mutation_key, trial.suggest_categorical(mutation_key, [True, False])) for mutation_key in self.opt.mutation_set.keys())
                if include
            ]
            result, variant, pdb_file = self.opt._call_estimator(selected_mutations, gpu_i=gpu_i)
            self.opt._trial_seq_pdb.append(
                (trial.number, variant, pdb_file)
            )
        return result  # Optuna minimizes by default, so return negative result to maximize


class OptTrajSuperimposer:
    """Class to superimpose structures from an optimization trajectory onto the first structure.
    
    Requires MDAnalysis
    First runs sequence alignment (Clustal) on sequences to the reference. Atom selection is made from this
    for the 3D alignment.
    """
    def __init__(
        self,
        optimizer: MutationSubsetOptimizer = None,
        sequences: List[str]=None,
        structure_files: List[str]=None,
        trials_dataframe: pd.DataFrame = None,
        output_dir: str = None,
    ):
        if optimizer is not None:
            self.sequences, self.structure_files = self._get_sorted_opt_traj(optimizer)
            self.optimizer = optimizer
        elif sequences is not None and structure_files is not None and trials_dataframe is not None:
            self.sequences = sequences
            self.structure_files = structure_files
            self.trials_dataframe = trials_dataframe
            self.optimizer = None
        else:
            raise ValueError("Must provide either an optimizer or sequences and structure files.")
        self._parse_universes()
        #create temp dir
        self.temp_dir = os.path.abspath('./tmp/opt_traj_superimposer')
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
        if output_dir is None:
            self.output_dir = self.temp_dir
        else:
            self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.output_files = None

    @staticmethod
    def _get_sorted_opt_traj(optimizer):
        _, sequences, pdb_files = zip(*sorted(optimizer.structure_seq_trajectory, key=lambda x: x[0]))
        return sequences, pdb_files
    
    def _parse_universes(self):
        self.universes = []
        for structure_file in self.structure_files:
            self.universes.append(MDAnalysis.Universe(structure_file))

    @property
    def ref_struct(self):
        return self.universes[0]
    
    def _superimpose_one(self, idx: int):
        """Superimpose one structure onto the reference structure.
        
        Use MDAnalysys fasta2select to make the selection of equivalent atoms
        https://docs.mdanalysis.org/1.0.1/documentation_pages/analysis/align.html#MDAnalysis.analysis.align.fasta2select
        """
        # create temporary fasta file
        with open(os.path.join(self.temp_dir, 'temp.fasta'), 'w') as f:
            f.write(f">ref\n{self.sequences[0]}\n>variant\n{self.sequences[idx]}\n")
        # make the alignment selection
        selection = MDAnalysis.analysis.align.fasta2select(
            os.path.join(self.temp_dir, 'temp.fasta'),
        )
        # superimpose
        old_rmsd, new_rmsd = MDAnalysis.analysis.align.alignto(
            self.universes[idx],
            self.ref_struct,
            select=selection,
            weights="mass",
        )
        logger.info(f"Superimposed structure {idx} onto reference structure with RMSD {old_rmsd} to {new_rmsd}")

    def run(self):
        # superimpose all structures onto the reference
        for i in range(1, len(self.universes)):
            self._superimpose_one(i)

        # save the trajectory
        return self._save_trajectory()

    def _save_trajectory(self):
        # write the new files to output dir
        output_files = []
        for i, universe in enumerate(self.universes):
            out = os.path.join(self.output_dir, f"structure_{i:04}.pdb")
            universe.atoms.write(out)
            output_files.append(out)
        self.output_files = output_files
        return output_files
    
    def _vmd_script_single(self, id, filepath):
        outfile = self.output_dir+f"/frame_{id:04}.tga"
        script = f"""
        mol new {filepath} type pdb first 0 last -1 step 1 waitfor 1 molid {id}
        mol selection all
        mol modcolor 0 {id} ResName
        mol color ResName
        mol modmaterial 0 {id} RTChrome
        mol material RTChrome
        mol modstyle 0 {id} Ribbons 0.000000 20.000000 5.000000
        mol representation Ribbons 0.000000 20.000000 5.000000
        axes location Off
        render TachyonInternal {outfile} /usr/bin/open %s -fullshade -auto_skylight 0.7 -aasamples 12 %s -res 6000 4000 -o %s.png
        mol delete {id}
        """
        return script, outfile

    def _run_vmd_script(self, vmd_script_file: str = './vmd_script.tcl'):
        """Write a vmd script to take the files and make a movie"""
        if self.output_files is None:
            raise ValueError("Must run `run` before writing VMD script.")
        png_files = []

        with open(vmd_script_file, 'w') as f:
            # Load remaining files as additional states for the molecule
            for i, pdb in enumerate(self.output_files):
                script, outfile = self._vmd_script_single(i, pdb)
                f.write(script)
                png_files.append(outfile)

        # now run the script
        result = subprocess.run(f"vmd -e {vmd_script_file}".split(), capture_output=True)
        logger.info(result.stdout.decode('utf-8'))
        return png_files
    
    def _make_movie(self, image_files, output_name):
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        from matplotlib.animation import FuncAnimation
        from matplotlib.animation import PillowWriter

        def animate_pngs(image_files, x, y):
            # Ensure all the lists are of the same length
            logger.info(f"Making movie with png files: {image_files}")
            assert len(image_files) == len(x) == len(y), "All lists must have the same length"

            # Create a figure with two subplots
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

            # First subplot: Lineplot of y vs x
            line, = ax[0].plot(x, y, '-r')
            point, = ax[0].plot([], [], 'o', color='blue', markersize=8)

            # get mean and std to set axis limits
            mean = np.mean(y)
            std = np.std(y)

            ax[0].set_xlim(np.min(x), np.max(x))
            ax[0].set_ylim(mean-3*std, mean+3*std)
            ax[0].set_xlabel("Step")
            ax[0].set_ylabel("Optimization target")

            # Second subplot: Display the first image
            img_display = ax[1].imshow(np.array(Image.open(image_files[0])))
            ax[1].axis('off')

            # Initialization function
            def init():
                point.set_data([], [])
                img_display.set_array(np.array(Image.open(image_files[0])))
                return point, img_display

            # Update function for animation
            def update(frame):
                point.set_data(x[frame], y[frame])
                img_display.set_array(np.array(Image.open(image_files[frame])))
                return point, img_display

            ani = FuncAnimation(fig, update, frames=len(image_files), init_func=init, blit=True)


            # Save animation as gif
            writer = PillowWriter(fps=4)  # Change fps as needed
            ani.save(output_name, writer=writer)

            plt.tight_layout()
            plt.show()

        result_df = self.optimizer_results
        x = result_df['number']
        y = result_df['value']
        animate_pngs(image_files, x, y)

    @property
    def optimizer_results(self):
        if self.optimizer is None:
            return self.trials_dataframe
        else:
            return self.optimizer.study.trials_dataframe()

    def make_optimization_movie(self):
        """Make a visual of the optimization process

        Requires vmd
        """
        if self.output_files is None:
            raise ValueError("Must run `run` before making movie.")
        
        logger.info("Running VMD script to visualize aligned structures into images...")
        png_files = self._run_vmd_script(vmd_script_file=os.path.join(self.output_dir, 'vmd_script.tcl'))

        logger.info("Making movie...")
        self._make_movie(png_files, os.path.join(self.output_dir, 'optimization_movie.gif'))

        



        