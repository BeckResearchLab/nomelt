"""Runs AF 

NEEDS UPDATING"""
import time
import os
import subprocess
from dataclasses import dataclass, field
import numpy as np
import shutil
import yaml

from nomelt.thermo_estimation.rosetta import minimize_structures, RosettaMinimizationParameters
from nomelt.thermo_estimation.estimator import ThermoStabilityEstimator
from typing import Dict, List

import logging
logger = logging.getLogger(__name__)

@dataclass
class AlphaFoldArgs:
    use_gpu: bool = True
    models_to_relax: str = 'none'
    enable_gpu_relax: bool = True
    gpu_devices: str = 'all'
    output_dir: str = './output'
    data_dir: str = "$AF_DOWNLOAD_DIR"
    docker_image_name: str = 'alphafold'
    max_template_date: str = "2021-11-01"
    db_preset: str = 'reduced_dbs'
    model_preset: str = 'monomer'
    num_multimer_predictions_per_model: int = 5
    benchmark: bool = False
    use_precomputed_msas: bool = False
    docker_user: str = f'{os.geteuid()}:{os.getegid()}'
    base_executable_path: str = 'path/to/your/python_script.py'

    @property
    def num_models(self):
        if self.model_preset == 'monomer':
            return 5
        elif self.model_preset.startswith('model'):
            return 1
        else:
            raise ValueError(f"Unsupported model preset: {self.model_preset}") 

def write_fasta(sequences: list[str], sequence_names: list[str], fasta_files: list[str]):
    """
    Write sequences to separate FASTA files.

    Args:
        sequences (list[str]): List of protein sequences.
        sequence_names (list[str]): List of corresponding sequence names.
        fasta_files (list[str]): List of paths to write the sequences to.
    """
    if len(sequences) != len(sequence_names) or len(sequences) != len(fasta_files):
        raise ValueError("Lengths of sequences, sequence_names, and fasta_files must be the same.")

    for sequence, name, fasta_file in zip(sequences, sequence_names, fasta_files):
        with open(fasta_file, 'w') as file:
            file.write(f">{name}\n{sequence}\n")

def run_alphafold(sequences: list[str], sequence_names: list[str], args: AlphaFoldArgs):
    """
    Runs the AlphaFold prediction for a given set of sequences.

    Note that this class will overwrite AlphaFoldArgs.output_dir if it already exists, and 
    AlphaFoldArgs.precomputed_msas will be ignored. The first call with a sequence will 
    run the MSA search, and subsequent calls will use the cached results.

    Parameters:
        sequences (list[str]): A list of protein sequences to predict.
        sequence_names (list[str]): A corresponding list of sequence names (or IDs).
        args (AlphaFoldArgs): An instance of the AlphaFoldArgs dataclass containing all necessary command-line arguments.

    Notes:
    Output files of the following format:
        <target_name>/
            features.pkl               -- Pickled features used by the model.
            ranked_{0,1,2,3,4}.pdb     -- Ranked PDB models.
            ranking_debug.json         -- Debugging information for ranking.
            relax_metrics.json         -- Metrics related to the relaxation step.
            relaxed_model_{1,2,3,4,5}.pdb -- Relaxed models.
            result_model_{1,2,3,4,5}.pkl  -- Pickled result models.
            timings.json               -- Timing information for different steps.
            unrelaxed_model_{1,2,3,4,5}.pdb -- Unrelaxed models.
            msas/                      -- Directory containing MSA files.
                bfd_uniref_hits.a3m    -- MSA file for BFD UniRef hits.
                mgnify_hits.sto        -- MSA file for MGnify hits.
                uniref90_hits.sto      -- MSA file for UniRef90 hits.

    The function creates a temporary FASTA file containing the provided sequences and names, and then runs the AlphaFold script with the specified parameters. After the process is complete, the results are stored in the specified output directory, following the structure described in the notes.
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    fasta_files = [os.path.join(args.output_dir, str(name)+".fasta") for name in sequence_names]

    write_fasta(sequences, sequence_names, fasta_files)

    command = [
        'python',
        args.base_executable_path,
        f'--use_gpu={args.use_gpu}',
        f'--models_to_relax={args.models_to_relax}',
        f'--enable_gpu_relax={args.enable_gpu_relax}',
        f'--gpu_devices={args.gpu_devices}',
        f'--fasta_paths={",".join(fasta_files)}',
        f'--output_dir={args.output_dir}',
        f'--data_dir={args.data_dir}',
        f'--docker_image_name={args.docker_image_name}',
        f'--max_template_date={args.max_template_date}',
        f'--db_preset={args.db_preset}',
        f'--model_preset={args.model_preset}',
        f'--num_multimer_predictions_per_model={args.num_multimer_predictions_per_model}',
        f'--benchmark={args.benchmark}',
        f'--use_precomputed_msas={args.use_precomputed_msas}',
        f'--docker_user={args.docker_user}'
    ]
    subprocess.run(command)


@dataclass
class mAFminDGArgs:
    af_params: AlphaFoldArgs = AlphaFoldArgs()  # Assuming AlphaFoldArgs is a dataclass
    rosetta_params: RosettaMinimizationParameters = RosettaMinimizationParameters()  # Assuming this is also a dataclass
    use_relaxed: bool = False
    wdir: str = './tmp'
    num_replicates: int = 5
    fix_msas: bool = False # always leave precomputed msas on even for first seq run
    residue_length_norm: bool = True

    def __post_init__(self):
        if type(self.af_params) == str:
            # we assume the user passed a file path to yaml
            with open(self.af_params, 'r') as file:
                self.af_params = AlphaFoldArgs(**yaml.safe_load(file))

class mAFminDGEstimator(ThermoStabilityEstimator):
    """Uses the method of AlphaFold ensembles and Rosetta minimization to estimate the folding free energy of a protein.
    
    See the following paper for more details:
    https://doi.org/10.1021/acs.jcim.2c01083
    """
    def __init__(self, args: mAFminDGArgs):
        super().__init__(args)
        self.af_params = args.af_params
        self.af_params.output_dir = self.args.wdir
        self.rosetta_params = args.rosetta_params
        self.pdb_files_history = {}

    def generate_alphafold_ensembles(self, sequences: list[str], ids: list[str]):
        # Update output_dir
        self.af_params.output_dir = self.args.wdir
        
        output_type = 'relaxed' if self.args.use_relaxed else 'unrelaxed'
        files_dict = {}
        replicates_to_run = {}  # New dictionary to keep track of remaining replicates for each ID
        
        # Check existing outputs
        for sequence_id, sequence in zip(ids, sequences):
            path = os.path.join(self.args.wdir, str(sequence_id))
            if os.path.exists(path):
                existing_replicates = sum('ensemble_replicate' in file for file in os.listdir(path))
                remaining_replicates = self.args.num_replicates - existing_replicates
                
                if remaining_replicates <= 0:
                    logger.info(f"Found existing output with enough replicates for {sequence_id}. Skipping.")
                    files_dict[sequence_id] = path
                    continue
                
                replicates_to_run[sequence_id] = remaining_replicates
            else:
                replicates_to_run[sequence_id] = self.args.num_replicates
        
        if not replicates_to_run:
            logger.info("No sequences to run. Skipping AF run.")
            return files_dict

        logger.info(f"Running AF on `ids: number of replicates remaining`. {replicates_to_run}.")
        
        for replicate in range(self.args.num_replicates):
            # Filter sequences and IDs to those that need more replicates
            sequences_to_run = [seq for id_, seq in zip(ids, sequences) if replicates_to_run.get(id_, 0) > 0]
            ids_to_run = [id_ for id_ in ids if replicates_to_run.get(id_, 0) > 0]
            
            if not sequences_to_run:
                break
            
            logger.info(f"Running AF on {ids_to_run}.")

            self.af_params.use_precomputed_msas = bool(replicate) or self.args.fix_msas

            time0 = time.time()
            run_alphafold(sequences_to_run, ids_to_run, self.af_params)
            time1 = time.time()
            
            logger.info(f"Finished replicate {replicate}. Time taken: {(time1 - time0)/60} mins.")
            
            # Rename output files and update replicates_to_run
            for sequence_id in ids_to_run:
                sequence_dir = os.path.join(self.args.wdir, str(sequence_id))
                if sequence_id not in files_dict:
                    files_dict[sequence_id] = sequence_dir

                # if there were replicates here earlier that were already renamed,
                # we want to start at the end of the number of replicates intsead of at 0 again, lest we overwrite existing files
                current_replicate_over_sessions = sum('ensemble_replicate' in file for file in os.listdir(sequence_dir))

                for file in os.listdir(sequence_dir):
                    if file.endswith('.pdb') and output_type in file:
                        old_name = os.path.join(sequence_dir, file)
                        new_name = os.path.join(sequence_dir, f"ensemble_replicate_{current_replicate_over_sessions}.pdb")
                        shutil.move(old_name, new_name)
                        logger.debug(f"Renamed {old_name} to {new_name}.")
                
                # Decrease remaining replicates for this sequence_id
                replicates_to_run[sequence_id] -= 1
        
        # do a final check to make sure we have all of the expected files
        # if not, this function should be rerun
        complete = True
        for sequence_id in ids:
            sequence_dir = os.path.join(self.args.wdir, str(sequence_id))
            pdb_files = [file for file in os.listdir(sequence_dir) if file.startswith("ensemble_replicate")]
            if len(pdb_files) < self.args.num_replicates * self.af_params.num_models:
                logger.warning(f"Found {len(pdb_files)} PDB files for {sequence_id}, expected {self.args.num_replicates * self.af_params.num_models}.")
                complete = False

        return files_dict, complete

    def compute_ensemble_energies(self, ids: list[str]) -> Dict[str, float]:
        """Compute energies for all PDB files (based on parameters) in the specified directory.
        
        Group them by sequence ID, which occurs first in the filename, and aggregate the energies.
        """
        logger.info("Computing ensemble energies...")
        all_energies = {}

        # Iterate through sequence IDs to find corresponding PDB files
        for sequence_id in ids:
            pdb_paths = []
            # Identify directorie corresponding to the sequence ID
            sequence_dir = None
            for file in os.listdir(self.args.wdir):
                if file == str(sequence_id) and os.path.isdir(os.path.join(self.args.wdir, file)):
                    sequence_dir = os.path.abspath(os.path.join(self.args.wdir, file))
            if sequence_dir is None:
                raise Exception(f"Could not find directory for sequence ID {sequence_id}.")
            pdb_files = [file for file in os.listdir(sequence_dir) if file.startswith("ensemble_replicate")]
            if len(pdb_files) < self.af_params.num_models * self.args.num_replicates:
                raise ValueError(f"Found {len(pdb_files)} PDB files for {sequence_id}, expected {self.af_params.num_models * self.args.num_replicates}.")
            logger.info(f"Computing energies for {sequence_id}... found {len(pdb_files)} PDB files in the ensemble.")
            # get full paths to the PDB files
            pdb_paths = [os.path.join(sequence_dir, pdb_file) for pdb_file in pdb_files]
            # Validate that the number of PDB files matches the expected count

            # Call the external function to minimize the structures and compute energies
            energies = minimize_structures(pdb_paths, self.rosetta_params)
            all_energies[sequence_id] = (np.mean(energies), np.std(energies))
            logger.info(f"Computed energies for {sequence_id}: {all_energies[sequence_id]}")
        return all_energies

    def _run(self, sequences: list[str], ids: list[str], gpu_id: int='all') -> Dict[str, float]:
        time0 = time.time()
        self.af_params.gpu_devices = str(gpu_id)
        complete = False
        while not complete:
            files_dict, complete = self.generate_alphafold_ensembles(sequences, ids)
        self.pdb_files_history.update(files_dict)
        energies = self.compute_ensemble_energies(ids)

        # normalize by length
        if self.args.residue_length_norm:
            for id_ in ids:
                og_values = np.array(energies[id_])
                logger.debug(f"Original energies for {id_}: {og_values}")
                seq = sequences[ids.index(id_)]
                seq_len = len(seq)
                logger.debug(f"Sequence length for {id_}: {seq_len}")
                energies[id_] = list(og_values / seq_len)
                logger.debug(f"Normalized energies for {id_}: {energies[id_]}")

        time1 = time.time()
        logger.info(f"Total time taken for estimator call: {(time1 - time0)/60} mins.")
        return energies
