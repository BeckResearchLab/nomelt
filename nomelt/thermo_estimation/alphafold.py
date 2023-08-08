"""Runs AF 

NEEDS UPDATING"""

import os
import subprocess
from dataclasses import dataclass, field
import tempfile
import shutil

from nomelt.thermo_estimation.rosetta import minimize_structures, RosettaMinimizationParameters
from nomelt.thermo_estimation.estimator import ThermoStabilityEstimator
from typing import Dict, List

import logging
logger = logging.getLogger(__name__)

@dataclass
class AlphaFoldParams:
    max_template_date: str = "2021-11-01"
    model_preset: str = "monomer"
    db_preset: str = "reduced_dbs"
    data_dir: str = "$AF_DOWNLOAD_DIR"
    output_dir: str = './af_output/'
    num_replicates: int = 5

def write_fasta(sequences, sequence_names, num_replicates, fasta_file):
    for sequence_name, sequence in zip(sequence_names, sequences):
        for i in range(num_replicates):
            fasta_file.write(f'>{sequence_name}_{i}\n')
            fasta_file.write(f'{sequence}\n')
    fasta_file.flush()

def run_alphafold(sequences: list[str], sequence_names: list[str], params: AlphaFoldParams):
    """
    Run the AlphaFold prediction process using specified parameters.

    This function will generate a temporary FASTA file containing the specified sequences, and then execute the AlphaFold docker script using the given parameters. The temporary FASTA file is deleted once the script execution completes.

    Args:
        params (AlphaFoldParams): An instance of AlphaFoldParams dataclass which includes:
            sequences (list): A list of amino acid sequences for which the structure prediction should be done.
            sequence_name (str): A unique identifier for the sequence.
            max_template_date (str): The maximum date for templates used in the prediction.
            model_preset (str): The type of model preset to use for the prediction.
            db_preset (str): The type of database preset to use for the prediction.
            data_dir (str): The path to the directory containing necessary AlphaFold data.
            output_dir (str): The path to the directory where the output files should be saved.
            num_replicates (int): The number of replicate predictions to run for each sequence.

    Returns:
        None. The function will directly produce output files in the specified output directory.

    Raises:
        subprocess.CalledProcessError: If the AlphaFold docker script fails to execute.


    Notes:
    Output files of the following format:
        <target_name>/
            features.pkl
            ranked_{0,1,2,3,4}.pdb
            ranking_debug.json
            relax_metrics.json
            relaxed_model_{1,2,3,4,5}.pdb
            result_model_{1,2,3,4,5}.pkl
            timings.json
            unrelaxed_model_{1,2,3,4,5}.pdb
            msas/
                bfd_uniref_hits.a3m
                mgnify_hits.sto
                uniref90_hits.sto
    """
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=True) as fasta_file:
        write_fasta(sequences, sequence_names, params.num_replicates, fasta_file)
        cmd = [
            'python3', 'docker/run_docker.py',
            '--fasta_paths=' + fasta_file.name,
            '--max_template_date=' + params.max_template_date,
            '--model_preset=' + params.model_preset,
            '--db_preset=' + params.db_preset,
            '--data_dir=' + params.data_dir,
            '--output_dir=' + params.output_dir,
        ]
        subprocess.run(cmd, check=True)


@dataclass
class mAFminDGArgs:
    af_models: list[int] = field(default_factory=lambda: [3, 4])
    af_relaced: bool = True
    af_params = AlphaFoldParams()
    rosetta_params = RosettaMinimizationParameters()

class mAFminDGEstimator(ThermoStabilityEstimator):
    """Uses method of AlphaFold ensembles and Rosetta minimization to estimate the folding free energy of a protein.
    
    See the following paper for more details:
    https://doi.org/10.1021/acs.jcim.2c01083
    """
    def __init__(self, sequences: list[str], ids: list[str], args: mAFminDGArgs):
        super().__init__(sequences, ids)
        self.af_params = args.af_params
        self.rosetta_params = args.rosetta_params

    def generate_alphafold_ensembles(self):
        logger.info("Generating AlphaFold ensembles...")
        run_alphafold(self.sequences, self.ids, self.af_params)
        logger.info("Generation of AlphaFold ensembles completed.")

    def compute_ensemble_energies(self, directory: str, ) -> Dict[str, float]:
        """Compute energies for all PDB files (based on parameters) in the specified directory.
        
        Group them by sequence ID, which occurs first in the filename, and aggregate the energies.
        """
        logger.info("Computing ensemble energies...")
        all_energies = {}
        output_type = 'relaxed' if self.af_params.af_relaxed else 'unrelaxed'
        output_models = self.af_params.af_models
        for sequence_id in self.ids:
            pdb_paths = []
            # these are each replicate
            sequence_replicates = [dir_name for dir_name in os.listdir(directory + '/') if dir_name.startswith(sequence_id)]
            if len(sequence_replicates) != self.af_params.num_replicates:
                raise ValueError(f"Expected {self.af_params.num_replicates} output files for sequence {sequence_id}, but found {len(sequence_replicates)}")
            # now get the specific pdb files for each replicate
            for replicate in sequence_replicates:
                pdb_paths += [os.path.join(directory, replicate, f'{output_type}_model_{model}.pdb') for model in output_models]
            # check the number of pdb files matcches expected count
            if len(pdb_paths) != len(output_models) * self.af_params.num_replicates:
                raise ValueError(f"Expected {len(output_models) * self.af_params.num_replicates} PDB files for sequence {sequence_id}, using {output_models} models, and {self.af_params.num_replicates} replicates, but found {len(pdb_paths)}")
            energies = minimize_structures(pdb_paths, self.rosetta_params)
            all_energies[sequence_id] = (np.mean(energies), np.std(energies))
            logger.info(f"Computed energies for {sequence_id}: {all_energies[sequence_id]}")
        return all_energies

    def run(self) -> Dict[str, float]:
        temp_dir = tempfile.mkdtemp()
        self.generate_alphafold_ensembles()
        # move the output files to a temporary directory
        logger.info("Moving AlphaFold output files to temporary directory...")
        for sequence_id in self.ids:
            sequence_outputs = [dir_name for dir_name in os.listdir(self.af_params.output_dir + '/') if dir_name.startswith(sequence_id)]
            if len(sequence_outputs) != self.af_params.num_replicates:
                raise ValueError(f"Expected {self.af_params.num_replicates} output files for sequence {sequence_id}, but found {len(sequence_outputs)}")
            for output_dir in sequence_outputs:
                shutil.move(os.path.join(self.af_params.output_dir, output_dir), temp_dir)
        energies = self.compute_ensemble_energies(temp_dir)
        shutil.rmtree(temp_dir)
        return energies
