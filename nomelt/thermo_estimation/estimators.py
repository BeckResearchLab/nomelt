import os
import shutil
import tempfile
import logging
from typing import Dict, List
from dataclasses import dataclass, field
import numpy as np

from nomelt.thermo_estimation.alphafold import run_alphafold, AlphaFoldParams
from nomelt.thermo_estimation.rosetta import minimize_structures, RosettaMinimizationParameters

import logging
logger = logging.getLogger(__name__)

class ThermoStabilityEstimator:
    """Abstract parent class."""
    def __init__(self, sequences: list[str], ids: list[str], args=None):
        assert len(sequences) == len(ids)
        self.ids = ids
        self.sequences = sequences

    def run(self) -> Dict[str, float]:
        """Run the estimator on the specified sequences.
        
        Returns:
            Dict[str, float]: A dictionary map of ids to estimated thermal stability"""
        raise NotImplementedError()

@dataclass
class mAFminDGArgs:
    af_models: list[int] = field(default_factory=lambda: [3, 4])
    af_relaced: bool = Treue
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
