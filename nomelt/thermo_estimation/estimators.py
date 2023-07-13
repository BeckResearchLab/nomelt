import os
import shutil
import tempfile
import logging
from typing import Dict, List
from dataclasses import dataclass, field
import numpy as np

from nomelt.thermo_estimation.alphafold import run_alphafold, AlphaFoldParams
from nomelt.thermo_estimation.rosetta import rosetta_minimization, RosettaMinimizationParameters

import logging
logger = logging.getLogger(__name__)

@dataclass
class mAFminDGArgs:
    af_models: list[int] = field(default_factory=lambda: [3, 4])
    af_relaced: bool = Treue
    af_params = AlphaFoldParams()
    rosetta_params = RosettaMinimizationParameters()

class mAFminDGEstimator:
    def __init__(self, sequences: list[str], ids: list[str], args: mAFminDGArgs):
        self.ids = ids
        self.sequences = sequences
        self.af_params = args.af_params
        self.rosetta_params = args.rosetta_params

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def generate_alphafold_ensembles(self):
        self.logger.info("Generating AlphaFold ensembles...")
        run_alphafold(self.sequences, self.ids, self.af_params)
        self.logger.info("Generation of AlphaFold ensembles completed.")

    def _rosetta_minimization_single(self, pdb_path: str, **kwargs) -> float:
        self.logger.info(f"Running Rosetta minimization for {pdb_path}...")
        energy = rosetta_minimization(pdb_path, **kwargs)
        self.logger.info(f"Completed Rosetta minimization for {pdb_path}.")
        return energy

    def _run_rosetta_minimization(self, pdb_paths: List[str], **kwargs) -> Dict[str, float]:
        energies = []
        for pdb_path in pdb_paths:
            energies.append(self._rosetta_minimization_single(pdb_path, **kwargs))
        return energies

    def compute_ensemble_energies(directory: str, self, **kwargs) -> Dict[str, float]:
        """Compute energies for all PDB files (based on parameters) in the specified directory.
        
        Group them by sequence ID, which occurs first in the filename, and aggregate the energies.
        """
        self.logger.info("Computing ensemble energies...")
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
            energies = self._run_rosetta_minimization(pdb_paths, **kwargs)
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
