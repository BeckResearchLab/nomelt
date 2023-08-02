import os
import shutil
import tempfile
import logging
from typing import Dict, List
from dataclasses import dataclass, field
import numpy as np
import torch
from tqdm import tqdm

from nomelt.thermo_estimation.alphafold import run_alphafold, AlphaFoldParams
from nomelt.thermo_estimation.rosetta import minimize_structures, RosettaMinimizationParameters
import esm

import logging
logger = logging.getLogger(__name__)

ESMFOLD = esm.pretrained.esmfold_v1().eval()

class ThermoStabilityEstimator:
    """Abstract parent class."""
    def __init__(self, sequences: list[str], ids: list[str], args=None):
        assert len(sequences) == len(ids)
        self.ids = ids
        self.sequences = sequences
        self.args=args

    def run(self) -> Dict[str, float]:
        """Run the estimator on the specified sequences.
        
        Returns:
            Dict[str, float]: A dictionary map of ids to estimated thermal stability"""
        raise NotImplementedError()

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

@dataclass
class ESMFoldDGArgs:
    rosetta_params: RosettaMinimizationParameters = RosettaMinimizationParameters()
    gpu_i: int = None
    wdir: str = os.path.abspath('./tmp/ESMFoldDGEstimator_structures')

class ESMFoldDGEstimator(ThermoStabilityEstimator):
    """Uses ESMfold to predict the structure of a protein and estimate its thermal stability."""
    
    def __init__(self, sequences: list[str], ids: list[str], args: ESMFoldDGArgs=ESMFoldDGArgs()):
        super().__init__(sequences, ids, args=args)

    def generate_esmfold_structures(self, temp_dir: str) -> Dict[str, str]:
        """Predict protein structures using ESMfold and save them to PDB files in the temporary directory."""

        if self.args.gpu_i is None:
            device='cuda'
        else:
            device=f'cuda:{self.args.gpu_i}'
        
        model = ESMFOLD
        model.to(device)
        
        pdb_outputs = {}
        self.pldtt = []
        logger.info(f'Running ESMfold on {len(self.sequences)} proteins.')
        
        with torch.no_grad():
            for pos in tqdm(range(0, len(self.sequences), 4)):
                batch_sequences = self.sequences[pos:pos + 4]
                batch_ids = self.ids[pos:pos + 4]
                
                outputs = model.infer(batch_sequences)
                self.pldtt.append(outputs["mean_plddt"].cpu().numpy())
                
                # Convert outputs to pdb and save them immediately for the current batch
                batch_pdb_list = model.output_to_pdb(outputs)
                for seq_id, pdb_data in zip(batch_ids, batch_pdb_list):
                    pdb_filename = os.path.join(temp_dir, f"{seq_id}.pdb")
                    with open(pdb_filename, "w") as f:
                        f.write(pdb_data)
                    pdb_outputs[seq_id] = pdb_filename
    
        del outputs
        torch.cuda.empty_cache()
        
        return pdb_outputs

    def compute_structures_energies(self, pdb_files: List[str]) -> List[float]:
        """Compute energies for a list of PDB files using Rosetta."""
        return minimize_structures(pdb_files, self.args.rosetta_params)

    def run(self) -> Dict[str, float]:
        temp_dir = self.args.wdir
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        pdb_files_dict = self.generate_esmfold_structures(temp_dir)
        self.pdb_files_dict = pdb_files_dict
    
        energies = self.compute_structures_energies(list(pdb_files_dict.values()))
    
        energy_outputs = dict(zip(pdb_files_dict.keys(), energies))

        return energy_outputs
