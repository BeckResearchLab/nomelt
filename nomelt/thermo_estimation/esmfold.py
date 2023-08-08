import os
import shutil
import tempfile
import copy
import logging
from typing import Dict, List
from dataclasses import dataclass, field
import numpy as np
import torch
from tqdm import tqdm

from nomelt.thermo_estimation.rosetta import minimize_structures, RosettaMinimizationParameters
from nomelt.thermo_estimation.estimator import ThermoStabilityEstimator
import esm

import logging
logger = logging.getLogger(__name__)

ESMFOLD = esm.pretrained.esmfold_v1().eval()

@dataclass
class ESMFoldDGArgs:
    rosetta_params: RosettaMinimizationParameters = RosettaMinimizationParameters()
    wdir: str = os.path.abspath('./tmp/ESMFoldDGEstimator_structures')

class ESMFoldDGEstimator(ThermoStabilityEstimator):
    """Uses ESMfold to predict the structure of a protein and estimate its thermal stability."""

    def __init__(self, args: ESMFoldDGArgs = ESMFoldDGArgs()):
        super().__init__(args=args)
        self.pdb_files_history = {}  # Add this line to initialize the history list
        self._gpus_used = {}

    def _run(self, sequences: list[str], ids: list[str], gpu_id: int=None) -> Dict[str, float]:
        temp_dir = self.args.wdir
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        pdb_files_dict = self.generate_esmfold_structures(temp_dir, sequences, ids, gpu_id=gpu_id)
        self.pdb_files_history.update(pdb_files_dict)  # Add this line to record the history

        energies = self.compute_structures_energies(list(pdb_files_dict.values()))

        energy_outputs = dict(zip(pdb_files_dict.keys(), energies))

        return energy_outputs

    def generate_esmfold_structures(self, temp_dir: str, sequences: list[str], ids: list[str], gpu_id: int=None) -> Dict[str, str]:
        """Predict protein structures using ESMfold and save them to PDB files in the temporary directory."""

        if gpu_id is None:
            device='cuda'
        else:
            device=f'cuda:{gpu_id}'

        if gpu_id in self._gpus_used:
            local_model = self._gpus_used[gpu_id]
        else:
            local_model = copy.deepcopy(ESMFOLD)
            local_model.to(device)
            self._gpus_used[gpu_id] = local_model
        
        pdb_outputs = {}
        self.pldtt = []
        logger.info(f'Running ESMfold on {len(sequences)} proteins.')
        
        with torch.no_grad():
            for pos in tqdm(range(0, len(sequences), 4)):
                batch_sequences = sequences[pos:pos + 4]
                batch_ids = ids[pos:pos + 4]
                
                outputs = local_model.infer(batch_sequences)
                self.pldtt.append(outputs["mean_plddt"].cpu().numpy())
                
                # Convert outputs to pdb and save them immediately for the current batch
                batch_pdb_list = local_model.output_to_pdb(outputs)
                for seq_id, pdb_data in zip(batch_ids, batch_pdb_list):
                    pdb_filename = os.path.join(temp_dir, f"{seq_id}.pdb")
                    with open(pdb_filename, "w") as f:
                        f.write(pdb_data)
                    pdb_outputs[seq_id] = pdb_filename
        
        return pdb_outputs

    def compute_structures_energies(self, pdb_files: List[str]) -> List[float]:
        """Compute energies for a list of PDB files using Rosetta."""
        return minimize_structures(pdb_files, self.args.rosetta_params)
