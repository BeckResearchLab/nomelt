"""Tool for getting folding free energy of a protein structure using PyRosetta. After a small minimzation.

This is the algorithm specified in https://doi.org/10.1021/acs.jcim.2c01083

Drafted with GPT4 by @evankomp from the paper specifications.
"""
import os
import tempfile
from dataclasses import dataclass
import pyrosetta.distributed.io as io
import multiprocessing as mp

import logging
logger = logging.getLogger(__name__)

@dataclass
class RosettaMinimizationParameters:
    """These default parmaeters are the ones used in the paper."""
    scorefxn_name: str = 'ref2015'
    min_type: str = 'lbfgs_armijo_nonmonotone'
    tolerance: float = 0.001
    use_constraints: bool = False
    n_workers: int = 1
    update_pdb: bool = False
    new_pdb_dir: str = None

def _minimize_structure(kwargs):
    """Meant to be mapped my multiprocessing"""
    import pyrosetta # Local import
    import pyrosetta.io as io # Local import
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    if not pyrosetta.rosetta.basic.was_init_called():
        pyrosetta.init()
    
    # Create a pose from the PDB file
    pose=io.pose_from_pdb(kwargs['pdb_file'])
    # Create a score function
    scorefxn = pyrosetta.create_score_function(kwargs['scorefxn_name'])
    # Add constraints to the score function, if requested
    if kwargs['use_constraints']:
        scorefxn.set_weight(pyrosetta.core.scoring.constraints, 1.0)
    # Setup the MinMover
    min_mover = MinMover()
    mm = pyrosetta.MoveMap()
    mm.set_bb(True)
    mm.set_chi(True)
    mm.set_jump(True)
    min_mover.movemap(mm)
    min_mover.score_function(scorefxn)
    min_mover.min_type(kwargs['min_type'])
    min_mover.tolerance(kwargs['tolerance'])

    # Apply the MinMover to the pose
    folding_free_energy_change = 1.0
    folding_free_energy = None
    tries = 0
    while folding_free_energy_change > 0.05:
        min_mover.apply(pose)
        tmp_folding_free_energy = scorefxn(pose)
        if folding_free_energy is None:
            pass
        else:
            folding_free_energy_change = abs((folding_free_energy - tmp_folding_free_energy)/folding_free_energy)
        folding_free_energy = tmp_folding_free_energy
        tries += 1
        if tries > 5:
            break

    if kwargs['update_pdb']:
        logger.debug(f"Updated PDB file {kwargs['pdb_file']} with minimized positions")
        pose.dump_pdb(kwargs['pdb_file'])
    elif kwargs['new_pdb_dir'] is not None:
        if not os.path.exists(kwargs['new_pdb_dir']):
            os.makedirs(kwargs['new_pdb_dir'])
        pdb_name = os.path.basename(kwargs['pdb_file'])
        new_pdb_path = os.path.join(kwargs['new_pdb_dir'], pdb_name)
        logger.debug(f"Writing new PDB file {new_pdb_path} with minimized positions")
        pose.dump_pdb(new_pdb_path)
    
    return folding_free_energy

def minimize_structures(pdb_files: list[str], params: RosettaMinimizationParameters=RosettaMinimizationParameters()):
    """
    This function minimizes a protein structure and calculates its folding free energy.

    Parameters:
    pdb_files list(str): Path to the input PDB files.
    params (MinimizationParameters): A dataclass object that contains the following fields:
        scorefxn_name (str): Name of the score function to use. Default is 'ref2015'.
        min_type (str): Type of minimization algorithm to use. Default is 'dfpmin_armijo_nonmonotone'.
        tolerance (float): Tolerance for the minimization. Default is 0.01.
        use_constraints (bool): Whether to use constraints in the score function. Default is False.
        init (bool): Whether to initialize PyRosetta. Default is True.
            Set to False if running many minimizations in parallel.

    Returns:
    list of energies (float): The folding free energy of each structure.

    This function uses the PyRosetta package to load the input protein structure, 
    minimize it using the specified algorithm and score function, and calculate 
    its folding free energy. If use_constraints is set to True, a constraints term 
    is added to the score function.
    """

    def create_kwargs(file):
        return {
            "pdb_file": file,
            "scorefxn_name": params.scorefxn_name,
            "min_type": params.min_type,
            "tolerance": params.tolerance,
            "use_constraints": params.use_constraints,
            "update_pdb": params.update_pdb,
            "new_pdb_dir": params.new_pdb_dir
        }

    tasks = [create_kwargs(file) for file in pdb_files]
    if params.n_workers == 1:
        results = [_minimize_structure(task) for task in tasks]
        return results
    else:
        pool = mp.Pool(params.n_workers)
        results = pool.map(_minimize_structure, tasks)
    return results

    
