"""Tool for getting folding free energy of a protein structure using PyRosetta. After a small minimzation.

This is the algorithm specified in https://doi.org/10.1021/acs.jcim.2c01083

Drafted with GPT4 by @evankomp from the paper specifications.
"""
import os
import tempfile
from dataclasses import dataclass
from dask.distributed import Client, LocalCluster
import pyrosetta
import pyrosetta.distributed.io as io
from pyrosetta.distributed.cluster import PyRosettaCluster
import multiprocessing as mp

pyrosetta.init()

@dataclass
class RosettaMinimizationParameters:
    """These default parmaeters are the ones used in the paper."""
    scorefxn_name: str = 'ref2015'
    min_type: str = 'lbfgs_armijo_nonmonotone'
    tolerance: float = 0.001
    use_constraints: bool = False
    n_workers: int = 1

def _minimize_structure(kwargs):
    """Meant to be mapped my multiprocessing"""
    import pyrosetta # Local import
    import pyrosetta.io as io # Local import
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    
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
    min_mover.apply(pose)

    # Calculate the approximated folding free energy
    folding_free_energy = scorefxn(pose)
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
            "use_constraints": params.use_constraints
        }

    tasks = [create_kwargs(file) for file in pdb_files]
    pool = mp.Pool(params.n_workers)
    results = pool.map(_minimize_structure, tasks)

    return results

    
