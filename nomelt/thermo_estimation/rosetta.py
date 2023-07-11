"""Tool for getting folding free energy of a protein structure using PyRosetta. After a small minimzation.

This is the algorithm specified in https://doi.org/10.1021/acs.jcim.2c01083

Drafted with GPT4 by @evankomp from the paper specifications.
"""

from dataclasses import dataclass
from pyrosetta import init, pose_from_pdb, create_score_function
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

@dataclass
class MinimizationParameters:
    """These default parmaeters are the ones used in the paper."""
    pdb_file: str
    scorefxn_name: str = 'ref2015'
    min_type: str = 'lbfgs_armijo_nonmonotone'
    tolerance: float = 0.001
    use_constraints: bool = False

def minimize_structure(params: MinimizationParameters):
    """
    This function minimizes a protein structure and calculates its folding free energy.

    Parameters:
    params (MinimizationParameters): A dataclass object that contains the following fields:
        pdb_file (str): Path to the input PDB file.
        scorefxn_name (str): Name of the score function to use. Default is 'ref2015'.
        min_type (str): Type of minimization algorithm to use. Default is 'dfpmin_armijo_nonmonotone'.
        tolerance (float): Tolerance for the minimization. Default is 0.01.
        use_constraints (bool): Whether to use constraints in the score function. Default is False.

    Returns:
    tuple: A tuple containing the following elements:
        folding_free_energy (float): The folding free energy of the minimized protein structure, 
            estimated using the specified score function.
        output_pdb (str): Path to the output PDB file containing the minimized structure.

    This function uses the PyRosetta package to load the input protein structure, 
    minimize it using the specified algorithm and score function, and calculate 
    its folding free energy. If use_constraints is set to True, a constraints term 
    is added to the score function.
    """
    # Initialize PyRosetta
    init()
    # Create a pose from the PDB file
    pose = pose_from_pdb(params.pdb_file)
    # Create a score function
    scorefxn = create_score_function(params.scorefxn_name)
    # Add constraints to the score function, if requested
    if params.use_constraints:
        scorefxn.set_weight(rosetta.core.scoring.constraints, 1.0)
    # Setup the MinMover
    min_mover = MinMover()
    min_mover.score_function(scorefxn)
    min_mover.min_type(params.min_type)
    min_mover.tolerance(params.tolerance)

    # Apply the MinMover to the pose
    min_mover.apply(pose)

    # Calculate the approximated folding free energy
    folding_free_energy = scorefxn(pose)
    return folding_free_energy
