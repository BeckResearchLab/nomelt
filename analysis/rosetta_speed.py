import os
import nomelt.thermo_estimation.rosetta
from pyrosetta import init, pose_from_pdb, create_score_function
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
import dask.distributed
import multiprocessing as mp
import time

wdir = '../learn2thermDB/tmp/structural_alignment/'
pdbs = os.listdir(wdir)
pdbs = [wdir + pdb for pdb in pdbs]

N_cpus = 1
N_sequences = 300

params = nomelt.thermo_estimation.rosetta.RosettaMinimizationParameters(n_workers=N_cpus)

if __name__ == '__main__':
    t0 = time.time()
    results = nomelt.thermo_estimation.rosetta.minimize_structures(pdbs[:N_sequences], params=params)
    t1 = time.time()
    print(results)
    print(f'N_cpus={N_cpus}, N_sequences={N_sequences}, time={(t1-t0)/60}')