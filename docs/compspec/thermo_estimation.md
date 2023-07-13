# Tools for estimating a thermostabilization quantity


### `mAFminDGEstimator`
Given a set of protein sequences, estimate delta G using the mAF-min method
described in https://doi.org/10.1021/acs.jcim.2c01083

The process is as follows:
1. For each sequence, predict an ensemble of AlphaFold structures
2. For each structure, run rosettta minimization and compute delta G
3. Average over each ensemble

__Methods__:

- `__init__`: input two sequences and args object
- `generate_alphafold_ensembles`: generate an ensemble of structures for a given sequence
- `_rosetta_minimization_single`: run rosetta minimization on a given pdb file
- `_run_rosetta_minimization`: run rosetta for a set of pdb files
- `compute_ensemble_energies`: compute the average and std of energy of all ensembles of structures for all sequences
- `run`: run the entire process

__Parameters__:
- `args`: args object, contains
  - `sequences`: list[str]: amino acid sequences
  - `ids`: list[str]: identifiers for amino acid sequences
  - `num_replicates`: number of structures to generate for each sequence
  - additional rosetta and alphafold params

### `generate_alphafold_ensembles()`
Generate an AlphaFold structurse for a given set of sequences and ids, output pdb files.

### `_rosetta_minimization_single(pdb_path: str, **kwargs)`
Run rosetta minimization on a given pdb file, return energy.

### `_run_rosetta_minimization(pdb_paths: list[str], **kwargs)`
Run rosetta minimization on a set of pdb files, return energies.

### `compute_ensemble_energies(**kwargs)`
Compute the average and std of energy of all ensembles of structures for all sequences.

### `run()`
Run the entire process.
