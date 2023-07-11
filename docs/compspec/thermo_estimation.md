# Tools for estimating the stabilizing effect of mutations


### `mAFminDDGEstimator`
Given a wild type and variant protein sequence, estimate delta delta G using the mAF-min method
described in https://doi.org/10.1021/acs.jcim.2c01083

The process is as follows:
1. For each sequence, predict an ensemble of AlphaFold structures
2. For each structure, run rosettta minimization and compute delta G
3. Average over each ensemble to estimate delta delta G

__Methods__:

- `__init__`: input two sequences and args object
- `_generate_ensemble`: generate an ensemble of structures for a given sequence
- `_compute_ddg_single`: compute ddg for a given structure
- `run`: run the estimation process

__Parameters__:
- `seq_wt`: wild type sequence
- `seq_var`: variant sequence
- `args`: args object, contains
  - `n_ensemble`: number of structures to generate for each sequence
  - additional rosetta and alphafold params

### `generate_alphafold_structure(sequence: str, **kwargs)`
Generate an AlphaFold structure for a given sequence, output pdb file

### `run_rosetta_minimization(pdb_path: str, **kwargs)`
Run rosetta minimization on a given pdb file, output pdb file
