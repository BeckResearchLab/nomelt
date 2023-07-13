# Tools for estimating a thermostabilization quantity

## `mAFminDGEstimator`
Given a set of protein sequences, estimate delta G using the mAF-min method described in [source](https://doi.org/10.1021/acs.jcim.2c01083)

The process is as follows:
1. For each sequence, predict an ensemble of AlphaFold structures
2. For each structure, run Rosetta minimization and compute delta G
3. Average over each ensemble

### Parameters:

- `sequences`: list[str]: Amino acid sequences
- `ids`: list[str]: Identifiers for amino acid sequences
- `args`: mAFminDGArgs object, contains:
    - `af_models`: list[int]: AlphaFold model versions to be used
    - `af_relaxed`: bool: Whether the structures are relaxed or not
    - `af_params`: AlphaFoldParams object: Contains AlphaFold-related parameters
    - `rosetta_params`: RosettaMinimizationParameters object: Contains Rosetta-related parameters

### Methods:

#### `__init__(sequences, ids, args)`

Input two sequences and mAFminDGArgs object

#### `generate_alphafold_ensembles()`

Generate an ensemble of structures for each given sequence using AlphaFold and store the structures in PDB files.

#### `_rosetta_minimization_single(pdb_path: str, **kwargs) -> float`

Run Rosetta minimization on a given pdb file and return energy.

#### `_run_rosetta_minimization(pdb_paths: list[str], **kwargs) -> list[float]`

Run Rosetta minimization on a set of pdb files and return a list of energies.

#### `compute_ensemble_energies(directory: str, **kwargs) -> Dict[str, float]`

Compute the average and standard deviation of energy of all ensembles of structures for all sequences. The method takes into consideration the AlphaFold model used and whether the structures are relaxed or not. The method then returns a dictionary where the keys are sequence IDs and values are tuples of mean and standard deviation of energies.

#### `run() -> Dict[str, float]`

Run the entire process. This method manages the lifecycle of the temporary directory used for holding PDB files and returns a dictionary where keys are sequence IDs and values are tuples of mean and standard deviation of energies.


# `MutationSubsetOptimizer`
Given a WT and variant sequence, find the optimal subset of mutations that maximizes the thermostability of the variant.

## Helper functions:
### `clean_gaps(sequence: str)`
Remove gaps from a sequence such that it can be given to downstream estimators.

### `hash_mutation_set(list[str])`
Hash a mutation set to a string for use in Estimator processing.

## Parameters:
- `wt`: str: Wild-type sequence
- `variant`: str: Variant sequence
- `estimator`: class of ThermoStabilityEstimator
- `estimator_args`: args for estimator
- `params`: OptimizerParams: parameters for the optimization
  - `direction`: str: 'maximize' or 'minimize'
  - `sampler`: Instance of optuna.samplers.BaseSampler
  - `n_trials`: int: Number of trials to run
  - `pruner`: Instance of optuna.pruners.BasePruner

## Methods:
### `__init__(wt, variant, estimator, params, estimator_args=None)`
Input wild-type sequence, variant sequence, estimator class, OptunaParams object and optional estimator args

### `_set_mutation_set()`
Set the mutation set to be the set of all possible mutations. The two sequences are aligned and the mutations are determined by the alignment.
See biopython pairwise2.align.globalxx for more details.
The alignment output will be of the form:
    MAKDBS--ASFASF--SFF
    ||  ||  ||||||  | |
    MA--BSSFASFASFBGSDF
Anywhere there is not a matching character, a mutation is assumed to have occurred, eg in the above example there are 7 mutation: 2 deletions, 4 insertions, and 1 substitution.
We then assign an attribute to the class called `mutation_set` which is a dict of positions and tuples of (wt, variant) amino acids or gaps. We also now have
attributes `aligned_wt` is the aligned wt sequence string including gaps. Needed so that we can assign mutations to the correct position in the variant sequence.

### `_get_variant_sequence(mutation_set: list[str])`
Given a mutation set, return the variant sequence.

### `_call_estimator(mutation_set: list[str])`

### `get_objective_function()`
Return an objective function for the optuna optimizer, with each possible mutation as as boolean parameter.
The objective function hashes the mutation set and checks if it has already been computed. If it has, it returns the cached value. 
If not, it instantiallizes the estimator and gets the delta G of the variant sequence. It then caches the result and returns it.
Optuna will then use this value to optimize the objective function. by modifying mutations.

### `run()`
Run the optimizer and return the optimal mutation set and final sequence.
