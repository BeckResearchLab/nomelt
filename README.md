# nomelt
Designing high temperature protein sequences via learned language processing

## Installation for using the trained model

1. Install the conda envorinment. Mamba is recommended for speed.
```
mamba env create -f environment.yml --name nomelt
```

2. Install the in-house codebase, which includes a wrapper around the trained model, and components for estimating and optimizing over thermal stability:
```
pip install -e .
```

3. Installation of pyrossetta is required to run mAF-dg predictor of thermal stability. This is not included in the conda environment, as it is not available via conda. See [here](https://www.pyrosetta.org/downloads#h.6vttn15ac69d) for instructions on how to install pyrosetta. This step can be skipped if you only want to create variants of a protein or evaluate a library of variants.

4. An alphafold container and dataset is also required to run mAF-dg predictor of thermal stability. It can be skipped if you only want to create variants of a protein or evaluate a library of variants. The setup for this is a little chaotic due to the format of our HPC cluster, which does not allow for docker containers. Thus the alphafold container had to be build after modification in Singularity. There are then multiple layers of configuration required. Sorry.
   - First, build the container SIF file using the def file in `./alphafold/Singularity.def`. This will take a while. Use the standard singularity command: `singularity build alphafold.sif Singularity.def`
   - Download alphafolds databases if not already done. This is an extremely large dataset. See their repo: https://github.com/google-deepmind/alphafold
   - Install the additional requirements in `./alphafold/requirements.txt` with pip: `pip install -r ./alphafold/requirements.txt`
   - Modify the `./alphafold/run_singularity.py` (Line 37) to point towards the SIF file created in the first step.
   - Modify the AF config file found at `.config/af_singularity_config.yaml` to point towards the alphafold database and the `run_singularity` python script from the previous step that runs the container, lines 2 and 5 respectively.
   - Finally, modify the NOMELT app config file at `./app/config.yaml` to point towards the AF config file, under the key optimization: estimator_args: af_params. See the example below:
```
# Step 4: In Silico Optimization
optimization:
  enabled: false
  estimator: mAFminDGEstimator
  estimator_args: 
    af_params: ./.config/af_singularity_config.yaml
    use_relaxed: False
...
```

## Installation for repeating the training and evaluation pipeline
Please first follow the installation instructions above. Then, follow the instructions below. We need some additonal non-conda packages for training and evaluation.

### FATCAT
Needed for comparing tertiary structure. See installation instructions: https://github.com/GodzikLab/FATCAT-dist

## Environmental variables
If running the pipeline, please set `TMP` which specifies the location of temporary files will be created. Also set `LOG_LEVEL` to e.g. `INFO` or `DEBUG` to control the verbosity of the logs for the pipeline.

## Config

There are a number of sporadic config files floating around for different parts of the software, living in the `./config` directory. 

- First is `af_singularity_config.yaml` which is used to configure the alphafold container such that the mAF DG method or single structure predictions can be used. The path variables here will need to be changed to match your AF executables after installation of AF. The other variables configure the AF2 calls and the values present are the ones used for this work.
- `./config/accelerate/default_config.yaml` contains the config for accelerate/DeepSpeed is used for training the transformer with ZeRO. 
- `./config/accelerate/data_parallel_config.yaml` contains the config for accelerate/DeepSpeed when running predictions, since model parallel BEAM search has a hard time.
- Both of the above need to match the number of GPUs being trained on. If this diverges, then the effective batch size will be wrong and the training may diverge from the behavior reported in the paper.

## Usage, rerunning the pipeline
1. The learn2therm dataset will need to be acquired. See [here](https://figshare.com/articles/dataset/learn2therm_relational_database/23581932). After downloading, place the duckdb file `learn2therm.ddb` in the `./data` directory as `./data/database.ddb` Then execute `dvc add ./data/database.ddb` This will be tracked by DVC.

2. Hypothetically, the entire pipeline can then be run with one command, assuming enough available resources by `dvc exp run` however it is recommended that individual pipeline steps be run in order with only the necessary resources. For example, data processing steps do not need access to GPUs. Runa single step by `dvc exp run -s STAGE_NAME --downstream`. You can see the names of stages by `dvc status`

## Usage, using the trained model

## TODO

- HF caching: caching and DVC clash a little bit. Be default, when you do operations on a HF dataset, it creates cache files in the dataset folder, which makes DVC think the dataset has changed. If you want to use those cache operations, you have to commit the data/dataset object to DVC with changes. Instead, it would be better if cacheing dataset operations were abstracted out into their own script, and the cache file manually pathed to a dvc tracked output. Thus if paramters in the pipeline would change the operation, that one stage would be run, but downstream stages that use the same operations (eg. tokenization) could reuse that dvc tracked cache.


