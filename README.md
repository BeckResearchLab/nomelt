# nomelt
Introducing variation onto protein sequences targeting high temperature stability via neural machine translation.

This work is associated with IN REVIEW. The preprint is available at XXX.

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
   - Clone and navigate to: https://github.com/EvanKomp/alphafold. This contains an old version of the AF code that we know works and some additional scripts to build a singularity container.
   - First, build the container SIF file using the def file in that repo `Singularity.def`. This will take a while. Use the standard singularity command: `singularity build alphafold.sif Singularity.def`
   - Download alphafolds databases if not already done. This is an extremely large dataset. See their repo: https://github.com/google-deepmind/alphafold
   - Modify the `./run_singularity.py` (Line 37) to point towards the SIF file created in the first step.
   - Navigate back to the NOMELT repo. Install the additional requirements in `./alphafold_reqs.txt` with pip: `pip install -r ./alphafold_reqs.txt`
   - Modify the AF config file found at `.config/af_singularity_config.yaml` to point towards the alphafold database and the `run_singularity` python script from two steps above that runs the container, lines 2 and 5 respectively.
   - Finally, modify the NOMELT app config file at `./app/config.yaml` to point towards the AF config file, under the key optimization: estimator_args: af_params. See the example below:
```
# Step 4: In Silico Optimization
optimization:
  enabled: false
  estimator: mAFminDGEstimator
  estimator_args: 
    af_params: ./.config/af_singularity_config.yaml # location of the alphafold config file
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
A wrapper was created around the trained model to make it easy to use, including BEAM search, stochastic sampling e.g. producing many variants, optimization over suggested mutations, and zero-shot prediction. These are chosen by enabling different steps in the config file, see below for the different steps that you can run.

Acquire the trained model parameters from Zenodo: https://doi.org/10.5281/zenodo.10607558

After installation above, `./app/run_nomelt.py` can be used to interact with the trained model. What
will be conducted is determined by the config file at `./app/config.yaml`. Each section after the first in this `yaml` file can be enabled and configured.

The first section, `model`, defines hyperparameters for loading the model. You probably shouldn't change these.

Calling the script has the following signature:
```
python run_nomelt.py [-h] input output_dir model_path config_file
```
- `input` is either a sequence or a library of sequences. If a sequence, it should be a string. If a library, it should be a text file with one sequence per line.
- `output_dir` is the path to the output directory. If the directory does not exist, it will be created. Results are dumped here
- `model_path` is the path to the NOMELT model directory. This should be the directory containing the `pytorch_model.bin` file you got from Zenodo.
- `config_file` is the path to the config.yaml file. This is the config file that controls the behavior of the script. See below for details.

The following subsections describe the steps that can be enabled.

### To produce a single translation of an input sequence
This produces the most likely translation of the input sequence, on average, according to the model. __Enable Step 1__ and configure the number of beams and max length of the sequence. Input the input sequence as a string to the script. It produces an output file "beam_search_sequence.txt" with the translation.

### To produce a large set of variants
This can be achieved in two ways:

Input the input sequence as a string to the script.

1. __In addition to enabling Step 1, enable Step 3__. This will conduct an alignment between the translation and the input, discretize a number of mutations upon that alignment resulting from the differences, and create a library of permuations over those suggested mutations. It outputs a file "library.txt". Note, this writes all combinations of mutations, which can be VERY large, for example with 20 mutations this is 2^20 sequences. The output file can be many gigabytes. Use the next option if the NOMELT model suggests a large number of mutations.

2. __Enable Step 2__. This creates a number of variants stochastically. The temperature, max difference in length between stochastic variants and the input, and the number of variants to create can be configured. One of NOMELT's failure modes is to reproduce the input sequence on BEAM searches. By setting a high temperature in this strategy, the model is more likely to produce variants that are different from the input, though no guarantee that the model makes a good set of suggestions. This outputs a file "stochastic_sequences.txt"

### To evaluate a library of variants zero-shot
Instead of inputting a sequence, input a library of sequences. The first sequence must be the wild type sequence. The library should be a text file with one sequence per line. __Enable Step 5__. This will evaluate the library of sequences and output a file "zero_shot_scores.txt" where each line is the predicted score associated with the input sequence on the same line.

### To conduct "optimization" over suggested mutations using an in silico estimator
This can be extremely expensive and requires multiple GPUs. As of Jan 2024, only the mAF-dg method has been used as a scorered and is suggested.

__Enable Step 4__. Configure the estimator to use, the number of trials in exploring the library, the type of sampler for choosing mutations to testm etc. This outputs a file "optimize_results.json" which contains the sequence, score, and predicted structure file of the best sequence found. It also outputs "trials.csv" which is a dataframe of all of the trials executed. 

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments
This work was funded under NSF Engineering Data Science Institute Grant OAC-1934292.

