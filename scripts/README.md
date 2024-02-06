# Scripts
This folder contains scripts used to define the method's pipeline, proof of principle.
Scripts should be executed by running DVC stages from the root of the NOMELT repo, do not run scripts from here

These are the scripts used to prepare data, train the model, evaluate the model, use it to engineer ENH1, and run it zero-shot on experimental data.

## Script Name: prepare_data.py

### Description
- Prepares data for training by filtering and balancing protein pairs, considering factors like temperature differences between meso and thermo homologs, alignment coverage, and more. Generates a Huggingface dataset containing protein pairs.

### Input Files
- `./scripts/prepare_data.py`: The main script file.
- `./data/database.ddb`: Database file used as the data source.

### Output Files
- `./data/dataset/`: A Huggingface dataset directory containing processed protein pairs.

### Parameters
- `data.min_temp_diff`: Current value - 0.0, Minimum temperature difference between meso and thermo homologs.
- `data.min_thermo_temp`: Current value - 60.0, Minimum temperature of thermo homologs.
- `data.max_meso_temp`: Current value - 40.0, Maximum temperature of meso homologs.
- `data.min_align_cov`: Current value - 0.95, Minimum alignment coverage of homologs to be considered a protein pair.
- `data.mmseq_params`: MMseqs2 clustering parameters.
- `data.test_size`: Current value - 0.1, Size of the test set as a fraction.
- `data.dev_sample_data`: Current value - false, Indicates if sample data is used for development.
- `data.additional_filters`: SQL expression strings used for filtering protein pair data.

### Metrics
- `./data/data_metrics.yaml`: Contains metrics like number of clusters in data splits and total data count.

### Additional Notes
- The script uses MMseqs2 clustering parameters defined in the parameters file to split data after applying additional SQL filters to refine the data selection process.

## Script Name: blast_test_train.py

### Description
- Performs BLAST (Basic Local Alignment Search Tool) analysis of the test set against the training set. This script is used to compute sequence identity and E-value scores, aiding in assessing the quality and diversity of the dataset.

### Input Files
- `./scripts/blast_test_train.py`: The main script file.
- `./data/dataset/`: The dataset directory used as input for BLAST analysis.

### Output Files
- No specific output files other than metrics and plots.

### Plots Generated
- `./data/plots/test_train_blast_hist.png`: Histograms showing the distribution of sequence identity and E-value scores of the test set blasted against the training set.

### Parameters
- This script does not use specific parameters from the params file.

### Metrics
- `./data/test_train_blast_metrics.json`: Contains scores for sequence identity and E-value of the test set blasted against the training set.

## Script Name: train.py

### Description
- Trains a model for either translation or reconstruction tasks using a pre-trained model from Huggingface. It utilizes the Accelerate library for efficient training, potentially leveraging DeepSpeed configurations.

### Input Files
- `./scripts/train.py`: The main training script.
- `./data/dataset/`: The dataset directory used for training the model.
- `./.config/accelerate/default_config.yaml`: DeepSpeed configuration file.

### Output Files
- `./data/nomelt-model/model/`: The directory where the trained Huggingface model is saved.
- `./data/nomelt-model/live/report.md`: A markdown report of the training run.
- `./data/nomelt-model/live/plots/`: Directory containing plots generated during the training process.

### Parameters
- `model.pretrained_model`: Current value - "Rostlab/prot_t5_xl_uniref50", Pretrained model string from Huggingface or disk.
- `model.task`: Current value - 'translation', Task for the model (translation or reconstruction).
- `model.generation_max_length`: Current value - 250, Maximum length of the generated sequence.
- `model.model_hyperparams`: Hyperparameters for the model, passed to the Huggingface parameters class.
- `training.*`: Various training parameters such as batch size, learning rate, epochs, early stopping criteria, etc.

### Metrics
- `./data/nomelt-model/live/metrics.json`: JSON file containing metrics from the training run as a function of steps.

### Plots Generated
- `./data/nomelt-model/live/static/`: Directory containing static plots from the training run.

## Script Name: train_all.py

### Description
- Similar to `train.py`, this script trains the model on the entire dataset without evaluation. It is intended for producing the final model.

### Input Files
- `./scripts/train_all.py`: Main script for training on the full dataset.
- `./data/dataset/`: Dataset directory used for training.
- `./.config/accelerate/default_config.yaml`: Configuration file for Accelerate/DeepSpeed.

### Output Files
- `./data/nomelt-model-full/model/`: Directory where the final model is saved.
- `./data/nomelt-model-full/live/report.md`: Markdown report of the training run.
- `./data/nomelt-model-full/live/plots/`: Directory of plots from the training run.

### Parameters
- Inherits all the parameters defined for `train.py`, such as `model.pretrained_model`, `model.task`, training-related parameters (`training.reweight`, `training.freeze_early_layers`, etc.), and other model and training configurations.

### Metrics
- `./data/nomelt-model-full/live/metrics.json`: JSON file of metrics from the training run as a function of steps.

### Plots Generated
- `./data/nomelt-model-full/live/static/`: Directory of static plots from the training run.

### Additional Notes
- This script is designed for final model training, utilizing the entire dataset without dividing it into training and evaluation sets. For downstream use and evaluation.

## Script Name: make_predictions.py

### Description
- Executes model inference on the test set, generating predictions. It utilizes the trained model to predict sequences based on the provided test data.

### Input Files
- `./scripts/make_predictions.py`: Main script for generating predictions.
- `./data/nomelt-model/model/`: Trained model directory.
- `./data/dataset/`: Dataset directory containing the test set.

### Output Files
- `./data/nomelt-model/predictions.tsv`: A TSV file containing the predictions, including mesophilic, thermophilic, and predicted sequences.

### Parameters
- `model.generation_max_length`: Current value - 250, Maximum length of generated sequence.
- `model.generation_num_beams`: Current value - 10, Number of beams used in beam search for sequence generation.

## Script Name: score_predictions.py

### Description
- Calculates and assesses various metrics on the test set based on the predictions made by the model. It's used to evaluate the model's performance quantitatively.

### Input Files
- `./scripts/score_predictions.py`: The main script for scoring predictions.
- `./data/nomelt-model/predictions.tsv`: File containing the model's predictions on the test set.

### Output Files
- `./data/nomelt-model/test_scores.json`: A JSON file containing metrics computed on the test set. These metrics typically include standard evaluation measures like BLEU score for sequence predictions.

### Additional Notes
- Measures Huggingface metrics like Google BLEU

## Script Name: translate_enh1.py

### Description
- Generates a thermostable variant of the protein ENH1 using the trained model. This script specifically focuses on applying the model for the translation task to create a modified sequence with enhanced properties.

### Input Files
- `./scripts/translate_enh1.py`: The script for translating ENH1.
- `./data/nomelt-model-full/model/`: Directory of the fully trained model.

### Output Files
- `./data/enh/translate_enh1.json`: A JSON file containing the generated sequence for ENH1.

### Parameters
- `model.generation_max_length`: Current value - 250, Defines the maximum length of the generated sequence.
- `model.generation_num_beams`: Current value - 10, Specifies the number of beams to use in the beam search algorithm during generation.

## Script Name: estimate_trans_energy_enh1.py

### Description
- Estimates the thermal stability of the generated sequence for ENH1. This script uses a specified estimator to evaluate the thermostability of the protein variant produced by the model.

### Input Files
- `./scripts/estimate_trans_energy_enh1.py`: Script for estimating thermal stability.
- `./data/enh/translate_enh1.json`: JSON file containing the generated sequence for ENH1.
- `./.config/af_singularity_config.yaml`: Configuration file for the AlphaFold2-based estimator.

### Output Files
- `./data/enh/translated_energy_enh1.json`: A JSON file with the estimated thermal stability of the generated sequence.
- `./data/enh/initial_estimate/`: Directory containing data dump from running the estimator.

### Parameters
- `optimize.estimator`: Current value - mAFminDGEstimator, Name of the estimator for estimating thermal stability.
- `optimize.estimator_args`: Arguments for the estimator, including AlphaFold parameters and settings.

I apologize for that oversight. Let me include the current values of the parameters for the `optimize_enh1.py` script:

## Script Name: optimize_enh1.py

### Description
- Employs thermostability estimation and Optuna optimization to find a subset of mutations likely to increase the thermostability of ENH1.

### Input Files
- `./scripts/optimize_enh1.py`: Main optimization script.
- `./data/enh/translate_enh1.json`: Contains the generated sequence for ENH1.

### Output Files
- `./data/enh/optimize_enh1/`: Data dump from the optimization process.
- `./data/enh/optimize_enh1_trials.csv`: Trials from the optimizer.
- `./data/enh/optimize_enh1_results.json`: Results, including the best sequence and estimated thermal stability.

### Parameters
- `optimize.estimator`: Current value - mAFminDGEstimator, Estimator for thermal stability.
- `optimize.estimator_args`: Configuration for the estimator.
- `optimize.n_trials`: Current value - 100, Number of optimization trials.
- `optimize.direction`: Current value - minimize, Direction of optimization.
- `optimize.sampler`: Current value - NSGAIISampler, Sampler used in Optuna.
- `optimize.cut_tails`: Number of gap spaces to keep on ends of alignment.
- `optimize.gap_compressed_mutations`: Current value - true, Whether to consider a string of gaps a single mutation.
- `optimize.matrix`: Current value - BLOSUM62, Substitution matrix used.
- `optimize.match_score`: Current value - 1, Score for matches.
- `optimize.mismatch_score`: Current value - -1, Score for mismatches.
- `optimize.gapopen`: Current value - -4, Gap open penalty.
- `optimize.gapextend`: Current value - -1, Gap extend penalty.
- `optimize.penalize_end_gaps`: Whether to penalize end gaps.
- `optimize.sampler_args`: Specific arguments for the sampler.
- `optimize.optuna_overwrite`: Current value - true, Whether to overwrite Optuna study.

### Additional Notes
- This script demonstrates the use of the model to design a protein - eg take the mutations suggested by the model and optimize over the variation, in this case with in silico estimators.

## Script Name: compute_test_embeddings.py

### Description
- Computes and saves residue-wise embeddings for the final layers of the model on the test set. This process is crucial for evaluating residue-wise scores and understanding the model's behavior at a granular level.

### Input Files
- `./scripts/compute_test_embeddings.py`: Script for computing embeddings.
- `./data/dataset/`: Dataset directory containing the test set.
- `./data/nomelt-model/model/`: Trained model directory.

### Output Files
- `./data/nomelt-model/test_loss.json`: JSON file documenting the loss on the test set.

### Metrics
- The script focuses on generating detailed embeddings rather than using specific parameters from the params file.

### Additional Notes
- used for disulfide bond analysis

## Script Name: compare_sequence_alignment.py

### Description
- Compares the generated sequences to thermophilic structures using Smith and Waterman alignments. It also includes comparisons to mesophilic sequences, providing a comprehensive view of the model's performance in terms of sequence similarity.

### Input Files
- `./scripts/compare_sequence_alignment.py`: Script for performing sequence alignments.
- `./data/nomelt-model/predictions.tsv`: Contains the model's predictions including mesophilic, thermophilic, and generated sequences.

### Output Files
- `./data/nomelt-model/test_predictions_aligned_results.json`: JSON file containing Smith-Waterman alignments in a triangular format for each meso, thermo, and generated sequence.

## Script Name: compare_structure.py

### Description
- Analyzes the structural alignment of generated sequences against thermophilic structures, offering insights into structural similarities and differences.

### Input Files
- `./scripts/compare_structure.py`: Script for structural comparison.
- `./data/nomelt-model/predictions.tsv`: Predictions including mesophilic, thermophilic, and generated sequences.

### Output Files
- `./data/nomelt-model/structure_metrics.json`: Contains metrics from FATCAT and DSSP comparisons.

## Script Name: data_estimator_distribution.py

### Description
- Evaluates the thermostability of a sample of data, including mesophilic, thermophilic, and generated sequences, using a thermostability estimator.

### Input Files
- `./scripts/data_estimator_distribution.py`: Script for estimating thermostability.
- `./data/nomelt-model/predictions.tsv`: Contains model predictions.

### Output Files
- `./data/thermo_gen_estimated.json`: JSON file with estimated thermal stability scores for mesophilic, thermophilic, and generated sequences.

## Script Name: zero_shot_estimation.py

### Description
- Conducts zero-shot estimation comparing model outputs to experimental data for thermal stability targets, focusing on proteins with multiple accumulated mutations.

### Input Files
- `./scripts/zero_shot_experiment.py`: Script for zero-shot estimation.
- `./data/nomelt-model-full/model/`: Trained model directory.

### Output Files
- `./data/nomelt-model-full/zero_shot_estimated.json`: JSON file with experimental versus model scores.
- `./data/plots/exp_tm_scores.png`: Regression plot of experimental versus model scores.

## Script Name: protein_gym_benchmark.py

### Description
- Runs a benchmark on the Protein Gym T50 dataset using the trained model, assessing the model's performance on a standardized set.

### Input Files
- `./scripts/protein_gym_benchmark.py`: Script for the benchmark test.
- `./data/nomelt-model-full/model/`: Directory containing the fully trained model.

### Output Files
- `./data/nomelt-model-full/lipa_gym_zero_shot.json`: Contains correlation coefficients and fraction of statistically significant pairs in the DMS library predicted by the model.
- `./data/plots/lipa_gym.png`: Density plot of experimental versus model scores.

# Proof of principle scripts
These are not part of the linear pipeline to train, evaluated, and use the model, but provide ground truths and various tests that aid on comparison.


## Script Name: natural_diversity_entropy.py

### Description
- Computes cross entropy from searching natural thermophilic diversity. This script assesses the entropy in the natural diversity of thermophilic proteins.

### Input Files
- `./scripts/proof_of_principle/natural_diversity_entropy.py`: Python script for calculating natural diversity entropy.
- `./data/dataset`: Dataset containing protein pairs for entropy analysis.

### Output Files
- `./data/proof_of_principle/natural_diversity_entropy.json`: JSON file containing the results of natural diversity entropy calculation, in cross entropy.

## Script Name: mAF_length_diff_test.py

### Description
- Analyzes the effect of length differences on mAF scores by comparing normalized mAF scores against experimental scores for a variety of indel variants.

### Input Files
- `./scripts/proof_of_principle/mAF_length_diff_test.py`: Script to test the impact of length differences on mAF scores.

### Output Files
- `./data/proof_of_principle/mAF_length_diff_test.json`: JSON file containing mAF scores normalized for length differences.

### Plots Generated
- `./data/plots/mAF_length_diff_test.png`: Plot showing mAF scores normalized vs experimental scores for a number of indel variants.

## Script Name: consensus_estimated.py

### Description
- Calculates the mAF score for a consensus sequence. This script assesses the thermostability of the consensus sequence using the af2dg thermostability estimator.

### Input Files
- `./scripts/proof_of_principle/consensus_estimated.py`: Script for estimating the mAF score of the consensus sequence.

### Output Files
- `./data/proof_of_principle/consensus_estimated.json`: JSON file containing the mAF score for the consensus sequence.

## Script Name: enh1_vary_mutations.py

### Description
- Evaluates the impact of various mutations on the thermostability of the wild type enhancer 1 (enh1) protein. This script applies different mutations to enh1 and uses the af2dg thermostability estimator to analyze their effects.

### Input Files
- `./scripts/proof_of_principle/enh1_vary_mutations.py`: Python script for assessing the impact of mutations on enh1.

### Output Files
- `./data/proof_of_principle/vary_mutations.json`: JSON file containing the results of thermostability estimation for different mutations applied to enh1.

## Script Name: enh1_consensus_optimize.py

### Description
- Executes the optimization process on the wild type enhancer 1 (enh1) versus the literature consensus sequence. This script aims to identify a subset of mutations from the wild type to the consensus sequence that increases the thermostability of enh1 the most.

### Input Files
- `./scripts/proof_of_principle/enh1_consensus_optimize.py`: Script for running the optimization process.

### Output Files
- `./data/proof_of_principle/optimize_enh1_cons_results.json`: JSON file containing the optimization results, best sequence, structure file, and estimated thermal stability.
- `./data/proof_of_principle/optimize_enh1_cons_trials.csv`: CSV file detailing the trials conducted during the optimization process.

## Script Name: enh1_random_opt.py

### Description
- Conducts optimization on the wild type enhancer 1 (enh1) against a random set of muations upon the WT sequence. This script aims to identify a subset of mutations from the wild type to a random sequence that increases the thermostability of enh1 the most.

### Input Files
- `./scripts/proof_of_principle/enh1_random_opt.py`: Script for optimizing enh1 against a random sequence.

### Output Files
- `./data/proof_of_principle/optimize_enh1_random_results.json`: JSON file containing the results of the optimization, including the best sequence, structural file, and estimated thermal stability.
- `./data/proof_of_principle/optimize_enh1_rand_trials.csv`: CSV file with details of the optimization trials.

## Script Name: tests_in_training_set.py

### Description
- Searches for specific test examples like ENH, LovD, and Lipase A within the training set of the model. This script evaluates the presence of these case study proteins in the training dataset.

### Input Files
- `./scripts/proof_of_principle/check_training_set_for_case_studies.py`: Script to check for specific proteins in the training dataset.

### Output Files
- `./data/enh/training_data_homologs.json`: JSON file containing the e-values of any hits found in the training set, relevant to the case study proteins.
