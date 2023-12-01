# Scripts

This folder contains scripts used to define the method's pipeline, proof of principle, and containerized interface.

At root level are scripts used for the training the model, evaluations, and some downstream evaluation of the results of the model on enh1:

- `prepare_data.py` - Used for getting the data ready for training. This includes querying learn2thermDB for the data and splitting with mmseqs2.
- `blast_test_train.py` - blast the test set against the train set and record metrics eg. number of sequences with hits, average identitty, etc.
  - NOTE: this script operates on the dataset, and HF Datasets adds cache files to the dataset that makes DVC think it has changed. If you would like to experiment on downstream scripts, it is advisable to commit the data/dataset object after each experiment (eg. the train stage) otherwise the cached operations will not be used, such as tokenization.
- `train.py` - Script to train the model. (adds caches to dataset)
- `make_predictions.py` - Executes the trained model on the test set to save generated sequences. (adds caches to dataset)
- `score_predictions.py` - Computes metrics on the test set based on the predictions. Classic NLP metrics.
- `compare_sequence_alignment.py` - compares the meso, thermo, and generated sequences using alignments.
- `compare_structure.py` - compares the meso, thermo, and generated sequences by ESMFold structure predictions and then dssp secondary structure analysis and FATCAT flexible alignment.
- 
- `translate_enh1.py` - Calls the model to generate a thermostable variant of enh1.
- `estimate_trans_energy_enh1.py` - Calls thermostability estimator on the generated sequence.
- `optimize_enh1.py` - Uses thermo estimator to find a subset of mutations likely to increase thermostability.
- `dna_binding_alignment.py` - Estimates the DNA binding of the enh1 variant proposed by the model as well as the optimization process.

`proof_of_principle` contains scripts used to test out sections of the method, independent of the NMT model. Many of the comparisons are using the previously published consensus homeodomain variant here:

- `enh1_vs_consensus_in_silico_estimator.py` - Checking that the in silico evaluator can differentiate enh1 and published stable variant.
- `data_estimator_distribution.py` - Calls the af2dg thermostability estimator on a sample of data and saves the scores.
- `enh1_vary_mutations.py` - Manually adds a few mutations to the wild type enh and calls the af2dg thermostability estimator to see its effects on the score.
- `enh1_consensus_optimize.py` - Runs the optimization process on the wild type enh versus the literature consensus sequence.
- `enh1_random_opt.py` - Runs the optimization process on the wild type enh versus a random sequence.
- `enh1_vs_consensus_docking_score.py` - Retrieves the docking scores from haddock of the wild type enh versus the consensus sequence.
