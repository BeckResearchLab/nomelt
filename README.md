# nomelt
Designing high temperature protein sequences via learned language processing

## Install

TODO
conda installs
rosetta install seperate
af install seperate
FATCAT

## Config

Accelerate config

## ENV variables
`TMP`
`AF_APPTAINER_SCRIPT`
`LOG_LEVEL`

## TODO

- HF caching: caching and DVC clash a little bit. Be default, when you do operations on a HF dataset, it creates cache files in the dataset folder, which makes DVC think the dataset has changed. If you want to use those cache operations, you have to commit the data/dataset object to DVC with changes. Instead, it would be better if cacheing dataset operations were abstracted out into their own script, and the cache file manually pathed to a dvc tracked output. Thus if paramters in the pipeline would change the operation, that one stage would be run, but downstream stages that use the same operations (eg. tokenization) could reuse that dvc tracked cache.

## Models
`nomelt` models are all designed to produce amino acid sequences of proteins stable at high temperature, conditioned on an input

- `nomelt-s2s`: (seq -> seq) translate from moderate to high temperature variants of proteins
  - Traditional architectures (eg seq2seq T5, autoregressive Decoder only) and tokenizers for protein LM usable out of the box
  - TODO
- `nomelt-hmm`: (hmm -> seq) develop high temperature variants of protein from a representative HMM
  - Traditional architectures (eg seq2seq T5, autoregressive Decoder only) LM usable out of the box
  - Novel tokenizer required to prepare HMM inputs
  - TODO
- `nomelt-hmm+`: (hmm, T -> seq) develop variants of a protein stable at a specific temperature from a representative HMM
  - Novel architecure and tokenizer required
  - TODO


