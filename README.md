# nomelt
Designing high temperature protein sequences via learned language processing

## Install

TODO
conda installs
rosetta install seperate
af install seperate
haddock install seperate


## Config

Accelerate config

## ENV variables
`TMP`
`AF_APPTAINER_SCRIPT`

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


