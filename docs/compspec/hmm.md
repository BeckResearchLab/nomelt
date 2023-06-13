# Components for using HMMs as inputs to ML models

These components are necessary for a future use case of a hmm -> high temp protein


#### HMModel
Represents a single HMM
Subclass https://pypi.org/project/hmm-profile/?

__Attributes__:
- `length`
- `has_compo`: bool, if compo data was retained

__Methods__:
- `__init__`: construction from arrays
- `from_string`: construction from string of whole model
- `to_pt`: create a pytorch tensor of the sequence by encoding position dependant weights


#### HMMTokenizer
Tokenizer that allows for input hmm strings to be passed through 
Needs to adhere to the tokenizer class, add methods as necessary.
Calls the above
