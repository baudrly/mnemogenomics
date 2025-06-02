# mnemogenomics

A proof of concept (adapted from [nanoGPT](https://github.com/karpathy/nanoGPT)) for training a language model on FASTA files:

* When fed a description as a prompt, it outputs a sequence (DNA, RNA or protein)
* When fed a sequence, it outputs a description (presumably a prediction)

This can give rise to syntethic generation *and* classification capabilities if trained on enough data and scaled up (assuming some version of the scaling hypothesis).

In order to determine the nature of the input (natural language description, DNA, RNA or protein sequence), special separator tokens are added to the vocabulary so that they trigger the beginning (or end) of a given sequence.

A live demo (with a very rudimentary example model) is available [here](https://baudrly.github.io/mnemogenomics).
