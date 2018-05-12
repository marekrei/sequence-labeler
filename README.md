Sequence labeler
=========================

This is a neural network sequence labeling system. Given a sequence of tokens, it will learn to assign labels to each token. Can be used for named entity recognition, POS-tagging, error detection, chunking, CCG supertagging, etc.

The main model implements a bidirectional LSTM for sequence tagging. In addition, you can incorporate character-level information -- either by concatenating a character-based representation, or by using an attention/gating mechanism for combining it with a word embedding.

Run with:

    python experiment.py config.conf

Preferably with Tensorflow set up to use CUDA, so the process can run on a GPU. The script will train the model on the training data, test it on the test data, and print various evaluation metrics.

Note: The original sequence labeler was implemented in Theano, but since Theano is soon ending support, I have reimplemented it in TensorFlow. I also used the chance to refactor the code a bit, and it should be better in every way. However, if you need the specific code used in previously published papers, you'll need to refer to older commits.

Requirements
-------------------------

* python (tested with 2.7.12 and 3.5.2)
* numpy (tested with 1.13.3 and 1.14.0)
* tensorflow (tested with 1.3.0 and 1.4.1)


Data format
-------------------------

The training and test data is expected in standard CoNLL-type tab-separated format. One word per line, separate column for token and label, empty line between sentences.

For error detection, this would be something like:

    I       c
    saws    i
    the     c
    show    c
    

The first column is assumed to be the token and the last column is the label. There can be other columns in the middle, which are currently not used. For example:

    EU      NNP     I-NP    S-ORG
    rejects VBZ     I-VP    O
    German  JJ      I-NP    S-MISC
    call    NN      I-NP    O
    to      TO      I-VP    O
    boycott VB      I-VP    O
    British JJ      I-NP    S-MISC
    lamb    NN      I-NP    O
    .       .       O       O
    

Configuration
-------------------------

Edit the values in config.conf as needed:

* **path_train** - Path to the training data, in CoNLL tab-separated format. One word per line, first column is the word, last column is the label. Empty lines between sentences.
* **path_dev** - Path to the development data, used for choosing the best epoch.
* **path_test** - Path to the test file. Can contain multiple files, colon separated.
* **conll_eval** - Whether the standard CoNLL NER evaluation should be run.
* **main_label** - The output label for which precision/recall/F-measure are calculated. Does not affect accuracy or measures from the CoNLL eval.
* **model_selector** - What is measured on the dev set for model selection: "dev_conll_f:high" for NER and chunking, "dev_acc:high" for POS-tagging, "dev_f05:high" for error detection.
* **preload_vectors** - Path to the pretrained word embeddings, in word2vec plain text format. If your embeddings are in binary, you can use [convertvec](https://github.com/marekrei/convertvec) to convert them to plain text.
* **word_embedding_size** - Size of the word embeddings used in the model.
* **crf_on_top** - If True, use a CRF as the output layer. If False, use softmax instead.
* **emb_initial_zero** - Whether word embeddings should have zero initialisation by default.
* **train_embeddings** - Whether word embeddings should be updated during training.
* **char_embedding_size** - Size of the character embeddings.
* **word_recurrent_size** - Size of the word-level LSTM hidden layers.
* **char_recurrent_size** - Size of the char-level LSTM hidden layers.
* **hidden_layer_size** - Size of the extra hidden layer on top of the bi-LSTM.
* **char_hidden_layer_size** - Size of the extra hidden layer on top of the character-based component.
* **lowercase** - Whether words should be lowercased when mapping to word embeddings.
* **replace_digits** - Whether all digits should be replaced by 0.
* **min_word_freq** - Minimal frequency of words to be included in the vocabulary. Others will be considered OOV.
* **singletons_prob** - The probability of mapping words that appear only once to OOV instead during training.
* **allowed_word_length** - Maximum allowed word length, clipping the rest. Can be necessary if the text contains unreasonably long tokens, eg URLs.
* **max_train_sent_length** - Discard sentences longer than this limit when training.
* **vocab_include_devtest** - Load words from dev and test sets also into the vocabulary. If they don't appear in the training set, they will have the default representations from the preloaded embeddings.
* **vocab_only_embedded** - Whether the vocabulary should contain only words in the pretrained embedding set.
* **initializer** - The method used to initialize weight matrices in the network.
* **opt_strategy** - The method used for weight updates.
* **learningrate** - Learning rate.
* **clip** - Clip the gradient to a range.
* **batch_equal_size** - Create batches of sentences with equal length.
* **epochs** - Maximum number of epochs to run.
* **stop_if_no_improvement_for_epochs** - Training will be stopped if there has been no improvement for n epochs.
* **learningrate_decay** - If performance hasn't improved for 3 epochs, multiply the learning rate with this value.
* **dropout_input** - The probability for applying dropout to the word representations. 0.0 means no dropout.
* **dropout_word_lstm** - The probability for applying dropout to the LSTM outputs.
* **tf_per_process_gpu_memory_fraction** - The fraction of GPU memory that the process can use.
* **tf_allow_growth** - Whether the GPU memory usage can grow dynamically.
* **main_cost** - Control the weight of the main labeling objective.
* **lmcost_max_vocab_size** = Maximum vocabulary size for the language modeling loss. The remaining words are mapped to a single entry.
* **lmcost_hidden_layer_size** = Hidden layer size for the language modeling loss.
* **lmcost_gamma** - Weight for the language modeling loss. 
* **char_integration_method** - How character information is integrated. Options are: "none" (not integrated), "concat" (concatenated), "attention" (the method proposed in Rei et al. (2016)).
* **save** - Path to save the model.
* **load** - Path to load the model.
* **garbage_collection** - Whether garbage collection is explicitly called. Makes things slower but can operate with bigger models.
* **lstm_use_peepholes** - Whether to use the LSTM implementation with peepholes.
* **random_seed** - Random seed for initialisation and data shuffling. This can affect results, so for robust conclusions I recommend running multiple experiments with different seeds and averaging the metrics.






Printing output
-------------------------

There is now a separate script for loading a saved model and using it to print output for a given input file. Use the **save** option in the config file for saving the model. The input file needs to be in the same format as the training data (one word per line, labels in a separate column). The labels are expected for printing output as well. If you don't know the correct labels, just print any valid label in that field.

To print the output, run:

    python print_output.py labels model_file input_file

This will print the input file to standard output, with an extra column at the end that shows the prediction. 

You can also use:

    python print_output.py probs model_file input_file

This will print the individual probabilities for each of the possible labels.
If the model is using CRFs, the *probs* option will output unnormalised state scores without taking the transitions into account.


References
-------------------------

The main sequence labeling model is described here:

[**Compositional Sequence Labeling Models for Error Detection in Learner Writing**](http://aclweb.org/anthology/P/P16/P16-1112.pdf)  
Marek Rei and Helen Yannakoudakis  
*In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL-2016)*
  

The character-level component is described here:

[**Attending to characters in neural sequence labeling models**](https://aclweb.org/anthology/C/C16/C16-1030.pdf)  
Marek Rei, Gamal K.O. Crichton and Sampo Pyysalo  
*In Proceedings of the 26th International Conference on Computational Linguistics (COLING-2016)*

The language modeling objective is described here:

[**Semi-supervised Multitask Learning for Sequence Labeling**](https://arxiv.org/abs/1704.07156)  
Marek Rei  
*In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL-2017)*

The CRF implementation is based on:

[**Neural Architectures for Named Entity Recognition**](https://arxiv.org/abs/1603.01360)  
Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami and Chris Dyer  
*In Proceedings of NAACL-HLT 2016*
  

The conlleval.py script is from: https://github.com/spyysalo/conlleval.py




License
---------------------------

The code is distributed under the Affero General Public License 3 (AGPL-3.0) by default. 
If you wish to use it under a different license, feel free to get in touch.

Copyright (c) 2018 Marek Rei

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
