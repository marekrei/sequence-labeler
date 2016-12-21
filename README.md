Sequence labeler
=========================

Run with:

    python sequence_labeling_experiment.py config.conf

Preferably with Theano set up to use CUDA, so the process can run on a GPU.

Requirements
-------------------------

* numpy
* theano
* lasagne

Configuration
-------------------------

Edit the values in config.conf as needed:

* **path_train** - Path to the training data.
* **path_dev** - Path to the development data, used for choosing the best epoch.
* **path_test** - Path to the test file. Can contain multiple files, colon separated.
* **main_label** - The output label for which precision/recall/F-measure are calculated.
* **conll_eval** - Whether the standard CoNLL NER evaluation should be run.
* **preload_vectors** - Path to the pretrained word embeddings, in word2vec plain text format. If your embeddings are in binary, you can use [convertvec](https://github.com/marekrei/convertvec) to convert them to plain text.
* **word_embedding_size** - Size of the word embeddings used in the model.
* **char_embedding_size** - Size of the character embeddings.
* **word_recurrent_size** - Size of the word-level LSTM hidden layers.
* **char_recurrent_size** - Size of the char-level LSTM hidden layers.
* **narrow_layer_size** - Size of the extra hidden layer on top of the bi-LSTM.
* **best_model_selector** - What is measured on the dev set for model selection: "dev_conll_f:high" for NER and chunking, "dev_acc:high" for POS-tagging, "dev_f05:high" for error detection.
* **epochs** - Maximum number of epochs to run.
* **stop_if_no_improvement_for_epochs** - Training will be stopped if there has been no improvement for n epochs.
* **learningrate** - Learning rate.
* **min_word_freq** - Minimal frequency of words to be included in the vocabulary. Others will be considered OOV.
* **max_batch_size** - Maximum batch size.
* **save** - Path to save the model.
* **load** - Path to load the model.
* **random_seed** - Random seed for initialisation and data shuffling.
* **char_integration_method** - How character information is integrated. Options are: "none" (not integrated), "input" (concatenated), "attention" (the method proposed in Rei et al. (2016)).


References
-------------------------

If you use the main sequence labeling code, please reference:

Compositional Sequence Labeling Models for Error Detection in Learner Writing
Marek Rei and Helen Yannakoudakis
In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL-2016)


If you use the character-level attention component, please reference:

Attending to characters in neural sequence labeling models
Marek Rei, Sampo Pyysalo and Gamal K.O. Crichton
In Proceedings of the 26th International Conference on Computational Linguistics (COLING-2016)


The current CRF implementation is based on:

Neural Architectures for Named Entity Recognition
Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami and Chris Dyer
In Proceedings of NAACL-HLT 2016


The conlleval.py script is from:

https://github.com/spyysalo/conlleval.py


License
---------------------------

MIT License

Copyright (c) 2016 Marek Rei

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
