import sys
import theano
import numpy
import collections
import cPickle
import lasagne

import crf
import recurrence

sys.setrecursionlimit(50000)
floatX=theano.config.floatX

class SequenceLabeler(object):
    def __init__(self, config):
        self.config = config
        self.params = collections.OrderedDict()
        self.rng = numpy.random.RandomState(config["random_seed"])

        word_ids = theano.tensor.imatrix('word_ids')
        char_ids = theano.tensor.itensor3('char_ids')
        char_mask = theano.tensor.ftensor3('char_mask')
        label_ids = theano.tensor.imatrix('label_ids')
        learningrate = theano.tensor.fscalar('learningrate')

        cost = 0.0
        input_tensor = None
        input_vector_size = 0

        self.word_embeddings = self.create_parameter_matrix('word_embeddings', (config["n_words"], config["word_embedding_size"]))
        input_tensor = self.word_embeddings[word_ids]
        input_vector_size = config["word_embedding_size"]

        char_embeddings = self.create_parameter_matrix('char_embeddings', (config["n_chars"], config["char_embedding_size"]))
        char_input_tensor = char_embeddings[char_ids].reshape((char_ids.shape[0]*char_ids.shape[1],char_ids.shape[2],config["char_embedding_size"]))
        char_mask_reshaped = char_mask.reshape((char_ids.shape[0]*char_ids.shape[1],char_ids.shape[2]))

        char_output_tensor = recurrence.create_birnn(char_input_tensor, config["char_embedding_size"], char_mask_reshaped, config["char_recurrent_size"], return_combined=True, fn_create_parameter_matrix=self.create_parameter_matrix, name="char_birnn")
        char_output_tensor = recurrence.create_feedforward(char_output_tensor, config["char_recurrent_size"]*2, config["word_embedding_size"], "tanh", fn_create_parameter_matrix=self.create_parameter_matrix, name="char_ff")
        char_output_tensor = char_output_tensor.reshape((char_ids.shape[0],char_ids.shape[1],config["word_embedding_size"]))

        if config["char_integration_method"] == "input":
            input_tensor = theano.tensor.concatenate([input_tensor, char_output_tensor], axis=2)
            input_vector_size += config["word_embedding_size"]

        elif config["char_integration_method"] == "attention":
            static_input_tensor = theano.gradient.disconnected_grad(input_tensor)
            is_unk = theano.tensor.eq(word_ids, config["unk_token_id"])
            is_unk_tensor = is_unk.dimshuffle(0,1,'x')
            char_output_tensor_normalised = char_output_tensor / char_output_tensor.norm(2, axis=2)[:, :, numpy.newaxis]
            static_input_tensor_normalised = static_input_tensor / static_input_tensor.norm(2, axis=2)[:, :, numpy.newaxis]
            cosine_cost = 1.0 - (char_output_tensor_normalised * static_input_tensor_normalised).sum(axis=2)
            cost += theano.tensor.switch(is_unk, 0.0, cosine_cost).sum()
            attention_evidence_tensor = theano.tensor.concatenate([input_tensor, char_output_tensor], axis=2)
            attention_output = recurrence.create_feedforward(attention_evidence_tensor, config["word_embedding_size"]*2, config["word_embedding_size"], "tanh", self.create_parameter_matrix, "attention_tanh")
            attention_output = recurrence.create_feedforward(attention_output, config["word_embedding_size"], config["word_embedding_size"], "sigmoid", self.create_parameter_matrix, "attention_sigmoid")
            input_tensor = input_tensor * attention_output + char_output_tensor * (1.0 - attention_output)

        processed_tensor = recurrence.create_birnn(input_tensor, input_vector_size, None, config["word_recurrent_size"], return_combined=False, fn_create_parameter_matrix=self.create_parameter_matrix, name="word_birnn")
        processed_tensor = recurrence.create_feedforward(processed_tensor, config["word_recurrent_size"]*2, config["narrow_layer_size"], "tanh", fn_create_parameter_matrix=self.create_parameter_matrix, name="narrow_ff")

        W_output = self.create_parameter_matrix('W_output', (config["narrow_layer_size"], config["n_labels"]))
        bias_output = self.create_parameter_matrix('bias_output', (config["n_labels"],))
        output = theano.tensor.dot(processed_tensor, W_output) + bias_output
        output = output[:,1:-1,:] # removing <s> and </s>

        if config["crf_on_top"] == True:
            all_paths_scores, real_paths_scores, best_sequence, scores = crf.construct("crf", output, config["n_labels"], label_ids, self.create_parameter_matrix)
            predicted_labels = best_sequence
            output_probs = scores
            cost += - (real_paths_scores - all_paths_scores).sum()
        else:
            output_probs = theano.tensor.nnet.softmax(output.reshape((word_ids.shape[0]*(word_ids.shape[1]-2), config["n_labels"])))
            predicted_labels = theano.tensor.argmax(output_probs.reshape((word_ids.shape[0], (word_ids.shape[1]-2), config["n_labels"])), axis=2)
            cost += theano.tensor.nnet.categorical_crossentropy(output_probs, label_ids.reshape((-1,))).sum()

        gradients = theano.tensor.grad(cost, self.params.values(), disconnected_inputs='ignore')
        updates = lasagne.updates.adadelta(gradients, self.params.values(), learningrate)

        input_vars_train = [word_ids, char_ids, char_mask, label_ids, learningrate]
        input_vars_test = [word_ids, char_ids, char_mask, label_ids]
        output_vars = [cost, predicted_labels]
        self.train = theano.function(input_vars_train, output_vars, updates=updates, on_unused_input='ignore', allow_input_downcast = True)
        self.test = theano.function(input_vars_test, output_vars, on_unused_input='ignore', allow_input_downcast = True)

    def create_parameter_matrix(self, name, size):
        param_vals = numpy.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
        param_shared = theano.shared(param_vals, name)
        self.params[name] = param_shared
        return param_shared


    def get_parameter_count(self):
        total = 0
        for key, val in self.params.iteritems():
            total += val.get_value().size
        return total

    def get_parameter_count_without_word_embeddings(self):
        total = 0
        for key, val in self.params.iteritems():
            if val == self.word_embeddings:
                continue
            total += val.get_value().size
        return total

    def save(self, filename):
        dump = {}
        dump["config"] = self.config
        dump["params"] = {}
        for param_name in self.params:
            dump["params"][param_name] = self.params[param_name].get_value()
        f = file(filename, 'wb')
        cPickle.dump(dump, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def load(filename, new_output_layer_size=None):
        f = file(filename, 'rb')
        dump = cPickle.load(f)
        f.close()
        if new_output_layer_size is not None:
            dump["n_labels"] = new_output_layer_size
        sequencelabeler = SequenceLabeler(dump["config"])
        for param_name in sequencelabeler.params:
            assert(param_name in dump["params"])
            if new_output_layer_size is not None and param_name in ["W_output", "bias_output"]:
                continue
            sequencelabeler.params[param_name].set_value(dump["params"][param_name])
        return sequencelabeler
