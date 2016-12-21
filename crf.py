import theano
import numpy

# CRF implementation based on Lample et al.
# "Neural Architectures for Named Entity Recognition"

floatX=theano.config.floatX

def log_sum(x, axis=None):
    x_max_value = x.max(axis=axis)
    x_max_tensor = x.max(axis=axis, keepdims=True)
    return x_max_value + theano.tensor.log(theano.tensor.exp(x - x_max_tensor).sum(axis=axis))


def forward(observation_weights, transition_weights, return_best_sequence=False):
    def recurrence(observation_weights, previous_scores, transition_weights):
        previous_scores = previous_scores.dimshuffle(0, 1, 'x')
        observation_weights = observation_weights.dimshuffle(0, 'x', 1)
        scores = previous_scores + observation_weights + transition_weights.dimshuffle('x', 0, 1)
        if return_best_sequence:
            best_scores = scores.max(axis=1)
            best_states = scores.argmax(axis=1)
            return best_scores, best_states
        else:
            return log_sum(scores, axis=1)

    initial = observation_weights[0]
    crf_states, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observation_weights[1:],],
        non_sequences=transition_weights
    )

    if return_best_sequence:
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[theano.tensor.arange(previous.shape[0]), previous],
            outputs_info=theano.tensor.cast(theano.tensor.argmax(crf_states[0][-1], axis=1), 'int32'),
            sequences=theano.tensor.cast(crf_states[1][::-1], 'int32')
        )
        sequence = theano.tensor.concatenate([sequence[::-1], [theano.tensor.argmax(crf_states[0][-1], axis=1)]])
        return sequence, crf_states[0]
    else:
        return log_sum(crf_states[-1], axis=1)


def construct(name, input_tensor, n_labels, gold_labels, fn_create_parameter_matrix):
    transition_weights = fn_create_parameter_matrix(name + "_crf_transition_weights", (n_labels + 2, n_labels + 2))

    small = -1000.0
    padding_start = theano.tensor.zeros((input_tensor.shape[0], 1, n_labels + 2)) + small
    padding_start = theano.tensor.set_subtensor(padding_start[:,:,-2], 0.0)
    padding_end = theano.tensor.zeros((input_tensor.shape[0], 1, n_labels + 2)) + small
    padding_end = theano.tensor.set_subtensor(padding_end[:,:,-1], 0.0)
    observation_weights = theano.tensor.concatenate([input_tensor, theano.tensor.zeros((input_tensor.shape[0], input_tensor.shape[1], 2)) + small], axis=2)
    observation_weights = theano.tensor.concatenate([padding_start, observation_weights, padding_end], axis=1)
    observation_weights = observation_weights.dimshuffle(1,0,2) # reordering the tensor (words, sentences, labels)

    # Score from tags
    real_paths_scores = input_tensor[theano.tensor.arange(input_tensor.shape[0])[:, numpy.newaxis], theano.tensor.arange(input_tensor.shape[1]), gold_labels].sum(axis=1)

    # Score from transition_weights
    padding_id_start = theano.tensor.zeros((gold_labels.shape[0], 1), dtype=numpy.int32) + n_labels
    padding_id_end = theano.tensor.zeros((gold_labels.shape[0], 1), dtype=numpy.int32) + n_labels + 1
    padded_gold_labels = theano.tensor.concatenate([padding_id_start, gold_labels, padding_id_end], axis=1)
    real_paths_scores += transition_weights[
        padded_gold_labels[theano.tensor.arange(gold_labels.shape[0])[:, numpy.newaxis], theano.tensor.arange(gold_labels.shape[1] + 1)],
        padded_gold_labels[theano.tensor.arange(gold_labels.shape[0])[:, numpy.newaxis], theano.tensor.arange(gold_labels.shape[1] + 1) + 1]
    ].sum(axis=1)

    all_paths_scores = forward(observation_weights, transition_weights)

    best_sequence, scores = forward(observation_weights, transition_weights, return_best_sequence=True)

    scores = scores.dimshuffle(1,0,2)[:,:-1,:-2]
    best_sequence = best_sequence.dimshuffle(1,0)[:,1:-1]

    return all_paths_scores, real_paths_scores, best_sequence, scores

