import theano
import numpy

floatX=theano.config.floatX

def create_birnn(input_tensor, input_size, mask, recurrent_size, return_combined, fn_create_parameter_matrix, name):
    rnn_mask = mask.dimshuffle(1,0) if (mask is not None) else None
    recurrent_forward = create_lstm(input_tensor.dimshuffle(1,0,2), input_size, rnn_mask, 
                                    recurrent_size, only_return_final=return_combined, go_backwards=False, fn_create_parameter_matrix=fn_create_parameter_matrix, name=name + "_forward")
    recurrent_backward = create_lstm(input_tensor.dimshuffle(1,0,2), input_size, rnn_mask, 
                                    recurrent_size, only_return_final=return_combined, go_backwards=True, fn_create_parameter_matrix=fn_create_parameter_matrix, name=name + "_backward")
    if return_combined == True:
        return theano.tensor.concatenate([recurrent_forward, recurrent_backward], axis=1)
    else:
        return theano.tensor.concatenate([recurrent_forward.dimshuffle(1,0,2), recurrent_backward.dimshuffle(1,0,2)], axis=2)


def create_lstm(input_tensor, input_size, mask, recurrent_size, only_return_final, go_backwards, fn_create_parameter_matrix, name):
    # LSTM. Following Graves et al.
    # "Hybrid speech recognition with deep bidirectional LSTM"
    def lstm_step(x, h_prev, c_prev, W_x, W_h, b, W_ci, W_cf, W_co):
        m_xhb = theano.tensor.dot(x, W_x) + theano.tensor.dot(h_prev, W_h) + b
        i = theano.tensor.nnet.sigmoid(_slice(m_xhb, 0, 4) + c_prev * W_ci)
        f = theano.tensor.nnet.sigmoid(_slice(m_xhb, 1, 4) + c_prev * W_cf)
        c = f * c_prev + i * theano.tensor.tanh(_slice(m_xhb, 2, 4))
        o = theano.tensor.nnet.sigmoid(_slice(m_xhb, 3, 4) + c * W_co)
        h = o * theano.tensor.tanh(c)
        return h, c

    def lstm_mask_step(x, mask, h_prev, c_prev, W_x, W_h, b, W_ci, W_cf, W_co):
        h_new, c_new = lstm_step(x, h_prev, c_prev, W_x, W_h, b, W_ci, W_cf, W_co)
        h = theano.tensor.switch(mask, h_new, h_prev)
        c = theano.tensor.switch(mask, c_new, c_prev)
        return h, c

    def _slice(M, slice_num, total_slices):
        if M.ndim == 3:
            l = M.shape[2] / total_slices
            return M[:, :, slice_num*l:(slice_num+1)*l]
        elif M.ndim == 2:
            l = M.shape[1] / total_slices
            return M[:, slice_num*l:(slice_num+1)*l]
        elif M.ndim == 1:
            l = M.shape[0] / total_slices
            return M[slice_num*l:(slice_num+1)*l]

    h_initial = theano.tensor.alloc(numpy.array(0, dtype=floatX), input_tensor.shape[1], recurrent_size)
    c_initial = theano.tensor.alloc(numpy.array(0, dtype=floatX), input_tensor.shape[1], recurrent_size)

    if mask is not None:
        mask = mask.dimshuffle(0, 1, 'x')
        fn_step = locals()["lstm_mask_step"]
        sequences = [input_tensor, mask]
    else:
        fn_step = locals()["lstm_step"]
        sequences = input_tensor

    W_x = fn_create_parameter_matrix('W_x_'+name, (input_size, recurrent_size*4))
    W_h = fn_create_parameter_matrix('W_h_'+name, (recurrent_size, recurrent_size*4))
    b = fn_create_parameter_matrix('b_'+name, (recurrent_size*4,))
    W_ci = fn_create_parameter_matrix('W_ci_'+name, (recurrent_size,))
    W_cf = fn_create_parameter_matrix('W_cf_'+name, (recurrent_size,))
    W_co = fn_create_parameter_matrix('W_co_'+name, (recurrent_size,))
    result, _ = theano.scan(
        fn_step,
        sequences = sequences,
        outputs_info = [h_initial, c_initial],
        non_sequences = [W_x, W_h, b, W_ci, W_cf, W_co],
        go_backwards=go_backwards)

    h = result[0]
    if only_return_final == True:
        h = h[-1]
    else:
        if go_backwards == True:
            h = h[::-1]
    return h



def create_feedforward(input_tensor, input_size, output_size, activation, fn_create_parameter_matrix, name):
    weights = fn_create_parameter_matrix('ff_weights_' + name, (input_size, output_size))
    bias = fn_create_parameter_matrix('ff_bias_' + name, (output_size,))
    output = theano.tensor.dot(input_tensor, weights) + bias
    if activation == "tanh":
        output = theano.tensor.tanh(output)
    elif activation == "sigmoid":
        output = theano.tensor.nnet.sigmoid(output)
    return output

