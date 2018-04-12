import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from .reservoir import Reservoir
from ..utils import ridge_regression


class ESN(nn.Module):
    """ Applies an Echo State Network to an input sequence. Multi-layer Echo
    State Network is based on paper
    Deep Echo State Network (DeepESN): A Brief Survey - Gallicchio, Micheli 2017

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h.
        output_size: The number of expected features in the output y.
        num_layers: Number of recurrent layers. Default: 1
        nonlinearity: The non-linearity to use ['tanh'|'relu'|'id'].
            Default: 'tanh'
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        leaking_rate: Leaking rate of reservoir's neurons. Default: 1
        spectral_radius: Desired spectral radius of recurrent weight matrix.
            Default: 0.9
        w_ih_scale: Scale factor for first layer's input weights (w_ih_l0). It
            can be a number or a list of size '1 + input_size' and first element
            is the bias' scale factor. Default: 1
        lambda_reg: Ridge regression's shrinkage parameter. Default: 1
        density: Recurrent weight matrix's density. Default: 1
        w_io: If 'True', then the network uses trainable input-to-output
            connections. Default: ``False``
        readout_training: Readout's traning algorithm ['offline'|'online']. If
            'offline', the network will learn readout's parameters during the
            forward pass using ridge regression. The coefficients are computed
            using SVD. If 'online', gradients are accumulated during backward
            pass.

    Inputs: input, h_0, washout, target
        input (seq_len, batch, input_size): tensor containing the features of
            the input sequence. The input can also be a packed variable length
            sequence. See `torch.nn.utils.rnn.pack_padded_sequence`
        h_0 (num_layers * num_directions, batch, hidden_size): tensor containing
             the initial reservoir's hidden state for each element in the batch.
             Defaults to zero if not provided.
        washout: number of initial timesteps during which output of the
            reservoir is not forwarded to the readout.
        target (seq_len*batch - washout*batch, output_size): tensor containing
            the features of the batch's target sequences rolled out along one
            axis, minus the washouts and the padded values. It is only needed
            for readout's training in offline mode. Use `prepare_target` to
            compute it.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output
        features (h_k) from the readout, for each k.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the reservoir's hidden state for k=seq_len.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 nonlinearity='tanh',
                 batch_first=False, leaking_rate=1, spectral_radius=0.9,
                 w_ih_scale=1,
                 lambda_reg=0, density=1, w_io=False,
                 readout_training='offline'):
        super(ESN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        if nonlinearity == 'tanh':
            mode = 'RES_TANH'
        elif nonlinearity == 'relu':
            mode = 'RES_RELU'
        elif nonlinearity == 'id':
            mode = 'RES_ID'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        self.batch_first = batch_first
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.lambda_reg = lambda_reg
        self.density = density
        self.w_io = w_io
        if readout_training == 'offline' or readout_training == 'online':
            self.readout_training = readout_training
        else:
            raise ValueError("Unknown readout training algorithm '{}'".format(
                readout_training))

        self.reservoir = Reservoir(mode, input_size, hidden_size, num_layers,
                                   leaking_rate, spectral_radius,
                                   w_ih_scale, density, batch_first=batch_first)

        if w_io:
            self.readout = nn.Linear(input_size + hidden_size * num_layers,
                                     output_size)
        else:
            self.readout = nn.Linear(hidden_size * num_layers, output_size)
        if readout_training == 'offline':
            self.readout.weight.requires_grad = False

    def forward(self, input, h_0, washout=0, target=None):
        is_packed = isinstance(input, PackedSequence)

        output, hidden = self.reservoir(input, h_0)
        if is_packed:
            output, seq_lengths = pad_packed_sequence(output,
                                                      batch_first=self.batch_first)
            seq_lengths = [x - washout for x in seq_lengths]
        else:
            if self.batch_first:
                seq_lengths = output.size(0) * [output.size(1) - washout]
            else:
                seq_lengths = output.size(1) * [output.size(0) - washout]

        if self.batch_first:
            output = output.transpose(0, 1)
        output = output[washout:]

        if self.w_io:
            if is_packed:
                padded_input, _ = pad_packed_sequence(input,
                                                      batch_first=self.batch_first)
                if self.batch_first:
                    padded_input = padded_input.transpose(0, 1)
                output = torch.cat([padded_input[washout:], output], -1)
            else:
                if self.batch_first:
                    input = input.transpose(0, 1)
                output = torch.cat([input[washout:], output], -1)

        if self.readout_training == 'online' or target is None:
            output = self.readout(output)

        elif self.readout_training == 'offline' and target is not None:
            batch_size = output.size(1)

            if self.w_io:
                X = torch.ones(target.size(0),
                               1 + self.input_size + self.hidden_size * self.num_layers)
            else:
                X = torch.ones(target.size(0),
                               1 + self.hidden_size * self.num_layers)

            col = 0
            for s in range(batch_size):
                X[col:col + seq_lengths[s], 1:] = output[:seq_lengths[s],
                                                  s].data
                col += seq_lengths[s]

            W = ridge_regression(X, target, self.lambda_reg).t()
            self.readout.bias = nn.Parameter(W[:, 0])
            self.readout.weight = nn.Parameter(W[:, 1:])

            flat_output = self.readout(Variable(X[:, 1:]))

            output = torch.zeros((seq_lengths[0], batch_size, self.output_size))
            col = 0
            for s in range(batch_size):
                output[:seq_lengths[s], s] = flat_output[
                                             col:col + seq_lengths[s]].data
                col += seq_lengths[s]
            output = Variable(output)

        if self.batch_first:
            output = output.transpose(0, 1)
        # if is_packed:
        #     output = pack_padded_sequence(output, seq_lengths,
        #                                   batch_first=self.batch_first)

        return output, hidden

    def reset_parameters(self):
        self.reservoir.reset_parameters()
        self.readout.reset_parameters()


