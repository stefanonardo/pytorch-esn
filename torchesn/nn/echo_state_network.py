import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from .reservoir import Reservoir
from ..utils import washout_tensor
import numpy as np


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
            can be a number or a tensor of size '1 + input_size' and first element
            is the bias' scale factor. Default: 1
        lambda_reg: Ridge regression's shrinkage parameter. Default: 1
        density: Recurrent weight matrix's density. Default: 1
        w_io: If 'True', then the network uses trainable input-to-output
            connections. Default: ``False``
        readout_training: Readout's traning algorithm ['gd'|'svd'|'cholesky'|'inv'].
            If 'gd', gradients are accumulated during backward
            pass. If 'svd', 'cholesky' or 'inv', the network will learn readout's
            parameters during the forward pass using ridge regression. The
            coefficients are computed using SVD, Cholesky decomposition or
            standard ridge regression formula. 'gd', 'cholesky' and 'inv'
            permit the usage of mini-batches to train the readout.
            If 'inv' and matrix is singular, pseudoinverse is used.
        output_steps: defines how the reservoir's output will be used by ridge
            regression method ['all', 'mean', 'last'].
            If 'all', the entire reservoir output matrix will be used.
            If 'mean', the mean of reservoir output matrix along the timesteps
            dimension will be used.
            If 'last', only the last timestep of the reservoir output matrix
            will be used.
            'mean' and 'last' are useful for classification tasks.

    Inputs: input, washout, h_0, target
        input (seq_len, batch, input_size): tensor containing the features of
            the input sequence. The input can also be a packed variable length
            sequence. See `torch.nn.utils.rnn.pack_padded_sequence`
        washout (batch): number of initial timesteps during which output of the
            reservoir is not forwarded to the readout. One value per batch's
            sample.
        h_0 (num_layers, batch, hidden_size): tensor containing
             the initial reservoir's hidden state for each element in the batch.
             Defaults to zero if not provided.

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
                 nonlinearity='tanh', batch_first=False, leaking_rate=1,
                 spectral_radius=0.9, w_ih_scale=1, lambda_reg=0, density=1,
                 w_io=False, readout_training='svd', output_steps='all'):
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
        if type(w_ih_scale) != torch.Tensor:
            self.w_ih_scale = torch.ones(input_size + 1)
            self.w_ih_scale *= w_ih_scale
        else:
            self.w_ih_scale = w_ih_scale

        self.lambda_reg = lambda_reg
        self.density = density
        self.w_io = w_io
        if readout_training in {'gd', 'svd', 'cholesky', 'inv'}:
            self.readout_training = readout_training
        else:
            raise ValueError("Unknown readout training algorithm '{}'".format(
                readout_training))

        self.reservoir = Reservoir(mode, input_size, hidden_size, num_layers,
                                   leaking_rate, spectral_radius,
                                   self.w_ih_scale, density,
                                   batch_first=batch_first)

        if w_io:
            self.readout = nn.Linear(input_size + hidden_size * num_layers,
                                     output_size)
        else:
            self.readout = nn.Linear(hidden_size * num_layers, output_size)
        if readout_training == 'offline':
            self.readout.weight.requires_grad = False

        if output_steps in {'all', 'mean', 'last'}:
            self.output_steps = output_steps
        else:
            raise ValueError("Unknown task '{}'".format(
                output_steps))

        self.XTX = None
        self.XTy = None

    def forward(self, input, washout, h_0=None, target=None):
        with torch.no_grad():
            is_packed = isinstance(input, PackedSequence)

            output, hidden = self.reservoir(input, h_0)
            if is_packed:
                output, seq_lengths = pad_packed_sequence(output,
                                                          batch_first=self.batch_first)
            else:
                if self.batch_first:
                    seq_lengths = output.size(0) * [output.size(1)]
                else:
                    seq_lengths = output.size(1) * [output.size(0)]

            if self.batch_first:
                output = output.transpose(0, 1)

            output, seq_lengths = washout_tensor(output, washout, seq_lengths)

            if self.w_io:
                if is_packed:
                    input, input_lengths = pad_packed_sequence(input,
                                                          batch_first=self.batch_first)
                else:
                    input_lengths = [input.size(0)] * input.size(1)

                if self.batch_first:
                    input = input.transpose(0, 1)

                input, _ = washout_tensor(input, washout, input_lengths)
                output = torch.cat([input, output], -1)

            if self.readout_training == 'gd' or target is None:
                with torch.enable_grad():
                    output = self.readout(output)

                    if is_packed:
                        for i in range(output.size(1)):
                            if seq_lengths[i] < output.size(0):
                                output[seq_lengths[i]:, i] = 0

                    if self.batch_first:
                        output = output.transpose(0, 1)

                    # Uncomment if you want packed output.
                    # if is_packed:
                    #     output = pack_padded_sequence(output, seq_lengths,
                    #                                   batch_first=self.batch_first)

                    return output, hidden

            else:
                batch_size = output.size(1)

                X = torch.ones(target.size(0), 1 + output.size(2), device=target.device)
                row = 0
                for s in range(batch_size):
                    if self.output_steps == 'all':
                        X[row:row + seq_lengths[s], 1:] = output[:seq_lengths[s],
                                                          s]
                        row += seq_lengths[s]
                    elif self.output_steps == 'mean':
                        X[row, 1:] = torch.mean(output[:seq_lengths[s], s], 0)
                        row += 1
                    elif self.output_steps == 'last':
                        X[row, 1:] = output[seq_lengths[s] - 1, s]
                        row += 1

                if self.readout_training == 'cholesky':
                    if self.XTX is None:
                        self.XTX = torch.mm(X.t(), X)
                        self.XTy = torch.mm(X.t(), target)
                    else:
                        self.XTX += torch.mm(X.t(), X)
                        self.XTy += torch.mm(X.t(), target)

                elif self.readout_training == 'svd':
                    # Scikit-Learn SVD solver for ridge regression.
                    U, s, V = torch.svd(X)
                    idx = s > 1e-15  # same default value as scipy.linalg.pinv
                    s_nnz = s[idx][:, None]
                    UTy = torch.mm(U.t(), target)
                    d = torch.zeros(s.size(0), 1, device=X.device)
                    d[idx] = s_nnz / (s_nnz ** 2 + self.lambda_reg)
                    d_UT_y = d * UTy
                    W = torch.mm(V, d_UT_y).t()

                    self.readout.bias = nn.Parameter(W[:, 0])
                    self.readout.weight = nn.Parameter(W[:, 1:])
                elif self.readout_training == 'inv':
                    if self.XTX is None:
                        self.XTX = torch.mm(X.t(), X)
                        self.XTy = torch.mm(X.t(), target)
                    else:
                        self.XTX += torch.mm(X.t(), X)
                        self.XTy += torch.mm(X.t(), target)

                return None, None

    def fit(self):
        if self.readout_training in {'gd' ,'svd'}:
            return

        if self.readout_training == 'cholesky':
            W = torch.gesv(self.XTy,
                           self.XTX + self.lambda_reg * torch.eye(
                               self.XTX.size(0), device=self.XTX.device))[0].t()
            self.XTX = None
            self.XTy = None

            self.readout.bias = nn.Parameter(W[:, 0])
            self.readout.weight = nn.Parameter(W[:, 1:])
        elif self.readout_training == 'inv':
            I = (self.lambda_reg * torch.eye(self.XTX.size(0))).to(
                self.XTX.device)
            A = self.XTX + I

            if torch.det(A) != 0:
                W = torch.mm(torch.inverse(A), self.XTy).t()
            else:
                pinv = torch.pinverse(A)
                W = torch.mm(pinv, self.XTy).t()

            self.readout.bias = nn.Parameter(W[:, 0])
            self.readout.weight = nn.Parameter(W[:, 1:])

            self.XTX = None
            self.XTy = None

    def reset_parameters(self):
        self.reservoir.reset_parameters()
        self.readout.reset_parameters()


