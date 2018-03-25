import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence


class Reservoir(nn.Module):

    def __init__(self, mode, input_size, hidden_size, num_layers, leaking_rate,
                 spectral_radius, w_ih_scale,
                 density, bias=True, batch_first=False, dropout=0):
        super(Reservoir, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.w_ih_scale = w_ih_scale
        self.density = density
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}

        self._all_weights = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size

            w_ih = nn.Parameter(torch.Tensor(hidden_size, layer_input_size))
            w_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            b_ih = nn.Parameter(torch.Tensor(hidden_size))
            layer_params = (w_ih, w_hh, b_ih)

            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            if bias:
                param_names += ['bias_ih_l{}{}']
            param_names = [x.format(layer, '') for x in param_names]

            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

            self.reset_parameters()

    def _apply(self, fn):
        ret = super(Reservoir, self)._apply(fn)
        return ret

    def reset_parameters(self):
        weight_dict = self.state_dict()
        for key, value in weight_dict.items():
            if key == 'weight_ih_l0':
                nn.init.uniform(value, -1, 1)
                value *= self.w_ih_scale[1:]
            elif re.fullmatch('weight_ih_l[^0]*', key):
                nn.init.uniform(value, -1, 1)
            elif re.fullmatch('bias_ih_l[0-9]*', key):
                nn.init.uniform(value, -1, 1)
                value *= self.w_ih_scale[0]
            elif re.fullmatch('weight_hh_l[0-9]*', key):
                nn.init.sparse(value, 1 - self.density)
                value *= (self.spectral_radius / torch.max(
                    torch.abs(torch.eig(value)[0])))

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = batch_sizes[0]
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        expected_hidden_size = (self.num_layers, mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size,
                              msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(
                    msg.format(expected_hidden_size, tuple(hx.size())))

        check_hidden_size(hidden, expected_hidden_size)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(
                1)

        if hx is None:
            hx = torch.autograd.Variable(input.data.new(self.num_layers,
                                                        max_batch_size,
                                                        self.hidden_size).zero_(),
                                         requires_grad=False)

        flat_weight = None

        self.check_forward_args(input, hx, batch_sizes)
        func = AutogradReservoir(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            batch_sizes=batch_sizes,
            dropout_state=self.dropout_state,
            flat_weight=flat_weight,
            leaking_rate=self.leaking_rate
        )
        output, hidden = func(input, self.all_weights, hx)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def __setstate__(self, d):
        super(Reservoir, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        self._all_weights = []
        for layer in range(num_layers):
            weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}']
            weights = [x.format(layer) for x in weights]
            if self.bias:
                self._all_weights += [weights]
            else:
                self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in
                self._all_weights]


def AutogradReservoir(mode, input_size, hidden_size, num_layers=1,
                      batch_first=False, dropout=0, train=True,
                      batch_sizes=None, dropout_state=None, flat_weight=None,
                      leaking_rate=1):
    if mode == 'RES_TANH':
        cell = ResTanhCell
    elif mode == 'RES_RELU':
        cell = ResReLUCell
    elif mode == 'RES_ID':
        cell = ResIdCell

    if batch_sizes is None:
        layer = (Recurrent(cell, leaking_rate),)
    else:
        layer = (VariableRecurrent(batch_sizes, cell, leaking_rate),)

    func = StackedRNN(layer,
                      num_layers,
                      False,
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight)

        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        return output, nexth

    return forward


def Recurrent(inner, leaking_rate):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, leaking_rate, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def VariableRecurrent(batch_sizes, inner, leaking_rate):
    def forward(input, hidden, weight):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], leaking_rate, *weight),)
            else:
                hidden = inner(step_input, hidden, leaking_rate, *weight)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):
    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert (len(weight) == total_layers)
        next_hidden = []
        all_layers_output = []

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)
            all_layers_output.append(input)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train,
                                  inplace=False)

        all_layers_output = torch.cat(all_layers_output, -1)
        next_hidden = torch.cat(next_hidden, 0).view(
            total_layers, *next_hidden[0].size())

        return next_hidden, all_layers_output

    return forward


def ResTanhCell(input, hidden, leaking_rate, w_ih, w_hh, b_ih=None):
    hy_ = F.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh))
    hy = (1 - leaking_rate) * hidden + leaking_rate * hy_
    return hy


def ResReLUCell(input, hidden, leaking_rate, w_ih, w_hh, b_ih=None):
    hy_ = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh))
    hy = (1 - leaking_rate) * hidden + leaking_rate * hy_
    return hy


def ResIdCell(input, hidden, leaking_rate, w_ih, w_hh, b_ih=None):
    hy_ = F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh)
    hy = (1 - leaking_rate) * hidden + leaking_rate * hy_
    return hy