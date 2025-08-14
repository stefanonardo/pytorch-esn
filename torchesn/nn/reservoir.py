"""
Reservoir implementation for Echo State Networks.

This module implements the core reservoir dynamics for ESNs, including the recurrent
neural network with fixed random weights, leaky integration, and various activation
functions. The reservoir is the key component that provides the temporal memory and
nonlinear transformation capabilities of Echo State Networks.

Key components:
- Reservoir: Main reservoir module with configurable dynamics
- ReservoirCells: Individual neuron update functions (Tanh, ReLU, Identity)
- Recurrent/VariableRecurrent: Handlers for fixed and variable-length sequences
- AutogradReservoir: Factory function for creating reservoir computations

The reservoir weights are randomly initialized once and kept fixed during training,
following the Echo State Property which ensures that the reservoir state is uniquely
determined by the driving input sequence.

Author: Stefano Nardo
"""
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence


def apply_permutation(tensor, permutation, dim=1):
    """Apply a permutation to reorder tensor elements along a specified dimension.
    
    Args:
        tensor (Tensor): Input tensor to permute
        permutation (Tensor): Permutation indices 
        dim (int): Dimension along which to apply permutation
        
    Returns:
        Tensor: Permuted tensor
    """
    return tensor.index_select(dim, permutation)


class Reservoir(nn.Module):
    """Reservoir module implementing the recurrent dynamics for Echo State Networks.
    
    The reservoir consists of randomly initialized, sparsely connected recurrent neurons
    with fixed weights. It provides the temporal memory and nonlinear dynamics that
    allow ESNs to model complex sequential patterns.
    
    Key principles:
    - Echo State Property: Reservoir states are uniquely determined by input history
    - Fixed random weights: Only readout layer is trained, reservoir weights are fixed
    - Sparse connectivity: Typically only 10-30% of recurrent connections are non-zero
    - Spectral radius control: Eigenvalues scaled to ensure stability and memory
    
    Args:
        mode (str): Reservoir neuron activation function type.
            Options: 'RES_TANH', 'RES_RELU', 'RES_ID' (identity/linear)
        input_size (int): Number of input features per timestep
        hidden_size (int): Number of neurons in each reservoir layer  
        num_layers (int): Number of reservoir layers (for Deep ESN)
        leaking_rate (float): Leaky integration rate ∈ (0,1]. Controls memory:
            - 1.0: No leakage, full memory retention
            - <1.0: Faster forgetting, shorter memory
        spectral_radius (float): Largest eigenvalue magnitude of recurrent weights.
            Controls stability and memory capacity:
            - <1.0: Stable, finite memory (recommended)
            - =1.0: Critical, potentially unstable
            - >1.0: Unstable, explosive dynamics
        w_ih_scale (Tensor): Scaling factors for input-to-hidden weights.
            Shape: (1 + input_size,) where first element scales bias
        density (float): Fraction of non-zero recurrent connections ∈ (0,1].
            Lower values create sparser, more structured reservoirs
        bias (bool, optional): Whether to include bias terms. Default: True
        batch_first (bool, optional): Input format (batch, seq, features) vs 
            (seq, batch, features). Default: False
    
    Forward pass:
        Input: (seq_len, batch, input_size) or PackedSequence
        Output: (seq_len, batch, hidden_size*num_layers), final_hidden_state
        
    Examples:
        >>> # Standard single-layer reservoir
        >>> reservoir = Reservoir('RES_TANH', input_size=10, hidden_size=100,
        ...                      num_layers=1, leaking_rate=0.9, 
        ...                      spectral_radius=0.95, w_ih_scale=torch.ones(11),
        ...                      density=0.1)
        >>> 
        >>> # Deep reservoir with 3 layers
        >>> deep_reservoir = Reservoir('RES_RELU', input_size=50, hidden_size=200,
        ...                           num_layers=3, leaking_rate=0.8,
        ...                           spectral_radius=0.9, w_ih_scale=torch.ones(51),
        ...                           density=0.2)
    
    Note:
        The reservoir weights are initialized once and remain fixed throughout training.
        This is fundamental to the ESN approach and ensures computational efficiency.
    """

    def __init__(self, mode, input_size, hidden_size, num_layers, leaking_rate,
                 spectral_radius, w_ih_scale,
                 density, bias=True, batch_first=False):
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
        """Initialize reservoir weights according to ESN principles.
        
        This method sets up the reservoir weights with the following strategy:
        1. Input weights (w_ih): Uniform random [-1,1], scaled by w_ih_scale
        2. Bias weights: Uniform random [-1,1], scaled by w_ih_scale[0] 
        3. Recurrent weights (w_hh): Uniform random, then:
           - Apply sparsity mask based on density parameter
           - Scale to achieve desired spectral radius
           
        The spectral radius scaling ensures the Echo State Property:
        W_hh := W_hh * (spectral_radius / max(|eigenvalues(W_hh)|))
        
        This initialization is critical for reservoir stability and performance.
        """
        weight_dict = self.state_dict()
        for key, value in weight_dict.items():
            if key == 'weight_ih_l0':
                # First layer input weights: uniform [-1,1] scaled by w_ih_scale
                nn.init.uniform_(value, -1, 1)
                value *= self.w_ih_scale[1:]  # Skip bias scale (index 0)
            elif re.fullmatch('weight_ih_l[^0]*', key):
                # Higher layer input weights: uniform [-1,1], no additional scaling
                nn.init.uniform_(value, -1, 1)
            elif re.fullmatch('bias_ih_l[0-9]*', key):
                # Bias weights: uniform [-1,1] scaled by bias scale factor
                nn.init.uniform_(value, -1, 1)
                value *= self.w_ih_scale[0]  # First element is bias scale
            elif re.fullmatch('weight_hh_l[0-9]*', key):
                # Recurrent weights: the core of reservoir dynamics
                w_hh = torch.Tensor(self.hidden_size * self.hidden_size)
                w_hh.uniform_(-1, 1)
                
                # Apply sparsity: randomly zero out connections
                if self.density < 1:
                    zero_weights = torch.randperm(
                        int(self.hidden_size * self.hidden_size))
                    zero_weights = zero_weights[
                                   :int(
                                       self.hidden_size * self.hidden_size * (
                                                   1 - self.density))]
                    w_hh[zero_weights] = 0
                
                # Reshape and scale to desired spectral radius
                w_hh = w_hh.view(self.hidden_size, self.hidden_size)
                abs_eigs = torch.abs(torch.linalg.eigvals(w_hh))
                # Scale largest eigenvalue to spectral_radius
                weight_dict[key] = w_hh * (self.spectral_radius / torch.max(abs_eigs))

        self.load_state_dict(weight_dict)

    def check_input(self, input, batch_sizes):
        """Validate input tensor dimensions and features.
        
        Args:
            input (Tensor): Input tensor to validate
            batch_sizes (Tensor, optional): Batch sizes for PackedSequence
            
        Raises:
            RuntimeError: If input dimensions don't match expected format
        """
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input, batch_sizes):
        """Calculate expected hidden state dimensions based on input.
        
        Args:
            input (Tensor): Input tensor
            batch_sizes (Tensor, optional): Batch sizes for PackedSequence
            
        Returns:
            tuple: Expected hidden state shape (num_layers, batch_size, hidden_size)
        """
        if batch_sizes is not None:
            mini_batch = batch_sizes[0]
            mini_batch = int(mini_batch)
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        expected_hidden_size = (self.num_layers, mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
        """Validate hidden state tensor dimensions.
        
        Args:
            hx (Tensor): Hidden state tensor to validate
            expected_hidden_size (tuple): Expected shape tuple
            msg (str): Error message template
            
        Raises:
            RuntimeError: If hidden state shape doesn't match expected
        """
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input, hidden, batch_sizes):
        """Validate all forward pass arguments for consistency.
        
        Args:
            input (Tensor): Input sequences
            hidden (Tensor): Hidden state tensor  
            batch_sizes (Tensor, optional): Batch sizes for PackedSequence
            
        Raises:
            RuntimeError: If arguments have incompatible shapes or formats
        """
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx, permutation):
        """Apply permutation to hidden state for PackedSequence compatibility.
        
        When using PackedSequence, batch elements may be reordered for efficiency.
        This function applies the corresponding permutation to hidden states.
        
        Args:
            hx (Tensor): Hidden state tensor
            permutation (Tensor, optional): Permutation indices, None for no-op
            
        Returns:
            Tensor: Permuted hidden state
        """
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    def forward(self, input, hx=None):
        """Forward pass through the reservoir layers.
        
        Computes the reservoir state evolution over time according to:
        h(t) = (1-α)h(t-1) + α·f(W_ih·x(t) + W_hh·h(t-1) + b_ih)
        
        where α is the leaking rate and f is the activation function.
        
        Args:
            input (Tensor or PackedSequence): Input sequences of shape:
                - (seq_len, batch, input_size) if batch_first=False
                - (batch, seq_len, input_size) if batch_first=True
                Can also be PackedSequence for variable-length sequences
            hx (Tensor, optional): Initial hidden state of shape:
                (num_layers, batch, hidden_size). Defaults to zeros.
                
        Returns:
            tuple: (output, final_hidden) where:
                - output: Reservoir activations over time, same format as input
                - final_hidden: Final hidden state for sequence continuation
                
        Note:
            The reservoir computation is performed without gradients since
            reservoir weights are fixed. Only the readout layer requires gradients.
        """
        # Handle both standard tensors and PackedSequence inputs
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        # Initialize hidden state if not provided
        if hx is None:
            hx = input.new_zeros(self.num_layers, max_batch_size,
                                 self.hidden_size, requires_grad=False)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        flat_weight = None  # Not used in current implementation

        # Validate input arguments
        self.check_forward_args(input, hx, batch_sizes)
        
        # Create the appropriate computation function
        func = AutogradReservoir(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            train=self.training,
            variable_length=is_packed,
            flat_weight=flat_weight,
            leaking_rate=self.leaking_rate
        )
        
        # Execute reservoir computation
        output, hidden = func(input, self.all_weights, hx, batch_sizes)
        
        # Repack output if input was PackedSequence
        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            
        # Restore original batch ordering for hidden state
        return output, self.permute_hidden(hidden, unsorted_indices)

    def extra_repr(self):
        s = '({input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        s += ')'
        return s.format(**self.__dict__)

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
                      batch_first=False, train=True,
                      batch_sizes=None, variable_length=False, flat_weight=None,
                      leaking_rate=1):
    """Factory function for creating reservoir computation graphs.
    
    This function creates the appropriate computational graph for reservoir
    dynamics based on the activation function and sequence type (fixed vs variable length).
    It handles the complexity of different sequence formats and activation functions
    in a unified interface.
    
    Args:
        mode (str): Activation function type ('RES_TANH', 'RES_RELU', 'RES_ID')
        input_size (int): Input feature dimension
        hidden_size (int): Reservoir size per layer
        num_layers (int): Number of reservoir layers
        batch_first (bool): Input tensor format preference
        train (bool): Training mode flag
        batch_sizes (Tensor, optional): Batch sizes for PackedSequence
        variable_length (bool): Whether input has variable-length sequences
        flat_weight (optional): Flattened weight tensor (unused in current impl)
        leaking_rate (float): Leaky integration coefficient
        
    Returns:
        function: Configured forward function for reservoir computation
        
    The returned function signature is:
        forward(input, weight, hidden, batch_sizes) -> (output, final_hidden)
    """
    if mode == 'RES_TANH':
        cell = ResTanhCell
    elif mode == 'RES_RELU':
        cell = ResReLUCell
    elif mode == 'RES_ID':
        cell = ResIdCell

    if variable_length:
        layer = (VariableRecurrent(cell, leaking_rate),)
    else:
        layer = (Recurrent(cell, leaking_rate),)

    func = StackedRNN(layer,
                      num_layers,
                      False,
                      train=train)

    def forward(input, weight, hidden, batch_sizes):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight, batch_sizes)

        if batch_first and not variable_length:
            output = output.transpose(0, 1)

        return output, nexth

    return forward


def Recurrent(inner, leaking_rate):
    """Create a recurrent computation function for fixed-length sequences.
    
    This function handles the temporal loop for standard (non-packed) sequences
    where all sequences in the batch have the same length. It applies the
    reservoir cell update at each timestep.
    
    Args:
        inner (function): Reservoir cell function (ResTanhCell, ResReLUCell, etc.)
        leaking_rate (float): Leaky integration rate
        
    Returns:
        function: Recurrent forward function with signature:
            forward(input, hidden, weight, batch_sizes) -> (final_hidden, all_outputs)
    """
    def forward(input, hidden, weight, batch_sizes):
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, leaking_rate, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def VariableRecurrent(inner, leaking_rate):
    """Create a recurrent computation function for variable-length sequences.
    
    This function handles PackedSequence inputs where different sequences in the
    batch can have different lengths. It carefully manages the hidden states as
    sequences of different lengths finish at different timesteps.
    
    The function maintains a stack of hidden states for sequences that have ended,
    allowing proper reconstruction of the full batch hidden state at the end.
    
    Args:
        inner (function): Reservoir cell function  
        leaking_rate (float): Leaky integration rate
        
    Returns:
        function: Variable recurrent forward function for PackedSequence inputs
    """
    def forward(input, hidden, weight, batch_sizes):
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


def StackedRNN(inners, num_layers, lstm=False, train=True):
    """Create a multi-layer recurrent computation function.
    
    This function stacks multiple reservoir layers to create Deep ESNs. Each layer
    receives input from the previous layer and maintains its own hidden state.
    The outputs from all layers are concatenated to form the final representation.
    
    Args:
        inners (list): List of recurrent layer functions
        num_layers (int): Number of layers to stack
        lstm (bool): LSTM compatibility flag (unused for reservoirs)
        train (bool): Training mode flag
        
    Returns:
        function: Multi-layer forward function that processes input through
                 all layers sequentially
    """
    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight, batch_sizes):
        assert (len(weight) == total_layers)
        next_hidden = []
        all_layers_output = []

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l], batch_sizes)
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)
            all_layers_output.append(input)

        all_layers_output = torch.cat(all_layers_output, -1)
        next_hidden = torch.cat(next_hidden, 0).view(
            total_layers, *next_hidden[0].size())

        return next_hidden, all_layers_output

    return forward


def ResTanhCell(input, hidden, leaking_rate, w_ih, w_hh, b_ih=None):
    """Reservoir cell with hyperbolic tangent activation function.
    
    Computes: h(t) = (1-α)h(t-1) + α·tanh(W_ih·x(t) + W_hh·h(t-1) + b_ih)
    
    The tanh activation provides bounded, smooth nonlinearity that is well-suited
    for temporal modeling tasks. This is the most commonly used reservoir activation.
    
    Args:
        input (Tensor): Current input x(t), shape (batch, input_size)
        hidden (Tensor): Previous hidden state h(t-1), shape (batch, hidden_size)  
        leaking_rate (float): Leaky integration rate α ∈ (0,1]
        w_ih (Tensor): Input-to-hidden weight matrix
        w_hh (Tensor): Hidden-to-hidden (recurrent) weight matrix
        b_ih (Tensor, optional): Input bias vector
        
    Returns:
        Tensor: Updated hidden state h(t)
    """
    hy_ = torch.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh))
    hy = (1 - leaking_rate) * hidden + leaking_rate * hy_
    return hy


def ResReLUCell(input, hidden, leaking_rate, w_ih, w_hh, b_ih=None):
    """Reservoir cell with Rectified Linear Unit (ReLU) activation function.
    
    Computes: h(t) = (1-α)h(t-1) + α·ReLU(W_ih·x(t) + W_hh·h(t-1) + b_ih)
    
    ReLU activation provides sparse, piecewise-linear dynamics that can be
    beneficial for certain tasks. The unbounded positive range may require
    careful tuning of spectral radius and input scaling.
    
    Args:
        input (Tensor): Current input x(t), shape (batch, input_size)
        hidden (Tensor): Previous hidden state h(t-1), shape (batch, hidden_size)
        leaking_rate (float): Leaky integration rate α ∈ (0,1] 
        w_ih (Tensor): Input-to-hidden weight matrix
        w_hh (Tensor): Hidden-to-hidden (recurrent) weight matrix
        b_ih (Tensor, optional): Input bias vector
        
    Returns:
        Tensor: Updated hidden state h(t)
    """
    hy_ = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh))
    hy = (1 - leaking_rate) * hidden + leaking_rate * hy_
    return hy


def ResIdCell(input, hidden, leaking_rate, w_ih, w_hh, b_ih=None):
    """Reservoir cell with identity (linear) activation function.
    
    Computes: h(t) = (1-α)h(t-1) + α·(W_ih·x(t) + W_hh·h(t-1) + b_ih)
    
    The identity activation creates a linear reservoir, which can be useful for
    tasks requiring linear temporal dependencies or as a baseline for comparison.
    The dynamics are purely determined by the weight matrices and leaky integration.
    
    Args:
        input (Tensor): Current input x(t), shape (batch, input_size)
        hidden (Tensor): Previous hidden state h(t-1), shape (batch, hidden_size)
        leaking_rate (float): Leaky integration rate α ∈ (0,1]
        w_ih (Tensor): Input-to-hidden weight matrix  
        w_hh (Tensor): Hidden-to-hidden (recurrent) weight matrix
        b_ih (Tensor, optional): Input bias vector
        
    Returns:
        Tensor: Updated hidden state h(t)
        
    Note:
        Linear reservoirs may have limited modeling capacity compared to nonlinear
        alternatives, but can be analyzed more easily from a theoretical perspective.
    """
    hy_ = F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh)
    hy = (1 - leaking_rate) * hidden + leaking_rate * hy_
    return hy
