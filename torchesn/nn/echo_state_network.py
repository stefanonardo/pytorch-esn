"""
Echo State Network (ESN) implementation in PyTorch.

This module provides a complete implementation of Echo State Networks, a type of
reservoir computing model that excels at temporal sequence modeling tasks. The
implementation supports both single-layer and multi-layer (Deep ESN) architectures
with various training algorithms and output aggregation methods.

Author: Stefano Nardo
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from .reservoir import Reservoir
from ..utils import washout_tensor


class ESN(nn.Module):
    """Echo State Network (ESN) implementation for sequence modeling and time series prediction.
    
    An Echo State Network is a recurrent neural network where the hidden recurrent weights
    are fixed and randomly initialized, while only the output (readout) layer is trained.
    This makes training extremely fast and stable. Multi-layer support is based on
    Deep Echo State Networks (DeepESN) from Gallicchio & Micheli (2017).
    
    Args:
        input_size (int): Number of expected features in the input sequences.
        hidden_size (int): Number of neurons in each reservoir layer.
        output_size (int): Number of output features/classes.
        num_layers (int, optional): Number of reservoir layers for Deep ESN. Default: 1
        nonlinearity (str, optional): Activation function for reservoir neurons.
            Options: 'tanh', 'relu', 'id' (identity/linear). Default: 'tanh'
        batch_first (bool, optional): If True, input/output tensors are 
            (batch, seq, feature). If False, (seq, batch, feature). Default: False
        leaking_rate (float, optional): Leaky integration rate ∈ (0,1]. 
            Controls how much of previous state to retain. 1.0 = no leakage. Default: 1.0
        spectral_radius (float, optional): Spectral radius of reservoir weight matrix.
            Controls memory capacity and stability. Should be ≤ 1.0. Default: 0.9
        w_ih_scale (float or Tensor, optional): Scaling factor for input-to-hidden weights.
            Can be scalar or tensor of size (1 + input_size) where first element
            scales bias. Default: 1.0
        lambda_reg (float, optional): L2 regularization parameter for ridge regression.
            Higher values increase regularization. Default: 0.0
        density (float, optional): Density of recurrent connections ∈ (0,1].
            1.0 = fully connected, lower values = sparser. Default: 1.0
        w_io (bool, optional): Whether to include direct input-to-output connections.
            Useful for tasks requiring input memory. Default: False
        readout_training (str, optional): Training algorithm for readout layer.
            Options:
            - 'svd': Single-pass SVD (most stable, no mini-batching)
            - 'cholesky': Cholesky decomposition (stable, supports batching)  
            - 'inv': Matrix inversion (fast, supports batching)
            - 'gd': Gradient descent (supports all PyTorch optimizers)
            Default: 'svd'
        output_steps (str, optional): How to aggregate reservoir outputs for training.
            Options:
            - 'all': Use all timesteps (for sequence-to-sequence)
            - 'mean': Average over time (for sequence classification)
            - 'last': Use only final timestep (for sequence classification)
            Default: 'all'
    
    Examples:
        >>> # Time series prediction
        >>> esn = ESN(input_size=1, hidden_size=100, output_size=1)
        >>> 
        >>> # Sequence classification with mean pooling
        >>> esn = ESN(input_size=10, hidden_size=200, output_size=5, 
        ...           output_steps='mean', readout_training='cholesky')
        >>> 
        >>> # Deep ESN with custom parameters
        >>> esn = ESN(input_size=50, hidden_size=300, output_size=10,
        ...           num_layers=3, spectral_radius=0.95, leaking_rate=0.8)
    
    Note:
        For large datasets that don't fit in memory, use 'cholesky' or 'inv' 
        readout training which support mini-batch accumulation via multiple
        forward passes followed by a single fit() call.

    
    Input/Output Specification:
        
        Forward pass inputs:
            input (Tensor or PackedSequence): Input sequences of shape:
                - (seq_len, batch, input_size) if batch_first=False
                - (batch, seq_len, input_size) if batch_first=True
                Can also be PackedSequence for variable-length sequences.
                
            washout (list of int): Number of initial timesteps to discard per sample.
                Length must equal batch size. Washout helps stabilize reservoir 
                dynamics by ignoring transient behavior.
                
            h_0 (Tensor, optional): Initial hidden state of shape:
                (num_layers, batch, hidden_size). Defaults to zeros if not provided.
                
            target (Tensor, optional): Target values for offline training methods.
                Required shape depends on output_steps:
                - 'all': (total_seq_len_after_washout, output_size) 
                - 'mean'/'last': (batch_size, output_size)
                Use torchesn.utils.prepare_target() to format correctly.
        
        Forward pass outputs:
            output (Tensor or None): Model predictions of shape:
                - (seq_len, batch, output_size) if batch_first=False
                - (batch, seq_len, output_size) if batch_first=True  
                Returns None during offline training accumulation phase.
                
            hidden (Tensor): Final reservoir hidden states of shape:
                (num_layers, batch, hidden_size). Useful for continuing sequences.
    
    Training Workflow:
        
        Offline Training (Ridge Regression):
            1. Create model with readout_training='svd'/'cholesky'/'inv'
            2. For 'svd': Single forward pass with target, automatic training
            3. For 'cholesky'/'inv': Multiple forward passes, then call fit()
            
            >>> # SVD method (single batch)
            >>> model = ESN(10, 100, 1, readout_training='svd')
            >>> model(X, washout, target=y)  # Automatically trains
            >>> 
            >>> # Cholesky method (mini-batches)  
            >>> model = ESN(10, 100, 1, readout_training='cholesky')
            >>> for batch in dataloader:
            ...     model(X_batch, washout_batch, target=y_batch)  # Accumulate
            >>> model.fit()  # Solve accumulated system
        
        Online Training (Gradient Descent):
            1. Create model with readout_training='gd'
            2. Use standard PyTorch training loop with optimizer
            
            >>> model = ESN(10, 100, 1, readout_training='gd')
            >>> optimizer = torch.optim.Adam(model.parameters())
            >>> for epoch in range(epochs):
            ...     for batch in dataloader:
            ...         optimizer.zero_grad()
            ...         output, _ = model(X_batch, washout_batch)
            ...         loss = criterion(output, y_batch)
            ...         loss.backward()
            ...         optimizer.step()
    
    References:
        - Jaeger, H. (2001). The "echo state" approach to analysing and training 
          recurrent neural networks. GMD Technical Report 148.
        - Gallicchio, C., & Micheli, A. (2017). Deep Echo State Network (DeepESN): 
          A Brief Survey. arXiv preprint arXiv:1712.04323.
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
        self.X = None

    def forward(self, input, washout, h_0=None, target=None):
        """Forward pass through the Echo State Network.
        
        Args:
            input (Tensor or PackedSequence): Input sequences
            washout (list of int): Washout lengths per batch sample  
            h_0 (Tensor, optional): Initial hidden state
            target (Tensor, optional): Target for offline training
            
        Returns:
            tuple: (output, hidden) where:
                - output: Model predictions or None during offline training
                - hidden: Final reservoir hidden states
                
        Note:
            When using offline training methods ('cholesky', 'inv'), multiple
            forward passes accumulate statistics. Call fit() after all batches
            to solve the linear system and set readout parameters.
        """
        with torch.no_grad():
            # Handle both regular tensors and packed sequences
            is_packed = isinstance(input, PackedSequence)

            # Forward through reservoir layers (fixed weights)
            output, hidden = self.reservoir(input, h_0)
            
            # Unpack if needed and get sequence lengths
            if is_packed:
                output, seq_lengths = pad_packed_sequence(output,
                                                          batch_first=self.batch_first)
            else:
                if self.batch_first:
                    seq_lengths = output.size(0) * [output.size(1)]
                else:
                    seq_lengths = output.size(1) * [output.size(0)]

            # Ensure time-first format for processing
            if self.batch_first:
                output = output.transpose(0, 1)

            # Apply washout: remove initial timesteps to stabilize dynamics
            output, seq_lengths = washout_tensor(output, washout, seq_lengths)

            # Optionally concatenate input features to reservoir output
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

            # Branch based on training method
            if self.readout_training == 'gd' or target is None:
                # Online training or inference: use current readout weights
                with torch.enable_grad():
                    output = self.readout(output)

                    # Zero out padded positions for packed sequences
                    if is_packed:
                        for i in range(output.size(1)):
                            if seq_lengths[i] < output.size(0):
                                output[seq_lengths[i]:, i] = 0

                    # Restore original batch dimension order
                    if self.batch_first:
                        output = output.transpose(0, 1)

                    # Uncomment if you want packed output.
                    # if is_packed:
                    #     output = pack_padded_sequence(output, seq_lengths,
                    #                                   batch_first=self.batch_first)

                    return output, hidden

            else:
                # Offline training: accumulate statistics for ridge regression
                batch_size = output.size(1)

                # Prepare design matrix X with bias column
                X = torch.ones(target.size(0), 1 + output.size(2), device=target.device)
                row = 0
                
                # Fill design matrix based on output aggregation method
                for s in range(batch_size):
                    if self.output_steps == 'all':
                        # Use all timesteps for sequence-to-sequence tasks
                        X[row:row + seq_lengths[s], 1:] = output[:seq_lengths[s], s]
                        row += seq_lengths[s]
                    elif self.output_steps == 'mean':
                        # Use temporal mean for classification
                        X[row, 1:] = torch.mean(output[:seq_lengths[s], s], 0)
                        row += 1
                    elif self.output_steps == 'last':
                        # Use final timestep for classification
                        X[row, 1:] = output[seq_lengths[s] - 1, s]
                        row += 1

                if self.readout_training == 'cholesky':
                    # Accumulate normal equations for Cholesky solve
                    if self.XTX is None:
                        self.XTX = torch.mm(X.t(), X)
                        self.XTy = torch.mm(X.t(), target)
                    else:
                        self.XTX += torch.mm(X.t(), X)
                        self.XTy += torch.mm(X.t(), target)

                elif self.readout_training == 'svd':
                    # Direct SVD solution (single batch only)
                    # Scikit-Learn SVD solver for ridge regression.
                    U, s, V = torch.svd(X)
                    idx = s > 1e-15  # same default value as scipy.linalg.pinv
                    s_nnz = s[idx][:, None]
                    UTy = torch.mm(U.t(), target)
                    d = torch.zeros(s.size(0), 1, device=X.device)
                    d[idx] = s_nnz / (s_nnz ** 2 + self.lambda_reg)
                    d_UT_y = d * UTy
                    W = torch.mm(V, d_UT_y).t()

                    # Set readout parameters directly
                    self.readout.bias = nn.Parameter(W[:, 0])
                    self.readout.weight = nn.Parameter(W[:, 1:])
                    
                elif self.readout_training == 'inv':
                    # Accumulate normal equations for matrix inversion solve
                    self.X = X  # Store for rank checking
                    if self.XTX is None:
                        self.XTX = torch.mm(X.t(), X)
                        self.XTy = torch.mm(X.t(), target)
                    else:
                        self.XTX += torch.mm(X.t(), X)
                        self.XTy += torch.mm(X.t(), target)

                return None, None

    def fit(self):
        """Solve the accumulated linear system for offline readout training.
        
        This method computes the optimal readout weights using the statistics
        accumulated during forward passes. Only applicable for 'cholesky' and 
        'inv' readout training methods.
        
        The method solves the ridge regression problem:
            W* = argmin ||XW - Y||² + λ||W||²
        
        where X contains reservoir states, Y contains targets, and λ is lambda_reg.
        
        Raises:
            RuntimeError: If called before any forward passes or with incompatible
                         readout_training method.
                         
        Note:
            After calling fit(), accumulated statistics are cleared and the model
            is ready for inference or additional training cycles.
        """
        if self.readout_training in {'gd', 'svd'}:
            # Nothing to do: gd uses gradients, svd solves directly
            return

        if self.readout_training == 'cholesky':
            # Solve (X^T X + λI) W = X^T y using Cholesky decomposition
            W = torch.cholesky_solve(self.XTy,
                           self.XTX + self.lambda_reg * torch.eye(
                               self.XTX.size(0), device=self.XTX.device)).t()
            # Clear accumulated statistics
            self.XTX = None
            self.XTy = None

            # Set learned parameters
            self.readout.bias = nn.Parameter(W[:, 0])
            self.readout.weight = nn.Parameter(W[:, 1:])
            
        elif self.readout_training == 'inv':
            # Solve (X^T X + λI) W = X^T y using matrix inversion
            I = (self.lambda_reg * torch.eye(self.XTX.size(0))).to(
                self.XTX.device)
            A = self.XTX + I
            
            # Check matrix rank and use appropriate solver
            X_rank = torch.linalg.matrix_rank(A).item()
            if X_rank == self.X.size(0):
                W = torch.mm(torch.inverse(A), self.XTy).t()
            else:
                # Use pseudoinverse for singular matrices
                W = torch.mm(torch.pinverse(A), self.XTy).t()

            # Set learned parameters and clear statistics
            self.readout.bias = nn.Parameter(W[:, 0])
            self.readout.weight = nn.Parameter(W[:, 1:])
            self.XTX = None
            self.XTy = None

    def reset_parameters(self):
        """Reset all learnable parameters to their initial values.
        
        This method reinitializes:
        - Reservoir weights (input, recurrent, bias) with fresh random values
        - Readout layer weights and biases to default initialization
        - Clears any accumulated training statistics
        
        Useful for:
        - Starting fresh training runs
        - Hyperparameter optimization
        - Ensemble methods requiring different initializations
        """
        self.reservoir.reset_parameters()
        self.readout.reset_parameters()
