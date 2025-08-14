# PyTorch-ESN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+-red.svg)](https://pytorch.org/)

A PyTorch implementation of Echo State Networks (ESNs) with leaky-integrated units for efficient sequence modeling and time series prediction.

## Overview

PyTorch-ESN provides a flexible and efficient implementation of Echo State Networks, a type of recurrent neural network particularly well-suited for:

- **Time series prediction and forecasting**
- **Sequence classification tasks**
- **Temporal pattern recognition**
- **Chaotic system modeling**

### Key Features

- 🚀 **Fast training**: Reservoir states are computed in one forward pass, only readout layer is trained
- 🔧 **Flexible**: Support for both offline (ridge regression) and online (gradient-based) training
- 🏗️ **Deep ESNs**: Multi-layer implementation based on [DeepESN](https://arxiv.org/abs/1712.04323)
- 🎯 **Memory efficient**: Batch processing support for large datasets
- ⚡ **GPU accelerated**: Full PyTorch integration with CUDA support

### Research Background

This implementation was developed as part of the master thesis ["An Empirical Comparison of Recurrent Neural Networks on Sequence Modeling"](https://www.dropbox.com/s/gyt48dcktht7qur/document.pdf?dl=0), supervised by Prof. Alessio Micheli and Dr. Claudio Gallicchio at the University of Pisa.

## Installation

### Prerequisites

- Python 3.11 or higher
- PyTorch 2.8.0 or higher

### Install from source

```bash
git clone https://github.com/stefanonardo/pytorch-esn.git
cd pytorch-esn
pip install -e .
```

## Usage

### Offline Training (Ridge Regression)

Ridge regression training is typically faster and more stable for ESNs, as it directly solves the linear system.

#### Single-batch SVD method

⚠️ **Note**: Mini-batch mode is not supported with SVD method.

```python
from torchesn.nn import ESN
from torchesn.utils import prepare_target

# Prepare target matrix for offline training
flat_target = prepare_target(target, seq_lengths, washout)

# Create model with SVD solver (default)
model = ESN(input_size, hidden_size, output_size)

# Train in one step
model(input, washout, hidden, flat_target)

# Inference
output, hidden = model(input, washout, hidden)
```

#### Batch-wise Cholesky or Matrix Inverse

✅ **Recommended** for large datasets that don't fit in memory.

```python
from torchesn.nn import ESN
from torchesn.utils import prepare_target

model = ESN(input_size, hidden_size, output_size,
           readout_training='cholesky')  # or 'inv'

# Accumulate statistics across batches
for batch in dataloader:
    X_batch, y_batch = batch
    washout_batch = [20] * X_batch.size(1)  # batch washout

    # Accumulate matrices for ridge regression
    model(X_batch, washout_batch, target=y_batch)

# Solve the accumulated system
model.fit()

# Inference on new data
output, hidden = model(test_input, test_washout)
```

### Classification Tasks

For classification problems, specify how to aggregate the temporal outputs:

```python
# Use mean pooling over time
model = ESN(input_size, hidden_size, output_size, output_steps='mean')

# Or use only the last timestep
model = ESN(input_size, hidden_size, output_size, output_steps='last')
```

💡 **Tip**: For more details, see section 4.7 of ["A Practical Guide to Applying Echo State Networks"](http://www.scholarpedia.org/article/Echo_state_network) by Mantas Lukoševičius.

### Online Training (PyTorch Optimizers)

For gradient-based optimization, treat the ESN like any other PyTorch module:

```python
import torch.optim as optim
from torch.nn import MSELoss

model = ESN(input_size, hidden_size, output_size,
           readout_training='gradient')
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = MSELoss()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        output, _ = model(X_batch, washout_batch)
        loss = criterion(output, y_batch)

        loss.backward()
        optimizer.step()
```

## Examples

The `examples/` directory contains complete working examples:

- **`mackey-glass.py`**: Time series prediction on the Mackey-Glass chaotic system
- **`mnist.py`**: MNIST digit classification with batch processing
- Demonstrates memory-efficient processing of large datasets

```bash
# Run the examples
cd examples
python mackey-glass.py
python mnist.py
```

## API Reference

### ESN Class

```python
ESN(input_size, hidden_size, output_size, num_layers=1, bias=True,
    output_steps='all', readout_training='svd', reservoir_bias=True,
    leaky_rate=1.0, spectral_radius=0.9, input_scaling=1.0,
    connectivity=0.1, regularization=1e-8)
```

#### Parameters

- **`input_size`** (int): Number of input features
- **`hidden_size`** (int): Number of neurons in the reservoir
- **`output_size`** (int): Number of output features
- **`num_layers`** (int, default=1): Number of ESN layers for Deep ESN
- **`output_steps`** (str, default='all'): Output aggregation method
  - `'all'`: Return outputs for all timesteps
  - `'last'`: Return only the last timestep
  - `'mean'`: Return the mean over all timesteps
- **`readout_training`** (str, default='svd'): Training method for readout
  - `'svd'`: Single-pass SVD (no mini-batching)
  - `'cholesky'`: Cholesky decomposition (supports batching)
  - `'inv'`: Matrix inversion (supports batching)
  - `'gradient'`: Gradient-based optimization
- **`leaky_rate`** (float, default=1.0): Leaky integration rate (1.0 = no leakage)
- **`spectral_radius`** (float, default=0.9): Spectral radius of reservoir weights
- **`input_scaling`** (float, default=1.0): Scaling factor for input weights
- **`connectivity`** (float, default=0.1): Sparsity of reservoir connections
- **`regularization`** (float, default=1e-8): Ridge regression regularization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/stefanonardo/pytorch-esn.git
cd pytorch-esn
pip install -e .[dev]  # Install with development dependencies
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@misc{pytorch-esn,
  author = {Stefano Nardo},
  title = {PyTorch-ESN: Echo State Networks in PyTorch},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/stefanonardo/pytorch-esn}}
}
```

## References

- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Technical Report 148.
- Gallicchio, C., Micheli, A., & Pedrelli, L. (2017). Deep reservoir computing: A critical experimental analysis. Neurocomputing, 268, 87-99.
- Lukoševičius, M. (2012). A practical guide to applying echo state networks. In Neural networks: Tricks of the trade (pp. 659-686). Springer.
