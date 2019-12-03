# PyTorch-ESN

PyTorch-ESN is a PyTorch module, written in Python, implementing Echo State Networks with leaky-integrated units. ESN's implementation with more than one layer is based on [DeepESN](https://arxiv.org/abs/1712.04323). The readout is trainable by ridge regression or by PyTorch's optimizers.

Its development started under my master thesis titled "An Empirical Comparison of Recurrent Neural Networks on Sequence Modeling", which was supervised by Prof. Alessio Micheli and Dr. Claudio Gallicchio at the University of Pisa.

## Prerequisites

* PyTorch

## Basic Usage

### Offline training (ridge regression)

#### SVD
Mini-batch mode is not allowed with this method.

```python
from torchesn.nn import ESN
from torchesn.utils import prepare_target

# prepare target matrix for offline training
flat_target = prepare_target(target, seq_lengths, washout)

model = ESN(input_size, hidden_size, output_size)

# train
model(input, washout, hidden, flat_target)

# inference
output, hidden = model(input, washout, hidden)
```

#### Cholesky or inverse
```python
from torchesn.nn import ESN
from torchesn.utils import prepare_target

# prepare target matrix for offline training
flat_target = prepare_target(target, seq_lengths, washout)

model = ESN(input_size, hidden_size, output_size, readout_training='cholesky')

# accumulate matrices for ridge regression
for batch in batch_iter:
    model(batch, washout[batch], hidden, flat_target)

# train
model.fit()

# inference
output, hidden = model(input, washout, hidden)
```

#### Classification tasks
For classification, just use one of the previous methods and pass 'mean' or
'last' to ```output_steps``` argument.

```python
model = ESN(input_size, hidden_size, output_size, output_steps='mean')
```

For more information see docstrings or section 4.7 of "A Practical Guide to Applying
Echo State Networks" by Mantas Lukoševičius.

### Online training (PyTorch optimizer)

Same as PyTorch.
