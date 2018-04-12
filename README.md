# PyTorch-ESN

PyTorch-ESN is a PyTorch module implementing Echo State Networks with leaky-integrated units. ESN's implementation with more than one layer is based on [DeepESN](https://arxiv.org/abs/1712.04323). The readout layer is trainable by ridge regression or by PyTorch's optimizers.

## Prerequisites

* PyTorch

## Basic Usage

### Offline training (ridge regression)

```python
from torchesn.nn import ESN
from torchesn.utils import prepare_target

# prepare target matrix for offline training
flat_target = prepare_target(target, seq_lengths, washout)

model = ESN(input_size, hidden_size, output_size)

output, hidden = model(input, hidden, washout, flat_target)
```

Then, if you want a prediction on the trained model, do not supply a value for the target.

```
output, hidden = model(input, hidden, washout)
```

### Online training (PyTorch optimizer)

Same as PyTorch.
