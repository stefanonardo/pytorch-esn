# PyTorch-ESN

PyTorch-ESN is a PyTorch module implementing an Echo State Network with leaky-integrated units. ESN's implementation with more than one layer is based on [DeepESN](https://arxiv.org/abs/1712.04323). The readout layer is trainable by ridge regression or by PyTorch's optimizers.

## Prerequisites

* PyTorch
* NumPy

## Basic Usage

### Offline training (ridge regression)

```
import torchesn.nn

model = torchesn.nn.ESN(input_size, hidden_size, output_size)

output, hidden = model(input, hidden, target, washout)
```
Then, if you want a prediction, pass None to target parameter.

```
output, hidden = model(input, hidden, None, washout)
```

### Online training (PyTorch optimizer)

Same as PyTorch.