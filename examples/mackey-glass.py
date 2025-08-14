
import torch.nn
"""
Mackey-Glass Time Series Prediction using Echo State Network (ESN).

This script demonstrates the use of an Echo State Network for predicting the Mackey-Glass
chaotic time series. The Mackey-Glass equation is a well-known benchmark problem in
nonlinear time series prediction and chaos theory.

The script performs the following operations:
1. Loads the Mackey-Glass dataset (mg17.csv) containing input-output pairs
2. Splits the data into training (first 5000 samples) and test sets
3. Creates and trains an ESN model with specified parameters
4. Evaluates the model performance on both training and test data
5. Reports training and test errors along with execution time

Parameters:
    - input_size: Dimension of input data (1 for univariate time series)
    - hidden_size: Number of reservoir neurons (500)
    - output_size: Dimension of output prediction (1)
    - washout: Number of initial samples to discard during training (500)
    - device: Computation device (CPU)
    - dtype: Data type for tensors (torch.double)

The ESN uses a washout period to allow the reservoir dynamics to stabilize before
training begins. The model is trained using the fit() method and evaluated using
Mean Squared Error (MSE) loss.

Expected output:
    - Training error: MSE on training data after washout period
    - Test error: MSE on test data
    - Execution time in seconds

Dataset:
    The mg17.csv file should contain two columns representing the input and target
    values of the Mackey-Glass time series with tau=17.
"""
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
import os

device = torch.device('cpu')
dtype = torch.double
torch.set_default_dtype(dtype)

data_path = os.path.join(os.path.dirname(__file__), 'datasets/mg17.csv')
if dtype == torch.double:
    data = np.loadtxt(data_path, delimiter=',', dtype=np.float64)
elif dtype == torch.float:
    data = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
X_data = np.expand_dims(data[:, [0]], axis=1)
Y_data = np.expand_dims(data[:, [1]], axis=1)
X_data = torch.from_numpy(X_data).to(device)
Y_data = torch.from_numpy(Y_data).to(device)

trX = X_data[:5000]
trY = Y_data[:5000]
tsX = X_data[5000:]
tsY = Y_data[5000:]

washout = [500]
input_size = output_size = 1
hidden_size = 500
loss_fcn = torch.nn.MSELoss()

if __name__ == "__main__":
    start = time.time()

    # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

    model = ESN(input_size, hidden_size, output_size)
    model.to(device)

    model(trX, washout, None, trY_flat)
    model.fit()
    output, hidden = model(trX, washout)
    print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

    # Test
    output, hidden = model(tsX, [0], hidden)
    print("Test error:", loss_fcn(output, tsY).item())
    print("Ended in", time.time() - start, "seconds.")
