import torch.nn
import numpy as np
from torchesn import nn, utils
import time

device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)

if dtype == torch.double:
    data = np.loadtxt('../../datasets/mg17.csv', delimiter=',', dtype=np.float64)
elif dtype == torch.float:
    data = np.loadtxt('../../datasets/mg17.csv', delimiter=',', dtype=np.float32)
X_data = np.expand_dims(data[:, [0]], axis=1)
Y_data = np.expand_dims(data[:, [1]], axis=1)
X_data = torch.from_numpy(X_data).to(device)
Y_data = torch.from_numpy(Y_data).to(device)

trX = X_data[:5000]
trY = Y_data[:5000]
tsX = X_data[5000:]
tsY = Y_data[5000:]

washout = [500]
hidden_size = 500
loss_fcn = torch.nn.MSELoss()

start = time.time()

# Training
trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

model = nn.ESN(1, hidden_size, 1)
model.to(device)

model(trX, washout, None, trY_flat)
output, hidden = model(trX, washout)
print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

# Test
output, hidden = model(tsX, [0], hidden)
print("Test error:", loss_fcn(output, tsY).item())

