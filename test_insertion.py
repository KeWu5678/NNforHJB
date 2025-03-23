import numpy as np
import torch
import deepxde as dde
from greedy_insertion import insertion

# Set the backend
import os
os.environ["DDE_BACKEND"] = "pytorch"
dde.config.backend = "pytorch"
print(f"Using backend: {dde.backend.backend_name}")

def gradient(z):
        """z is give as (2, 0)"""
        z = z.reshape(2, 1)
        a, b = stereo(z) # shape = (2, 1), (1, 1)
        kernel = relu(X_train @ a + b)  # shape = (K, 1)
        # print(f"kernel: {kernel}")
        v_pred = model.predict(X_train) # shape = (K, 1)
        
        multi_kernel = np.repeat(p * (kernel ** (p - 1)), 2, axis=1) # shape = (K, 2)
        dV_pred = get_dV_pred(X_train) # shape = (K, 2)
        coeff = (dV_pred - dV_train) * a.T.repeat(K, axis=0) # shape = (K, 2)
        
        return np.sum(kernel ** p * (v_pred - V_train), axis=0) + np.sum(coeff * multi_kernel)