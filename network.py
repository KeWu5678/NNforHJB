#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import deepxde as dde
from scipy.spatial import KDTree


def network(data, power, regularization, inner_weights = None, inner_bias = None):
    """
    args:
        inner_weights: (N, 2) - matrix with N rows (number of neurons) and 2 columns (input features)
        inner_bias: (N,) - vector with N bias values, one per neuron
        path: tuple of (x, v, dv)
    return:
        model: the trained model
    """
    

    # Get the raw data
    ob_x, ob_v, ob_dv = data["x"], data["v"], data["dv"]

    # Customized to dimension
    def VdV(x, y, ex):
        y = y[:, 0:1]
        v1 = ex[:, 0:1]
        v2 = ex[:, 1:2]
        dy_dx1 = dde.grad.jacobian(y, x, i=0, j = 0)
        dy_dx2 = dde.grad.jacobian(y, x, i=0, j = 1)
        return [
            v1 - dy_dx1,
            v2 - dy_dx2,
        ]

    geom = dde.geometry.Rectangle([0, 0], [3, 3])

    def aux_function(x):
        """Return the auxiliary variables (dV/dx values) for the given points x."""
        # Print diagnostic information about the input
        # print(f"aux_function called with {len(x)} points")
        # print(f"First point shape: {x[0].shape}, dtype: {x[0].dtype}")
        # print(f"ob_x shape: {ob_x.shape}, dtype: {ob_x.dtype}")
        
        # Check if all points in x are in ob_x
        if not hasattr(aux_function, 'kdtree'):
            aux_function.kdtree = KDTree(ob_x)
        # Find indices of closest points
        distances, indices = aux_function.kdtree.query(x, k=1)
        
        # Print information about matches
        # if np.max(distances) > 0:
        #     print(f"Max distance: {np.max(distances)}, mean distance: {np.mean(distances)}")
        return ob_dv[indices]

    def value_function(x):
        """Return the value function V at points x."""
        # Use KDTree for efficient and robust matching
        if not hasattr(value_function, 'kdtree'):
            value_function.kdtree = KDTree(ob_x)
        
        # Find indices of closest points
        _, indices = value_function.kdtree.query(x, k=1)
        
        # Return corresponding V values
        return ob_v[indices].reshape(-1, 1)  # Make sure it's a column vector

    data = dde.data.PDE(
        geom,
        VdV,
        [],
        num_domain=0,
        num_boundary=0,
        anchors=ob_x,
        auxiliary_var_function=aux_function,
        solution=value_function
    )
    
    if inner_weights is None:
        # if no weights are given, use default 2 neurons
        n = 0
    else:
        # Check and print shapes for debugging
        print(f"DEBUG - inner_weights shape: {inner_weights.shape}")
        if inner_bias is not None:
            print(f"DEBUG - inner_bias shape: {inner_bias.shape}")
        
        # Number of neurons is the first dimension for PyTorch
        n = inner_weights.shape[0]
        print(f"Creating network with {n} neurons")
    
    # Layer sizes: input dimension, hidden layer size, output dimension
    net = dde.nn.SHALLOW(
        [2] + [n] + [1], "tanh", "Glorot normal", p = power, inner_weights = inner_weights, 
        inner_bias = inner_bias, regularization = regularization
        )
    model = dde.Model(data, net)
    model.compile("adam", lr=0.005, loss="mse", loss_weights=[1.0, 1.0])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_dir, "train_history")
    os.makedirs(model_save_path, exist_ok=True)
    print(f"Training model, saving to {model_save_path}")
    losshistory, train_state = model.train(iterations=20000, display_every=1000, model_save_path=model_save_path)

    # Detach the tensors to remove them from the computation graph before returning
    weight, bias = model.net.get_hidden_params()
    return model, weight.numpy(), bias.numpy()


