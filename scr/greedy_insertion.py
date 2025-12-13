import numpy as np
import torch
from scipy.optimize import minimize
from .utils import stereo, remove_duplicates

def _sample_uniform_sphere_points(N):
    """
    Sample N points uniformly on the unit sphere and project them to the 2D plane.
    
    Args:
        N: Number of points to sample
        
    Returns:
        Z: numpy array of shape (2, N) containing 2D coordinates in the plane
    """
    # Step 1: Sample points uniformly on the unit sphere
    theta = np.random.uniform(0, 2*np.pi, N)
    phi = np.arccos(2 * np.random.uniform(0, 1, N) - 1)  # Important formula for uniform distribution on sphere

    # Step 2: Convert to 3D Cartesian coordinates on the sphere
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Step 3: Get a and b representation (these are uniformly distributed on unit sphere)
    a = np.vstack([x, y])
    b = z.reshape(1, -1)
    
    # Return a.T for weights (N, 2) and b.squeeze() for bias (N,)
    return a.T, b.squeeze()

def insertion(data_train, model, N, alpha):
    """
    Insert N neurons that 
    Args:
        data: tuple of (x, v, dv)
        model: trained model in the last step. 
        N: number of neurons inserted
    return:
        weights: np.ndarray, shape = (N, 2) for PyTorch linear layer
        bias: np.ndarray, shape = (N,) for PyTorch linear layer
    """

    def p(z):
        """Objective used in greedy insertion; z has shape (2,)"""
        z = z.reshape(2, 1)
        a, b = stereo(z)  # (2,1), (1,1)

        # Predict with PyTorch
        X = torch.tensor(X_train, dtype=torch.float64, requires_grad=True)
        model.net.eval()
        pred_v = model.net(X)  # (K,1)
        pred_dv = torch.autograd.grad(pred_v.sum(), X, create_graph=False, retain_graph=False)[0]  # (K,2)

        v_pred = pred_v.detach().cpu().numpy()
        dV_pred = pred_dv.detach().cpu().numpy()

        # Kernel terms
        activation_fn = model.net.activation
        pre_activation = torch.tensor(X_train @ a + b, dtype=torch.float64)
        kernel = activation_fn(pre_activation).detach().cpu().numpy()  # (K,1)
        multi_kernel = np.repeat(power * (kernel ** (power - 1)), 2, axis=1)  # (K,2)

        coeff = (dV_pred - dV_train) * a.T.repeat(K, axis=0)  # (K,2)

        result = (
            (model.loss_weights[0] + model.loss_weights[1]) * np.sum((kernel ** power) * (v_pred - V_train), axis=0)
            + model.loss_weights[1] * np.sum(coeff * multi_kernel)
        )

        return result
        

    # Get the data first (handle dict or tuple; tensors or arrays)
    if isinstance(data_train, dict):
        X_train, V_train, dV_train = data_train["x"], data_train["v"], data_train["dv"]
    else:
        X_train, V_train, dV_train = data_train  # expected: (x, v, dv)

    # If torch tensors, detach to numpy for SciPy and NumPy ops below
    if hasattr(X_train, "detach"):
        X_train = X_train.detach().cpu().numpy()
    if hasattr(V_train, "detach"):
        V_train = V_train.detach().cpu().numpy()
    if hasattr(dV_train, "detach"):
        dV_train = dV_train.detach().cpu().numpy()

    V_train = V_train.reshape(-1, 1)
    K = len(X_train)
    power = model.power
    
    # Step 1: Sample candidate points 
    a, b = _sample_uniform_sphere_points(N) # a has shape (N, 2), b has shape (N,)
    a_sphere = a.T  # Shape (2, N)
    b_sphere = b.reshape(1, -1)  # Shape (1, N)
    Z = a_sphere / (1 + b_sphere)  # Apply inverse stereographic projection to get z values

    # Step 2: Insert the neuron by the algorithm in the paper. 
    result = []
    for i in range(Z.shape[1]):
        z = Z[:, i]  # Get the i-th candidate point
        res = minimize(lambda x: -np.abs(p(x)), x0=z)
        if res.fun < - alpha:
            result.append(res.x)
    
    result = np.array(result)
    result = result.T

    result_a, result_b = stereo(result)
    result_b = result_b.squeeze(axis=0)

    return result_a.T, result_b


    

    







