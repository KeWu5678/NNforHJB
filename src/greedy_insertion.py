import numpy as np
import torch
from scipy.optimize import minimize
from .utils import remove_duplicates

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

def insertion(data, model, N, alpha):
    """
    Insert N neurons
    Args:
        data: tuple of (x, v, dv)
        model: trained model in the last step. 
        N: number of neurons inserted
    return:
        weights: np.ndarray, shape = (N, 2) for PyTorch linear layer
        bias: np.ndarray, shape = (N,) for PyTorch linear layer
    """

    def stereo(z):
        """
        The function transform a point in Rd in Sd:
        Input:
        z:  (2, n) np.array
        Return:  
        shape: (2, n), (1, n)
        """ 
        #A = (2 * Z) / (1 + np.sum(Z**2, axis=1)).reshape(-1, 1) #shape = (N, 2)
        #B = (1 - np.sum(Z**2, axis=1)).reshape(-1, 1) / (1 + np.sum(Z**2, axis=1)).reshape(-1, 1) # shape = (N, 1)
        if z.shape[0] != 2:
            raise ValueError("z must be a (2, n) np.array")
        denominator = (1 + np.sum(z**2, axis=0)).reshape(1, -1)
        return [
            (2 * z) / denominator, 
            (1 - np.sum(z**2, axis=0)) / denominator
            ]

    def gradient_op(inputs, outputs):
        """Custom operator to compute gradient of model output with respect to input"""
        # Sum over batch dimension if needed (for scalar output per example)
        # You can modify this if you need gradients for specific output components
        return torch.autograd.grad(
            outputs.sum(), inputs, create_graph=True, retain_graph=True
        )[0]

    def get_dV_pred(x):
        return model.predict(x, operator=gradient_op)
    
    def get_numpy_activation(activation_fn, x):
        """Convert PyTorch activation function to work with NumPy arrays
        
        Args:
            activation_fn: The activation function from model.net.activation
            x: NumPy array input
            
        Returns:
            NumPy array after applying the activation function
        """
        # Convert numpy to tensor, apply activation, then convert back to numpy
        x_tensor = torch.as_tensor(x)
        return activation_fn(x_tensor).detach().cpu().numpy()
    
    def gradient(z):
        """z is give as (2, 0)"""
        z = z.reshape(2, 1)
        a, b = stereo(z) # shape = (2, 1), (1, 1)
        
        # Use the model's activation function directly
        activation_fn = model.net.activation
        kernel = get_numpy_activation(activation_fn, X_train @ a + b)  # shape = (K, 1)
        
        v_pred = model.predict(X_train) # shape = (K, 1)
        
        multi_kernel = np.repeat(p * (kernel ** (p - 1)), 2, axis=1) # shape = (K, 2)
        dV_pred = get_dV_pred(X_train) # shape = (K, 2)
        coeff = (dV_pred - dV_train) * a.T.repeat(K, axis=0) # shape = (K, 2)
        if model.loss_weights[1] != 0:
            return np.sum(kernel ** p * (v_pred - V_train), axis=0) + np.sum(coeff * multi_kernel)
        else:
            return np.sum(kernel ** p * (v_pred - V_train), axis=0)

    # Get the data first
    X_train, V_train, dV_train = data["x"], data["v"], data["dv"]
    V_train = V_train.reshape(-1, 1)
    K = len(X_train)
    p = model.net.p
    
    # Sample candidate points using the new method
    a, b = _sample_uniform_sphere_points(N)
    # a has shape (N, 2), b has shape (N,)
    # Convert back to the format needed for the stereo function
    a_sphere = a.T  # Shape (2, N)
    b_sphere = b.reshape(1, -1)  # Shape (1, N)
    Z = a_sphere / (1 + b_sphere)  # Apply inverse stereographic projection to get z values

    # Step 2: Insert the neuron by the algorithm in the paper. 
    result = []
    for i in range(Z.shape[1]):
        z = Z[:, i]  # Get the i-th candidate point
        res = minimize(lambda x: -np.abs(gradient(x)), x0=z)
        if res.fun < - alpha:
            result.append(res.x)
    
    # Transfer the dimension
    result = np.array(result)
    result = result.T
    # result_z_removed = remove_duplicates(result_z_raw, tolerance=1e-20)

    result_a, result_b = stereo(result)
    result_b = result_b.squeeze(axis=0)

    # if result_a.T.shape[1] != 2:
    #     print(f"Warning: Unexpected shape for result_a.T: {result_a.T.shape}")
    #     print(f"result.shape.raw: {result.shape}")
    #     print(f"result.shape.removed: {result.shape}")
    #     print(f"result_a.shape: {result_a.shape}")
    #     print(f"result_b.shape: {result_b.shape}")
    # if len(result_b.shape) != 1:
    #     print(f"Warning: Unexpected shape for result_b: {result_b.shape}")
    #     print(f"result_final: {result}")
    #     print(f"result_b: {result_b}")
    return result_a.T, result_b


    

    







