import torch
import torch.nn as nn
import numpy as np


class ShallowNetwork(nn.Module):
    """
    Standalone PyTorch implementation of the SHALLOW network.
    
    Args:
        layer_sizes (list): [input_dim, hidden_dim, output_dim] 
        activation: activation function (torch.nn functional)
        initializer: weight initialization method
        p (float): power for activation function (default: 2)
        inner_weights (array, optional): pre-defined hidden weights
        inner_bias (array, optional): pre-defined hidden bias
        regularization (tuple, optional): regularization parameters
    """
    
    def __init__(
        self, 
        layer_sizes, 
        activation, 
        p=2,
        initializer="xavier_uniform_", 
        inner_weights=None, inner_bias=None, outer_weights=None,
        ):
        super().__init__()

        if len(layer_sizes) != 3:
            raise ValueError("This is not a shallow net! layer_sizes must have 3 elements.")

        # Store parameters
        self.p = p
        self.activation = activation
        # Resolve initializer: allow passing full function name with trailing underscore
        # e.g., "xavier_uniform_" or a callable. Default to xavier_uniform_.
        if isinstance(initializer, str):
            self.initializer = getattr(nn.init, initializer, nn.init.xavier_uniform_)
        else:
            self.initializer = initializer
        
        # Create hidden layer
        self.hidden = nn.Linear(layer_sizes[0], layer_sizes[1])
        
        # Initialize or set inner weights/bias
        if inner_weights is None or inner_bias is None:
            # Initialize hidden weights; bias 
            self.initializer(self.hidden.weight)
            nn.init.uniform_(self.hidden.bias, -0.1, 0.1)
        else:
            # Delete existing parameters and set custom ones
            del self.hidden.weight
            del self.hidden.bias
            
            # Convert to tensors if needed
            if isinstance(inner_weights, np.ndarray):
                inner_weights = torch.tensor(inner_weights, dtype=torch.float64)
            if isinstance(inner_bias, np.ndarray):
                inner_bias = torch.tensor(inner_bias, dtype=torch.float64)
            
            # Assign new weights (these become trainable parameters)
            self.hidden.weight = torch.nn.Parameter(inner_weights.clone())
            self.hidden.bias = torch.nn.Parameter(inner_bias.clone())
        
        # Create output layer
        self.output = nn.Linear(layer_sizes[1], layer_sizes[2])
        
        # Initialize output weights
        if outer_weights is None:
            self.initializer(self.output.weight)
        else:
            if isinstance(outer_weights, np.ndarray):
                outer_weights = torch.tensor(outer_weights, dtype=torch.float64)
            with torch.no_grad():
                # Copy provided weights into existing parameter
                self.output.weight.copy_(outer_weights)

        # the output bias is set to zero and not trainable         
        nn.init.zeros_(self.output.bias)
        self.output.bias.requires_grad = False
        
        # Ensure layers use double precision to match input data
        self.hidden.double()
        self.output.double()
    
    def forward(self, x):
        # Hidden layer transformation
        x = torch.nn.functional.linear(x, self.hidden.weight, self.hidden.bias)
        # Apply activation with power
        x = self.activation(x) ** self.p
        # Output layer
        x = self.output(x)
        return x
    
    def forward_network_matrix(self, x):
        """Forward pass that also returns hidden activations for SSN optimizer."""
        # Hidden layer transformation
        hidden = torch.nn.functional.linear(x, self.hidden.weight, self.hidden.bias)
        # Apply activation with power
        network_matrix = self.activation(hidden) ** self.p
        return network_matrix

    def forward_gradient_kernel(self, x):
        """Compute the gradient kernel dS/dx, stacked over input dimensions.

        For the gradient loss, the prediction is pred_dv_{k,j} = sum_n u_n * dS_{k,n}/dx_{k,j}.
        Each input dimension j defines a kernel S^{(j)} of shape (N, n), and
        stacking them gives the full gradient kernel of shape (N*d, n).

        Uses chain rule: dS_{k,n}/dx_{k,j} = (dS/dz)_{k,n} * W_{n,j}

        Args:
            x: Input tensor of shape (N, d)

        Returns:
            S_grad: Stacked gradient kernel of shape (N*d, n)
        """
        z = torch.nn.functional.linear(x, self.hidden.weight, self.hidden.bias)
        z = z.detach().requires_grad_(True)
        with torch.enable_grad():
            S = self.activation(z) ** self.p
            dS_dz = torch.autograd.grad(
                S, z,
                grad_outputs=torch.ones_like(S),
                create_graph=False
            )[0]  # (N, n)
        dS_dz = dS_dz.detach()
        W = self.hidden.weight.detach()  # (n, d)

        # dS/dx_{k,n,j} = dS/dz_{k,n} * W_{n,j}  →  shape (N, n, d)
        dS_dx = dS_dz.unsqueeze(2) * W.unsqueeze(0)  # (N, n, d)

        # Stack over input dimensions: (N, n, d) → (N, d, n) → (N*d, n)
        S_grad = dS_dx.permute(0, 2, 1).reshape(-1, W.shape[0])
        return S_grad