import torch
import torch.nn as nn
import numpy as np


class ShallowNetwork(nn.Module):
    """
    Standalone PyTorch implementation of the SHALLOW network.
    
    Args:
        layer_sizes (list): [input_dim, hidden_dim, output_dim] 
        activation: activation function (torch.nn functional)
        kernel_initializer: weight initialization method
        p (float): power for activation function (default: 2)
        inner_weights (array, optional): pre-defined hidden weights
        inner_bias (array, optional): pre-defined hidden bias
        regularization (tuple, optional): regularization parameters
    """
    
    def __init__(self, layer_sizes, activation, kernel_initializer="xavier_uniform", 
                 p=2, inner_weights=None, inner_bias=None):
        super().__init__()
        
        if len(layer_sizes) != 3:
            raise ValueError("This is not a shallow net! layer_sizes must have 3 elements.")
        
        # Store parameters
        self.p = p
            
        # Use activation function directly
        self.activation = activation
        
        # Create hidden layer
        self.hidden = nn.Linear(layer_sizes[0], layer_sizes[1])
        
        # Initialize or set inner weights/bias
        if inner_weights is None or inner_bias is None:
            # Initialize with zeros (matching original behavior)
            nn.init.zeros_(self.hidden.weight)
            nn.init.zeros_(self.hidden.bias)
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
        if kernel_initializer == "xavier_uniform":
            nn.init.xavier_uniform_(self.output.weight)
        elif kernel_initializer == "zeros":
            nn.init.zeros_(self.output.weight)
        else:
            nn.init.xavier_uniform_(self.output.weight)  # default
        
        # Initialize output bias to zero and freeze it
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
    
    def forward_with_hidden(self, x):
        """Forward pass that also returns hidden activations for SSN optimizer."""
        # Hidden layer transformation
        hidden = torch.nn.functional.linear(x, self.hidden.weight, self.hidden.bias)
        # Apply activation with power
        hidden_activated = self.activation(hidden) ** self.p
        # Output layer
        output = self.output(hidden_activated)
        return output, hidden_activated
    
    def get_hidden_params(self):
        """Return the parameters of the hidden layer."""
        return self.hidden.weight.detach().clone(), self.hidden.bias.detach().clone()
        
    def get_output_params(self):
        """Return the parameters of the output layer."""
        return self.output.weight.detach().clone() 