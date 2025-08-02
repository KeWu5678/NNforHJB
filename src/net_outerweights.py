import torch
import torch.nn as nn
import numpy as np


class ShallowOuterWeightsNetwork(nn.Module):
    """
    Standalone PyTorch implementation of the SHALLOW network with frozen inner weights.
    Only outer weights are trainable.
    
    Args:
        layer_sizes (list): [input_dim, hidden_dim, output_dim] 
        activation: activation function (torch.nn functional)
        kernel_initializer: weight initialization method
        p (float): power for activation function (default: 2)
        inner_weights (array, optional): pre-defined hidden weights (frozen)
        inner_bias (array, optional): pre-defined hidden bias (frozen)
        outer_weights (array, optional): pre-defined output weights
    """
    
    def __init__(self, layer_sizes, activation, p, 
                 inner_weights, inner_bias, outer_weights = None,
                 ):
        super().__init__()
        
        if len(layer_sizes) != 3:
            raise ValueError("This is not a shallow net! layer_sizes must have 3 elements.")
        
        # Store parameters
        self.p = p  # activation power; nothing else is network-specific
            
        # Use activation function directly
        self.activation = activation
        
        # Create hidden layer
        self.hidden = nn.Linear(layer_sizes[0], layer_sizes[1])
        
        if isinstance(inner_weights, np.ndarray):
            inner_weights = torch.tensor(inner_weights, dtype=torch.float64)
        if isinstance(inner_bias, np.ndarray):
            inner_bias = torch.tensor(inner_bias, dtype=torch.float64)
            
        # Set the weights and freeze them
        with torch.no_grad():
            self.hidden.weight.copy_(inner_weights)
            self.hidden.bias.copy_(inner_bias)
        
        # Freeze inner weights (no gradients)
        self.hidden.weight.requires_grad = False
        self.hidden.bias.requires_grad = False
        
        # Create output layer
        self.output = nn.Linear(layer_sizes[1], layer_sizes[2])
        
        # Initialize output weights (trainable)
        # Convert to tensors if needed
        if outer_weights is None:
            # Initialize outer weights if not provided
            nn.init.xavier_normal_(self.output.weight)
        else:
            # Ensure outer_weights is a tensor in double precision
            if isinstance(outer_weights, np.ndarray):
                outer_weights = torch.tensor(outer_weights, dtype=torch.float64)
            with torch.no_grad():
                self.output.weight.copy_(outer_weights)
        # Initialize output bias to zero and freeze it
        nn.init.zeros_(self.output.bias)
        self.output.bias.requires_grad = False
        
        # Ensure layers use double precision to match input data
        self.hidden.double()
        self.output.double()
    
    def forward(self, x):
        # Hidden layer transformation
        x = self.hidden(x)
        # Apply activation with power
        x = self.activation(x) ** self.p
        # Output layer
        x = self.output(x)
        return x
    
    def forward_with_hidden(self, x):
        """Forward pass that also returns hidden activations for SSN optimizer."""
        # Hidden layer transformation
        hidden = self.hidden(x)
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