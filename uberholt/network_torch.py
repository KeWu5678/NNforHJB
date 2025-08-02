import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class HJBDataset(Dataset):
    """Custom dataset that includes input data, target values, and target gradients"""
    def __init__(self, x, v, dv):
        """
        Args:
            x: Input features (N, input_dim)
            v: Target values (N, 1)
            dv: Target gradients (N, input_dim)
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32).reshape(-1, 1)
        self.dv = torch.tensor(dv, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.v[idx], self.dv[idx]


class ShallowNetwork(nn.Module):
    """
    Shallow neural network with a single hidden layer and inner weights and biases given by the user
    The output layer is initialized with Glorot (Xavier) normal
    """
    def __init__(self, input_dim, n_neurons, activation='relu', p=1.0, inner_weights=None, inner_bias=None):
        """
        Args:
            input_dim: Dimension of input features
            n_neurons: Number of neurons in the hidden layer
            activation: Activation function ('tanh' or 'relu')
            p: Power parameter for activation
            inner_weights: Optional pre-initialized weights for hidden layer (n_neurons, input_dim)
            inner_bias: Optional pre-initialized bias for hidden layer (n_neurons)
        """
        super(ShallowNetwork, self).__init__()
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.p = p
        
        # Hidden layer
        self.hidden = nn.Linear(input_dim, n_neurons)
        
        # Initialize with provided weights and biases if given
        if inner_weights is not None and n_neurons > 0:
            with torch.no_grad():
                self.hidden.weight.copy_(torch.tensor(inner_weights, dtype=torch.float32))
                if inner_bias is not None:
                    self.hidden.bias.copy_(torch.tensor(inner_bias, dtype=torch.float32))
        
        # Output layer
        self.output = nn.Linear(n_neurons, 1)
        
        # Activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Initialize with Glorot (Xavier) normal
        if inner_weights is None:
            nn.init.xavier_normal_(self.hidden.weight)
            nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        if self.n_neurons > 0:
            h = self.hidden(x)
            h = self.activation(h) ** self.p
            out = self.output(h)
        else:
            # If no hidden neurons, return zeros (or other default value)
            out = torch.zeros(x.shape[0], 1, device=x.device)
        return out
    
    def get_hidden_params(self):
        """Return the weights and bias of the hidden layer"""
        return self.hidden.weight.detach(), self.hidden.bias.detach()


class ShallowNetworkModel:
    """Model class that handles training and prediction"""
    def __init__(self, input_dim, n_neurons=0, activation='tanh', p=1.0, 
                 regularization=None, inner_weights=None, inner_bias=None):
        """
        Args:
            input_dim: Dimension of input features
            n_neurons: Number of neurons in hidden layer
            activation: Activation function ('tanh' or 'relu')
            p: Power parameter for activation
            regularization: Tuple of ('phi', gamma, alpha) for phi regularization
            inner_weights: Optional pre-initialized weights
            inner_bias: Optional pre-initialized bias
        """
        self.regularizer = regularization
        self.net = ShallowNetwork(input_dim, n_neurons, activation, p, inner_weights, inner_bias)
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        
    def compile(self, optimizer='adam', lr=0.001, weight_decay=0):
        """Configure the model for training"""
        if optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    def gradient(self, x):
        """Compute gradients of the network output with respect to the input"""
        # Handle case where network has zero neurons
        if self.net.n_neurons == 0:
            return torch.zeros(x.shape[0], x.shape[1], device=self.device)
            
        x = x.clone().detach().requires_grad_(True)
        y = self.net(x)
        
        gradients = []
        for i in range(y.shape[0]):
            self.net.zero_grad()
            
            # Check if y[i] has a grad_fn (gradient function) before calling backward
            if y[i].grad_fn is not None:
                y[i].backward(retain_graph=True)
                
                # Check if x.grad exists (could be None if no gradient was computed)
                if x.grad is not None:
                    gradients.append(x.grad[i].clone().detach())
                else:
                    gradients.append(torch.zeros_like(x[i]))
                
                # Zero gradients for next iteration
                if x.grad is not None:
                    x.grad.zero_()
            else:
                # If no gradient function, append zeros
                gradients.append(torch.zeros_like(x[i]))
        
        return torch.stack(gradients)
        
    def train(self, x, v, dv, batch_size=32, epochs=1000, verbose=True, 
              validation_split=0.0, model_save_path=None, display_every=500):
        """
        Train the model
        
        Args:
            x: Input features
            v: Target values
            dv: Target gradients
            batch_size: Batch size for training
            epochs: Number of training epochs
            verbose: Whether to print progress
            validation_split: Fraction of data to use for validation
            model_save_path: Directory to save model checkpoints
            display_every: Print progress every N epochs
        """
        dataset = HJBDataset(x, v, dv)
        
        # Split data for validation if needed
        if validation_split > 0:
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            val_loader = None
        
        # History for tracking loss
        history = {'train_loss': [], 'val_loss': [] if validation_split > 0 else None}
        
        # Create save directory if needed
        if model_save_path:
            os.makedirs(model_save_path, exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            self.net.train()
            train_loss = 0
            train_value_loss = 0
            train_grad_loss = 0
            
            for x_batch, v_batch, dv_batch in train_loader:
                x_batch = x_batch.to(self.device)
                v_batch = v_batch.to(self.device)
                dv_batch = dv_batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                v_pred = self.net(x_batch)
                
                # Compute gradients with respect to input
                dv_pred = self.gradient(x_batch)
                
                # Value loss (MSE)
                epsilon = 1e-10  # Small value to prevent division by zero
                
                # Relative value loss: divide by squared target values
                # squared_v_batch = v_batch**2 + epsilon
                value_loss = torch.mean((v_pred - v_batch)**2)
                
                # Gradient loss (MSE for each component)
                # squared_dv_batch = torch.sum(dv_batch**2, dim=1, keepdim=True) + epsilon
                grad_loss = torch.mean((dv_pred - dv_batch)**2)
                
                # Total loss (equal weighting for simplicity)
                total_loss = value_loss + grad_loss
                
                # Add regularization if specified
                if self.regularizer is not None:
                    if self.regularizer[0] == "phi":
                        gamma = self.regularizer[1]
                        alpha = self.regularizer[2]
                        phi_penalty = 0
                        for param in self.net.output.parameters():
                            phi_penalty += torch.sum( 1/2 * ( param + 1/gamma * torch.log(1 + gamma * torch.abs(param))))
                        total_loss += alpha * phi_penalty
                
                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()
                
                train_loss += total_loss.item()
                train_value_loss += value_loss.item()
                train_grad_loss += grad_loss.item()
            
            # Calculate average losses
            train_loss /= len(train_loader)
            train_value_loss /= len(train_loader)
            train_grad_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation if needed
            if val_loader:
                self.net.eval()
                val_loss = 0
                val_value_loss = 0
                val_grad_loss = 0
                
                with torch.no_grad():
                    for x_batch, v_batch, dv_batch in val_loader:
                        x_batch = x_batch.to(self.device)
                        v_batch = v_batch.to(self.device)
                        dv_batch = dv_batch.to(self.device)
                        
                        v_pred = self.net(x_batch)
                        dv_pred = self.gradient(x_batch)
                        
                        value_loss = torch.mean((v_pred - v_batch)**2 / (v_batch**2 + 1e-10))
                        grad_loss = torch.mean((dv_pred - dv_batch)**2 / (torch.sum(dv_batch**2, dim=1, keepdim=True) + 1e-10))
                        
                        total_loss = value_loss + grad_loss
                        val_loss += total_loss.item()
                        val_value_loss += value_loss.item()
                        val_grad_loss += grad_loss.item()
                
                val_loss /= len(val_loader)
                val_value_loss /= len(val_loader)
                val_grad_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
            
            # Display progress
            if verbose and epoch % display_every == 0:
                val_str = f", val_loss: {val_loss:.4e}" if val_loader else ""
                print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4e} (value: {train_value_loss:.4e}, grad: {train_grad_loss:.4e}){val_str}")
            
            # Save model checkpoint
            if model_save_path and epoch % display_every == 0:
                checkpoint_path = os.path.join(model_save_path, f"model_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': train_loss,
                }, checkpoint_path)
        
        return history
    
    def predict(self, x, operator=None):
        """
        Make predictions with the model
        
        Args:
            x: Input features (numpy array)
            operator: Optional operator to apply to the output
        
        Returns:
            Predictions as numpy array
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.net.eval()
        
        with torch.no_grad():
            if operator == 'gradient':
                # Return gradients
                return self.gradient(x_tensor).cpu().numpy()
            else:
                # Return values
                return self.net(x_tensor).cpu().numpy()
    
    def get_weights(self):
        """Get all model weights"""
        return [param.detach().cpu().numpy() for param in self.net.parameters()]


def network(data, power, regularization = None, inner_weights=None, inner_bias=None):
    """
    Create and train a ShallowNetworkModel
    
    Args:
        data: Dictionary containing 'x', 'v', 'dv' keys
        power: Power parameter for activation function
        regularization: Tuple of ('phi', gamma, alpha) for phi regularization
        inner_weights: Optional pre-initialized weights
        inner_bias: Optional pre-initialized bias
    
    Returns:
        model: Trained model
        weight: Hidden layer weights
        bias: Hidden layer bias
    """
    # Get the data
    x, v, dv = data["x"], data["v"], data["dv"]
    
    # Determine number of neurons
    n_neurons = 0 if inner_weights is None else inner_weights.shape[0]
    
    # Create model
    model = ShallowNetworkModel(
        input_dim=x.shape[1], 
        n_neurons=n_neurons,
        activation='tanh',
        p=power,
        regularization=regularization,
        inner_weights=inner_weights,
        inner_bias=inner_bias
    )
    
    # Compile model
    model.compile(optimizer='adam', lr=0.001)
    
    # Train model
    epochs = 200 if n_neurons == 0 else 1000  # Use fewer epochs for fine-tuning
    
    model.train(
        x=x, 
        v=v.reshape(-1, 1) if len(v.shape) == 1 else v,
        dv=dv,
        batch_size=32,
        epochs=epochs,
        display_every=1000 if epochs > 1000 else 100,
        model_save_path="train_history"
    )
    
    # Get weights and biases from the trained model
    weight, bias = model.net.get_hidden_params()
    return model, weight.cpu().numpy(), bias.cpu().numpy()


if __name__ == "__main__":
    data = np.load("data_result/VDP_beta_0.1_grid_30x30.npy")
    weights = np.random.randn(1000, 2)
    bias = np.random.randn(1000)
    regularization = ('phi', 0.01, 0.5)
    model, weight, bias = network(data, 2.0, inner_weights=weights, inner_bias=bias)