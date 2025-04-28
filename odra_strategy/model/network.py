"""
Neural Network Architecture for ODRA Strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class ODRANetwork(nn.Module):
    """Neural network for ODRA strategy."""
    
    def __init__(self,
                 input_dim: int = 5,  # t/T, ewma_volume, pool_price, center_bucket, wealth
                 hidden_dims: List[int] = [16] * 5,  # 5 hidden layers with 16 units each
                 tau: int = 3,
                 dropout: float = 0.1):
        """Initialize network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            tau: Rebalancing threshold (determines output dimension)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.tau = tau
        self.output_dim = 2 * tau + 2  # Buckets + outside pool positions
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, 2*tau + 2) with allocation probabilities
        """
        # Pass through network
        logits = self.network(x)
        
        # Apply softmax to get allocation probabilities
        allocations = F.softmax(logits, dim=-1)
        
        return allocations
        
    def compute_utility(self, 
                       wealth: torch.Tensor,
                       risk_aversion: float = 0.01) -> torch.Tensor:
        """Compute CARA utility.
        
        Args:
            wealth: Tensor of wealth values
            risk_aversion: Risk aversion parameter
            
        Returns:
            Utility values
        """
        if risk_aversion == 0:
            return wealth
        return (1 - torch.exp(-risk_aversion * wealth)) / risk_aversion 