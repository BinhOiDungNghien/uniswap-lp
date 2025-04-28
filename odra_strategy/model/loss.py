"""
Loss functions for ODRA strategy
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class CARAUtilityLoss(nn.Module):
    """CARA (Constant Absolute Risk Aversion) utility loss."""
    
    def __init__(self, risk_aversion: float = 0.01):
        """Initialize loss.
        
        Args:
            risk_aversion: Risk aversion parameter
        """
        super().__init__()
        self.risk_aversion = risk_aversion
        
    def forward(self, wealth: torch.Tensor) -> torch.Tensor:
        """Compute CARA utility loss.
        
        Args:
            wealth: Wealth tensor
            
        Returns:
            Loss value (negative utility)
        """
        if self.risk_aversion == 0:
            return -torch.mean(wealth)
            
        utility = (1 - torch.exp(-self.risk_aversion * wealth)) / self.risk_aversion
        return -torch.mean(utility)  # Negative because we want to maximize utility

class EntropyRegularizedLoss(nn.Module):
    """Entropy regularized CARA utility loss."""
    
    def __init__(self,
                 risk_aversion: float = 0.01,
                 entropy_coef: float = 0.01):
        """Initialize loss.
        
        Args:
            risk_aversion: Risk aversion parameter
            entropy_coef: Entropy regularization coefficient
        """
        super().__init__()
        self.risk_aversion = risk_aversion
        self.entropy_coef = entropy_coef
        self.cara_loss = CARAUtilityLoss(risk_aversion)
        
    def compute_entropy(self, allocations: torch.Tensor) -> torch.Tensor:
        """Compute entropy of allocation distribution.
        
        Args:
            allocations: Allocation probabilities
            
        Returns:
            Entropy value
        """
        # Add small epsilon to prevent log(0)
        eps = 1e-10
        entropy = -torch.sum(
            allocations * torch.log(allocations + eps),
            dim=-1
        )
        return torch.mean(entropy)
        
    def forward(self,
                wealth: torch.Tensor,
                allocations: torch.Tensor) -> torch.Tensor:
        """Compute entropy regularized loss.
        
        Args:
            wealth: Wealth tensor
            allocations: Allocation probabilities
            
        Returns:
            Loss value
        """
        # Compute CARA utility loss
        utility_loss = self.cara_loss(wealth)
        
        # Compute entropy regularization
        entropy = self.compute_entropy(allocations)
        
        # Combine losses
        total_loss = utility_loss - self.entropy_coef * entropy
        
        return total_loss

class TransactionCostLoss(nn.Module):
    """Transaction cost aware CARA utility loss."""
    
    def __init__(self,
                 risk_aversion: float = 0.01,
                 tx_cost_rate: float = 0.001):
        """Initialize loss.
        
        Args:
            risk_aversion: Risk aversion parameter
            tx_cost_rate: Transaction cost rate
        """
        super().__init__()
        self.risk_aversion = risk_aversion
        self.tx_cost_rate = tx_cost_rate
        self.cara_loss = CARAUtilityLoss(risk_aversion)
        
    def compute_rebalance_cost(self,
                             old_allocations: torch.Tensor,
                             new_allocations: torch.Tensor) -> torch.Tensor:
        """Compute rebalancing transaction costs.
        
        Args:
            old_allocations: Previous allocation weights
            new_allocations: New allocation weights
            
        Returns:
            Transaction cost
        """
        # Compute absolute changes in allocations
        changes = torch.abs(new_allocations - old_allocations)
        
        # Apply transaction cost rate
        costs = changes * self.tx_cost_rate
        
        return torch.sum(costs, dim=-1).mean()
        
    def forward(self,
                wealth: torch.Tensor,
                old_allocations: torch.Tensor,
                new_allocations: torch.Tensor) -> torch.Tensor:
        """Compute transaction cost aware loss.
        
        Args:
            wealth: Wealth tensor
            old_allocations: Previous allocation weights
            new_allocations: New allocation weights
            
        Returns:
            Loss value
        """
        # Compute CARA utility loss
        utility_loss = self.cara_loss(wealth)
        
        # Compute transaction costs
        tx_costs = self.compute_rebalance_cost(old_allocations, new_allocations)
        
        # Combine losses
        total_loss = utility_loss + tx_costs
        
        return total_loss 