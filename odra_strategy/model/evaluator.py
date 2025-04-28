"""
Evaluator module for ODRA strategy
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .network import ODRANetwork
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

class ODRAEvaluator:
    """Evaluator for ODRA strategy."""
    
    def __init__(self,
                 network: ODRANetwork,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize evaluator.
        
        Args:
            network: ODRA neural network
            config: Evaluation configuration
            device: Device to evaluate on
        """
        self.network = network.to(device)
        self.config = config
        self.device = device
        
        # Evaluation parameters
        self.risk_aversion = config.get('risk_aversion', 0.01)
        self.output_dir = Path(config.get('output_dir', 'outputs/evaluation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_episode(self,
                        features: np.ndarray,
                        wealth: np.ndarray) -> Dict[str, float]:
        """Evaluate model on single episode.
        
        Args:
            features: Input features
            wealth: Wealth values
            
        Returns:
            Dictionary of metrics
        """
        self.network.eval()
        
        # Convert to tensors
        features = torch.FloatTensor(features).to(self.device)
        wealth = torch.FloatTensor(wealth).to(self.device)
        
        with torch.no_grad():
            # Get allocations
            allocations = self.network(features)
            
            # Compute metrics
            utility = self.network.compute_utility(wealth, self.risk_aversion)
            mean_utility = torch.mean(utility).item()
            
            # Compute allocation entropy
            eps = 1e-10
            entropy = -torch.sum(
                allocations * torch.log(allocations + eps),
                dim=-1
            ).mean().item()
            
            # Compute wealth metrics
            final_wealth = wealth[-1].item()
            wealth_mean = torch.mean(wealth).item()
            wealth_std = torch.std(wealth).item()
            
            # Compute allocation metrics
            mean_allocation = torch.mean(allocations, dim=0).cpu().numpy()
            allocation_std = torch.std(allocations, dim=0).cpu().numpy()
            
        return {
            'mean_utility': mean_utility,
            'entropy': entropy,
            'final_wealth': final_wealth,
            'wealth_mean': wealth_mean,
            'wealth_std': wealth_std,
            'mean_allocation': mean_allocation,
            'allocation_std': allocation_std
        }
        
    def evaluate_multiple_episodes(self,
                                 episodes: List[Tuple[np.ndarray, np.ndarray]],
                                 save_plots: bool = True) -> pd.DataFrame:
        """Evaluate model on multiple episodes.
        
        Args:
            episodes: List of (features, wealth) tuples
            save_plots: Whether to save evaluation plots
            
        Returns:
            DataFrame with evaluation metrics
        """
        results = []
        for i, (features, wealth) in enumerate(episodes):
            metrics = self.evaluate_episode(features, wealth)
            metrics['episode'] = i
            results.append(metrics)
            
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        if save_plots:
            self._plot_evaluation_results(df)
            
        return df
        
    def _plot_evaluation_results(self, results: pd.DataFrame) -> None:
        """Plot evaluation results.
        
        Args:
            results: DataFrame with evaluation metrics
        """
        # Wealth distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results, x='final_wealth', bins=30)
        plt.title('Distribution of Final Wealth')
        plt.xlabel('Final Wealth')
        plt.ylabel('Count')
        plt.savefig(self.output_dir / 'wealth_distribution.png')
        plt.close()
        
        # Utility vs Entropy
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results, x='entropy', y='mean_utility')
        plt.title('Utility vs Entropy')
        plt.xlabel('Entropy')
        plt.ylabel('Mean Utility')
        plt.savefig(self.output_dir / 'utility_vs_entropy.png')
        plt.close()
        
        # Mean allocation distribution
        mean_alloc = results['mean_allocation'].mean()
        std_alloc = results['allocation_std'].mean()
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(mean_alloc))
        plt.bar(x, mean_alloc)
        plt.fill_between(
            x,
            mean_alloc - std_alloc,
            mean_alloc + std_alloc,
            alpha=0.2
        )
        plt.title('Mean Allocation Distribution')
        plt.xlabel('Bucket')
        plt.ylabel('Allocation Weight')
        plt.savefig(self.output_dir / 'allocation_distribution.png')
        plt.close()
        
        # Save metrics summary
        metrics_summary = results.describe()
        metrics_summary.to_csv(self.output_dir / 'metrics_summary.csv')
        
    def evaluate_rebalancing(self,
                           features: np.ndarray,
                           price_changes: np.ndarray) -> Dict[str, float]:
        """Evaluate rebalancing behavior.
        
        Args:
            features: Input features
            price_changes: Price changes between steps
            
        Returns:
            Dictionary of rebalancing metrics
        """
        self.network.eval()
        features = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            # Get allocations for all steps
            allocations = self.network(features)
            
            # Compute allocation changes
            allocation_changes = torch.abs(
                allocations[1:] - allocations[:-1]
            ).sum(dim=-1)
            
            # Compute correlation with price changes
            correlation = np.corrcoef(
                allocation_changes.cpu().numpy(),
                np.abs(price_changes[1:])
            )[0, 1]
            
            # Compute rebalancing statistics
            mean_change = allocation_changes.mean().item()
            max_change = allocation_changes.max().item()
            change_std = allocation_changes.std().item()
            
        return {
            'price_correlation': correlation,
            'mean_allocation_change': mean_change,
            'max_allocation_change': max_change,
            'allocation_change_std': change_std
        }
        
    def evaluate_risk_metrics(self,
                            wealth_history: np.ndarray,
                            risk_free_rate: float = 0.0) -> Dict[str, float]:
        """Compute risk-adjusted performance metrics.
        
        Args:
            wealth_history: Array of wealth values
            risk_free_rate: Risk-free rate for Sharpe ratio
            
        Returns:
            Dictionary of risk metrics
        """
        # Convert to returns
        returns = np.diff(wealth_history) / wealth_history[:-1]
        
        # Compute metrics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Sharpe ratio
        excess_returns = returns - risk_free_rate
        sharpe = np.sqrt(252) * np.mean(excess_returns) / std_return
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdowns)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns)
        sortino = np.sqrt(252) * mean_return / downside_std
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'mean_return': mean_return,
            'return_std': std_return
        } 