#!/usr/bin/env python3
"""
Main entry point for ODRA strategy
"""

import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple
import os

from odra_strategy.data.data_loader import DataLoader
from odra_strategy.features.feature_engine import FeatureEngine
from odra_strategy.strategy.strategy_simulator import StrategySimulator
from odra_strategy.model.network import ODRANetwork
from odra_strategy.model.trainer import ODRATrainer
from odra_strategy.model.evaluator import ODRAEvaluator
from odra_strategy.model.loss import EntropyRegularizedLoss, TransactionCostLoss
from odra_strategy.utils.logging_utils import setup_logging
from utils.kaggle_utils import adjust_paths_for_kaggle, setup_kaggle_environment

def load_config(config_path: str) -> Dict:
    """Load and adjust configuration based on environment."""
    return adjust_paths_for_kaggle(config_path)

def prepare_data(data_loader: DataLoader,
                feature_engine: FeatureEngine,
                simulator: StrategySimulator,
                config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training data.
    
    Args:
        data_loader: Data loading and processing
        feature_engine: Feature computation engine
        simulator: Strategy simulator
        config: Configuration dictionary
        
    Returns:
        Tuple of features and wealth arrays
    """
    # Load and process tick data
    tick_data = data_loader.get_data()
    
    # Extract features
    features = feature_engine.compute_features(tick_data)
    features = feature_engine.normalize_features(features)
    
    # Initialize arrays
    n_steps = len(features)
    wealth_history = np.zeros(n_steps)
    wealth_history[0] = 1.0  # Start with unit wealth
    
    # Simulate strategy with random allocations initially
    n_buckets = 2 * config['simulator']['tau'] + 2
    
    for t in range(1, n_steps):
        # Random allocation for initial data collection
        allocation = np.random.dirichlet(np.ones(n_buckets))
        
        # Simulate step
        result = simulator.simulate_step(
            features.iloc[t],
            allocation
        )
        
        wealth_history[t] = result['wealth']
        
    return features.values, wealth_history

def train_model(features: np.ndarray,
                wealth: np.ndarray,
                config: Dict) -> ODRANetwork:
    """Train ODRA network.
    
    Args:
        features: Input features
        wealth: Wealth values
        config: Configuration dictionary
        
    Returns:
        Trained network
    """
    # Create network
    network = ODRANetwork(
        input_dim=features.shape[1],
        hidden_dims=[config['model']['hidden_units']] * config['model']['hidden_layers'],
        tau=config['simulator']['tau'],
        dropout=0.1
    )
    
    # Initialize trainer
    trainer = ODRATrainer(
        network=network,
        config=config['model'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Split data
    train_size = int(0.8 * len(features))
    train_features = features[:train_size]
    train_wealth = wealth[:train_size]
    val_features = features[train_size:]
    val_wealth = wealth[train_size:]
    
    # Train
    trainer.train(
        train_features=train_features,
        train_wealth=train_wealth,
        val_features=val_features,
        val_wealth=val_wealth
    )
    
    return network

def evaluate_model(network: ODRANetwork,
                  features: np.ndarray,
                  wealth: np.ndarray,
                  config: Dict) -> None:
    """Evaluate trained model.
    
    Args:
        network: Trained network
        features: Input features
        wealth: Wealth values
        config: Configuration dictionary
    """
    # Initialize evaluator
    evaluator = ODRAEvaluator(
        network=network,
        config=config['evaluation']
    )
    
    # Create evaluation episodes
    n_episodes = config['evaluation'].get('n_episodes', 10)
    episode_length = config['evaluation'].get('episode_length', 1000)
    
    episodes = []
    for _ in range(n_episodes):
        start_idx = np.random.randint(0, len(features) - episode_length)
        end_idx = start_idx + episode_length
        episode_features = features[start_idx:end_idx]
        episode_wealth = wealth[start_idx:end_idx]
        episodes.append((episode_features, episode_wealth))
    
    # Evaluate episodes
    results = evaluator.evaluate_multiple_episodes(episodes)
    
    # Compute risk metrics
    risk_metrics = evaluator.evaluate_risk_metrics(wealth)
    
    # Log results
    logging.info("Evaluation Results:")
    logging.info(f"Mean Utility: {results['mean_utility'].mean():.4f}")
    logging.info(f"Mean Entropy: {results['entropy'].mean():.4f}")
    logging.info(f"Mean Final Wealth: {results['final_wealth'].mean():.4f}")
    logging.info(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
    logging.info(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.4f}")
    logging.info(f"Maximum Drawdown: {risk_metrics['max_drawdown']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='ODRA Strategy Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['process', 'train', 'evaluate'],
                       help='Mode to run: process data, train model, or evaluate')
    args = parser.parse_args()

    # Setup Kaggle environment if needed
    setup_kaggle_environment()
    
    # Load config with adjusted paths
    config = load_config(args.config)
    
    if args.mode == 'process':
        data_loader = DataLoader(config['data'])
        data_loader.process_raw_data()
    
    elif args.mode == 'train':
        trainer = ODRATrainer(config)
        trainer.train()
    
    elif args.mode == 'evaluate':
        evaluator = ODRAEvaluator(config)
        evaluator.evaluate()

if __name__ == '__main__':
    main() 