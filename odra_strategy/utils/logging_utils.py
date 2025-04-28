"""
Logging utilities for ODRA strategy
"""

import logging
import json
from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def setup_logging(log_dir: str = 'logs', level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure file handler
    file_handler = logging.FileHandler(log_path / 'odra_strategy.log')
    file_handler.setLevel(level)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get root logger and add handlers
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_simulation_step(step_data: Dict[str, Any], log_dir: str = 'logs') -> None:
    """Log simulation step data to JSON file."""
    log_path = Path(log_dir) / 'simulation_steps.jsonl'
    with open(log_path, 'a') as f:
        f.write(json.dumps(step_data) + '\n')

def plot_simulation_results(log_dir: str = 'logs', output_dir: str = 'outputs/plots'):
    """Generate plots from simulation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load simulation data
    steps_data = []
    log_path = Path(log_dir) / 'simulation_steps.jsonl'
    with open(log_path, 'r') as f:
        for line in f:
            steps_data.append(json.loads(line))
    
    df = pd.DataFrame(steps_data)
    
    # Plot wealth over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y='wealth')
    plt.title('Wealth Over Time')
    plt.xlabel('Step')
    plt.ylabel('Wealth')
    plt.savefig(output_path / 'wealth_over_time.png')
    plt.close()
    
    # Plot fees collected
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y='fees_collected')
    plt.title('Cumulative Fees Collected')
    plt.xlabel('Step')
    plt.ylabel('Fees')
    plt.savefig(output_path / 'fees_collected.png')
    plt.close()
    
    # Plot number of active positions
    active_positions = []
    for step in steps_data:
        n_active = sum(1 for pos in step['positions'].values() 
                      if pos['tick_lower'] <= step.get('current_tick', 0) <= pos['tick_upper'])
        active_positions.append(n_active)
    
    plt.figure(figsize=(12, 6))
    plt.plot(active_positions)
    plt.title('Number of Active Positions')
    plt.xlabel('Step')
    plt.ylabel('Active Positions')
    plt.savefig(output_path / 'active_positions.png')
    plt.close()
    
    # Plot liquidity distribution
    if len(steps_data) > 0:
        last_step = steps_data[-1]
        positions = pd.DataFrame(last_step['positions'].values())
        
        plt.figure(figsize=(12, 6))
        plt.hist(positions['liquidity'], bins=30)
        plt.title('Liquidity Distribution (Last Step)')
        plt.xlabel('Liquidity')
        plt.ylabel('Count')
        plt.savefig(output_path / 'liquidity_distribution.png')
        plt.close()

def plot_price_ranges(positions: Dict[str, Dict], current_tick: int, 
                     output_dir: str = 'outputs/plots'):
    """Plot current price and position ranges."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    # Plot position ranges
    for pos_id, pos in positions.items():
        plt.hlines(
            y=pos['liquidity'],
            xmin=pos['tick_lower'],
            xmax=pos['tick_upper'],
            label=f'Position {pos_id[:8]}',
            alpha=0.5
        )
    
    # Plot current tick
    plt.axvline(x=current_tick, color='r', linestyle='--', label='Current Tick')
    
    plt.title('Position Ranges and Current Price')
    plt.xlabel('Tick')
    plt.ylabel('Liquidity')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path / 'position_ranges.png')
    plt.close() 