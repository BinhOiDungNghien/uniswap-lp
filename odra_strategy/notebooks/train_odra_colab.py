# %% [markdown]
# # ODRA Strategy Training on Kaggle
# 
# This notebook implements the Optimal Dynamic Reset Allocation (ODRA) strategy training using Uniswap v3 tick-level data.

# %% [markdown]
# ## Setup Environment and Dependencies

# %%
import os
import sys
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
# Setup paths
KAGGLE_BASE = '/kaggle/working/uniswap-lp'
os.chdir(KAGGLE_BASE)
sys.path.append(KAGGLE_BASE)

# Create necessary directories
for dir_path in [
    'odra_strategy/data/raw',
    'odra_strategy/data/processed',
    'odra_strategy/outputs/logs',
    'odra_strategy/outputs/models',
    'odra_strategy/outputs/plots'
]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Upload Data
# 
# Before running this cell, make sure to:
# 1. Click "Add data" in Kaggle notebook
# 2. Upload your tick data files (*.tick.csv)
# 3. Set the dataset path in TICK_DATA_PATH

# %%
# Set path to your uploaded tick data
TICK_DATA_PATH = '/kaggle/input/your-dataset-name/*.tick.csv'  # Update this path

# Copy tick data to raw directory
import glob
import shutil

tick_files = glob.glob(TICK_DATA_PATH)
if not tick_files:
    raise FileNotFoundError(f"No tick data files found in {TICK_DATA_PATH}")

print(f"Found {len(tick_files)} tick data files")
for file in tick_files:
    shutil.copy(file, 'odra_strategy/data/raw/')
print("Data files copied to raw directory")

# %% [markdown]
# ## Import Dependencies and Setup

# %%
# Import project modules
from odra_strategy.data.data_loader import DataLoader
from odra_strategy.features.feature_engine import FeatureEngine
from odra_strategy.strategy.strategy_simulator import StrategySimulator
from odra_strategy.model.network import ODRANetwork
from odra_strategy.model.trainer import ODRATrainer
from odra_strategy.utils.logging_utils import setup_logging

# %%
# Load and adjust configuration
with open('odra_strategy/config/odra_config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Update paths for Kaggle environment
config['data']['raw_path'] = os.path.join(KAGGLE_BASE, 'odra_strategy/data/raw')
config['data']['processed_path'] = os.path.join(KAGGLE_BASE, 'odra_strategy/data/processed/odra_dataset.pkl')
config['logging']['log_dir'] = os.path.join(KAGGLE_BASE, 'odra_strategy/outputs/logs')
config['logging']['model_dir'] = os.path.join(KAGGLE_BASE, 'odra_strategy/outputs/models')
config['logging']['plot_dir'] = os.path.join(KAGGLE_BASE, 'odra_strategy/outputs/plots')

# Setup logging
setup_logging(config['logging']['log_dir'])

# %% [markdown]
# ## Initialize Components

# %%
# Initialize components
data_loader = DataLoader(config['data'])
feature_engine = FeatureEngine(config['features'])
simulator = StrategySimulator(config['simulator'])

# %% [markdown]
# ## Data Preparation

# %%
def prepare_data(data_loader, feature_engine, simulator, config):
    """Prepare training data."""
    print("Loading tick data...")
    tick_data = data_loader.get_data()
    
    print("Computing features...")
    features = feature_engine.compute_features(tick_data)
    features = feature_engine.normalize_features(features)
    
    print(f"Feature shape: {features.shape}")
    print("Features:", features.columns.tolist())
    
    # Initialize arrays
    n_steps = len(features)
    wealth_history = np.zeros(n_steps)
    wealth_history[0] = 1.0  # Start with unit wealth
    
    # Simulate strategy with random allocations
    n_buckets = 2 * config['simulator']['tau'] + 2
    
    print("Simulating initial strategy...")
    for t in tqdm(range(1, n_steps)):
        # Random allocation for initial data collection
        allocation = np.random.dirichlet(np.ones(n_buckets))
        
        # Simulate step
        result = simulator.simulate_step(
            features.iloc[t],
            allocation
        )
        
        wealth_history[t] = result['wealth']
        
    print(f"Wealth history shape: {wealth_history.shape}")
    print(f"Average wealth: {wealth_history.mean():.4f}")
    
    return features.values, wealth_history

# %%
# Prepare data
features, wealth = prepare_data(data_loader, feature_engine, simulator, config)

# %% [markdown]
# ## Model Training

# %%
# Create network
network = ODRANetwork(
    input_dim=features.shape[1],
    hidden_dims=[config['model']['hidden_units']] * config['model']['hidden_layers'],
    tau=config['simulator']['tau'],
    dropout=0.1
)

print("Network architecture:")
print(network)

# %%
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

print(f"Training set size: {len(train_features)}")
print(f"Validation set size: {len(val_features)}")

# %%
# Train model
print("Starting training...")
trainer.train(
    train_features=train_features,
    train_wealth=train_wealth,
    val_features=val_features,
    val_wealth=val_wealth
)

# %% [markdown]
# ## Save Results

# %%
# Save trained model back to Drive
os.system('mkdir -p "/content/drive/MyDrive/odra_models"')
os.system('cp -r outputs/models/* "/content/drive/MyDrive/odra_models/"')

# %% [markdown]
# ## Training Visualization

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(trainer.train_losses, label='Train')
plt.plot(trainer.val_losses, label='Validation')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(wealth_history)
plt.title('Wealth Evolution')
plt.xlabel('Step')
plt.ylabel('Wealth')

plt.tight_layout()
plt.show()

# Save plot
plt.savefig('outputs/plots/training_results.png')
plt.close()

print("Training complete! Check the plots and saved models.") 