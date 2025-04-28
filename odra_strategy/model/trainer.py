"""
Trainer module for ODRA strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import time

from .network import ODRANetwork
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

class ODRATrainer:
    """Trainer for ODRA strategy network."""
    
    def __init__(self,
                 network: ODRANetwork,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize trainer.
        
        Args:
            network: ODRA neural network
            config: Training configuration
            device: Device to train on
        """
        self.network = network.to(device)
        self.config = config
        self.device = device
        
        # Training parameters
        self.learning_rate = config['learning_rate']
        self.batch_size = config.get('batch_size', 32)
        self.n_epochs = config.get('n_epochs', 100)
        self.risk_aversion = config.get('risk_aversion', 0.01)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate
        )
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def prepare_batch(self, 
                     features: torch.Tensor,
                     wealth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch for training.
        
        Args:
            features: Feature tensor
            wealth: Wealth tensor
            
        Returns:
            Tuple of features and wealth tensors on device
        """
        features = features.to(self.device)
        wealth = wealth.to(self.device)
        return features, wealth
        
    def train_step(self,
                  features: torch.Tensor,
                  wealth: torch.Tensor) -> float:
        """Single training step.
        
        Args:
            features: Input features
            wealth: Final wealth values
            
        Returns:
            Loss value
        """
        self.network.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        allocations = self.network(features)
        
        # Compute utility-based loss
        utility = self.network.compute_utility(wealth, self.risk_aversion)
        loss = -torch.mean(utility)  # Negative because we want to maximize utility
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def validate(self,
                val_loader: DataLoader) -> float:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation loss
        """
        self.network.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for features, wealth in val_loader:
                features, wealth = self.prepare_batch(features, wealth)
                allocations = self.network(features)
                utility = self.network.compute_utility(wealth, self.risk_aversion)
                loss = -torch.mean(utility)
                total_loss += loss.item()
                n_batches += 1
                
        return total_loss / n_batches if n_batches > 0 else float('inf')
        
    def train(self,
              train_features: np.ndarray,
              train_wealth: np.ndarray,
              val_features: Optional[np.ndarray] = None,
              val_wealth: Optional[np.ndarray] = None,
              checkpoint_dir: str = 'outputs/models') -> Dict:
        """Train network with detailed progress tracking.
        
        Args:
            train_features: Training features
            train_wealth: Training wealth values
            val_features: Validation features
            val_wealth: Validation wealth values
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        self.logger.info("Converting data to tensors...")
        train_features = torch.FloatTensor(train_features).to(self.device)
        train_wealth = torch.FloatTensor(train_wealth).to(self.device)
        
        if val_features is not None and val_wealth is not None:
            val_features = torch.FloatTensor(val_features).to(self.device)
            val_wealth = torch.FloatTensor(val_wealth).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(train_features, train_wealth)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        if val_features is not None and val_wealth is not None:
            val_dataset = TensorDataset(val_features, val_wealth)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size
            )
        else:
            val_loader = None
            
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_start = time.time()
        
        # Print training setup
        self.logger.info("=" * 50)
        self.logger.info("Training Setup:")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(train_features):,}")
        self.logger.info(f"Validation samples: {len(val_features):,}" if val_features is not None else "No validation set")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Max epochs: {self.n_epochs}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info("=" * 50)
        
        # Main training loop with progress bar
        epoch_pbar = tqdm(range(self.n_epochs), desc="Training Progress", position=0)
        
        for epoch in epoch_pbar:
            # Training phase
            self.network.train()
            train_losses = []
            
            # Batch progress bar
            batch_pbar = tqdm(train_loader, 
                            desc=f"Epoch {epoch+1}/{self.n_epochs}",
                            position=1, 
                            leave=False)
            
            for batch_features, batch_wealth in batch_pbar:
                features, wealth = self.prepare_batch(batch_features, batch_wealth)
                loss = self.train_step(features, wealth)
                train_losses.append(loss)
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'avg_loss': f"{np.mean(train_losses):.4f}"
                })
            
            # Close batch progress bar
            batch_pbar.close()
            
            # Validation phase
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(
                        self.network.state_dict(),
                        checkpoint_path / 'best_model.pt'
                    )
                else:
                    patience_counter += 1
                
                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'train_loss': f"{np.mean(train_losses):.4f}",
                    'val_loss': f"{val_loss:.4f}",
                    'best_val': f"{best_val_loss:.4f}",
                    'time': f"{(time.time() - training_start)/60:.1f}m"
                })
                
                if patience_counter >= self.config.get('patience', 10):
                    self.logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
            else:
                # Update epoch progress bar without validation metrics
                epoch_pbar.set_postfix({
                    'train_loss': f"{np.mean(train_losses):.4f}",
                    'time': f"{(time.time() - training_start)/60:.1f}m"
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_freq', 10) == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses
                    },
                    checkpoint_path / f'checkpoint_epoch_{epoch}.pt'
                )
        
        # Training summary
        training_time = (time.time() - training_start) / 60
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Training Summary:")
        self.logger.info(f"Total epochs trained: {epoch + 1}")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        self.logger.info(f"Final training loss: {np.mean(train_losses):.4f}")
        self.logger.info(f"Training time: {training_time:.1f} minutes")
        self.logger.info("=" * 50)
        
        return {
            'final_train_loss': np.mean(train_losses),
            'final_val_loss': val_loss if val_loader else float('inf'),
            'best_val_loss': best_val_loss if val_loader else float('inf'),
            'epochs_trained': epoch + 1,
            'training_time_minutes': training_time
        }
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
        else:
            self.network.load_state_dict(checkpoint) 