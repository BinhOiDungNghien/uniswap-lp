"""
DataLoader for processing Uniswap v3 tick-level data
"""

import os
import glob
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads and processes tick-level data from Uniswap v3."""
    
    def __init__(self, config: Dict):
        """Initialize DataLoader.
        
        Args:
            config: Configuration dictionary containing data paths and parameters
        """
        self.raw_path = config['raw_path']
        self.processed_path = config['processed_path']
        self.episode_length = config['episode_length']
        
        # Create processed directory if not exists
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load and combine all CSV files from raw directory.
        
        Returns:
            Combined DataFrame of all tick data
        """
        # Get all CSV files
        csv_files = glob.glob(os.path.join(self.raw_path, "*.tick.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No tick data files found in {self.raw_path}")
            
        logger.info(f"Found {len(csv_files)} tick data files")
        
        # Define dtypes for columns
        dtypes = {
            'block_timestamp': str,  # Will convert to datetime later
            'tx_type': str,
            'amount0': float,
            'amount1': float,
            'sqrtPriceX96': str,  # Large numbers need string first
            'current_tick': float,
            'liquidity': float,
            'position_id': str,
            'tick_lower': float,
            'tick_upper': float
        }
        
        # Read and combine all files
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file, dtype=dtypes, low_memory=False)
            dfs.append(df)
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Convert timestamp to datetime
        combined_df['block_timestamp'] = pd.to_datetime(combined_df['block_timestamp'])
        
        return combined_df
        
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw tick data into features.
        
        Args:
            df: Raw tick data DataFrame
            
        Returns:
            Processed DataFrame with features
        """
        # Convert sqrtPriceX96 to numeric carefully
        df['sqrtPriceX96'] = pd.to_numeric(df['sqrtPriceX96'], errors='coerce')
        
        # Calculate price using numpy to handle large numbers
        sqrt_price = df['sqrtPriceX96'].to_numpy(dtype=np.float64)
        df['price'] = np.square(sqrt_price / (2**96))
        
        # Process different transaction types
        df_processed = pd.DataFrame()
        
        # For SWAP transactions
        swaps = df[df['tx_type'] == 'SWAP'].copy()
        if not swaps.empty:
            df_processed = swaps[['block_timestamp', 'tx_type', 'amount0', 'amount1', 
                                'sqrtPriceX96', 'current_tick', 'price']]
                                
        # Add MINT/BURN/COLLECT info
        for tx_type in ['MINT', 'BURN', 'COLLECT']:
            type_df = df[df['tx_type'] == tx_type].copy()
            if not type_df.empty:
                type_df = type_df[['block_timestamp', 'tx_type', 'amount0', 'amount1',
                                 'liquidity', 'position_id', 'tick_lower', 'tick_upper']]
                df_processed = pd.concat([df_processed, type_df], ignore_index=True)
        
        # Sort by timestamp
        df_processed = df_processed.sort_values('block_timestamp')
        
        return df_processed
        
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data to pickle file.
        
        Args:
            df: Processed DataFrame to save
        """
        df.to_pickle(self.processed_path)
        logger.info(f"Saved processed data to {self.processed_path}")
        
    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """Load processed data from pickle file if exists.
        
        Returns:
            Processed DataFrame if exists, None otherwise
        """
        if os.path.exists(self.processed_path):
            return pd.read_pickle(self.processed_path)
        return None
        
    def get_data(self) -> pd.DataFrame:
        """Main method to get processed data.
        
        Returns:
            Processed DataFrame ready for feature extraction
        """
        # Try loading processed data first
        df = self.load_processed_data()
        if df is not None:
            logger.info("Loaded pre-processed data")
            return df
            
        # Load and process raw data if processed not found
        logger.info("Processing raw data...")
        raw_df = self.load_raw_data()
        processed_df = self.process_data(raw_df)
        self.save_processed_data(processed_df)
        
        return processed_df
        
    def get_episodes(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split data into episodes of fixed length.
        
        Args:
            df: Full processed DataFrame
            
        Returns:
            List of episode DataFrames
        """
        episodes = []
        
        # Calculate number of complete episodes
        n_episodes = len(df) // self.episode_length
        
        for i in range(n_episodes):
            start_idx = i * self.episode_length
            end_idx = start_idx + self.episode_length
            episode = df.iloc[start_idx:end_idx].copy()
            episodes.append(episode)
            
        return episodes 