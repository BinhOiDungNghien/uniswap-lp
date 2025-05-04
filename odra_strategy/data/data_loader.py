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
            config: Configuration dictionary containing:
                - raw_path: Path to raw data directory
                - processed_path: Path to save processed data
                - episode_length: Length of each episode
                - token0_decimals: Decimals for token0
                - token1_decimals: Decimals for token1
        """
        self.raw_path = config['raw_path']
        self.processed_path = config['processed_path']
        self.episode_length = config['episode_length']
        
        # Token decimals (default to 18 for most ERC20 tokens)
        self.token0_decimals = config.get('token0_decimals', 18)
        self.token1_decimals = config.get('token1_decimals', 18)
        
        # Create processed directory if not exists
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        
    def calculate_price(self, sqrt_price_x96: np.ndarray) -> np.ndarray:
        """Calculate real token price from sqrtPriceX96.
        
        The price is calculated as:
        price = (sqrtPriceX96/2^96)^2 * (10^decimal1)/(10^decimal0)
        
        Args:
            sqrt_price_x96: Array of sqrtPriceX96 values
            
        Returns:
            Array of real token prices
        """
        # Convert to float64 for precision
        sqrt_price = sqrt_price_x96.astype(np.float64)
        
        # Calculate base price (sqrtPriceX96/2^96)^2
        base_price = np.square(sqrt_price / (2**96))
        
        # Adjust for token decimals
        decimal_adjustment = 10**(self.token1_decimals - self.token0_decimals)
        
        return base_price * decimal_adjustment
        
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
        
        # Calculate price using decimal-adjusted formula
        sqrt_price = df['sqrtPriceX96'].to_numpy(dtype=np.float64)
        df['price'] = self.calculate_price(sqrt_price)
        
        # Add price in both directions for convenience
        df['price0_1'] = df['price']  # Price of token0 in terms of token1
        df['price1_0'] = 1 / df['price']  # Price of token1 in terms of token0
        
        # Process different transaction types
        df_processed = pd.DataFrame()
        
        # For SWAP transactions
        swaps = df[df['tx_type'] == 'SWAP'].copy()
        if not swaps.empty:
            # Adjust amounts for decimals
            swaps['amount0_adjusted'] = swaps['amount0'] / (10**self.token0_decimals)
            swaps['amount1_adjusted'] = swaps['amount1'] / (10**self.token1_decimals)
            
            df_processed = swaps[['block_timestamp', 'tx_type', 
                                'amount0_adjusted', 'amount1_adjusted',
                                'sqrtPriceX96', 'current_tick', 
                                'price0_1', 'price1_0']]
                                
        # Add MINT/BURN/COLLECT info
        for tx_type in ['MINT', 'BURN', 'COLLECT']:
            type_df = df[df['tx_type'] == tx_type].copy()
            if not type_df.empty:
                # Adjust amounts for decimals
                type_df['amount0_adjusted'] = type_df['amount0'] / (10**self.token0_decimals)
                type_df['amount1_adjusted'] = type_df['amount1'] / (10**self.token1_decimals)
                
                type_df = type_df[['block_timestamp', 'tx_type', 
                                 'amount0_adjusted', 'amount1_adjusted',
                                 'liquidity', 'position_id', 
                                 'tick_lower', 'tick_upper']]
                df_processed = pd.concat([df_processed, type_df], ignore_index=True)
        
        # Sort by timestamp
        df_processed = df_processed.sort_values('block_timestamp')
        
        # Validate prices
        if (df_processed['price0_1'] <= 0).any():
            raise ValueError("Found non-positive prices after conversion")
            
        logger.info(f"Processed {len(df_processed):,} transactions with decimal adjustment")
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