"""
Data Loader for Uniswap v3 Tick-Level Data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import glob
import os

logger = logging.getLogger(__name__)

class UniswapDataLoader:
    """Loader for Uniswap v3 tick-level data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary containing data paths and parameters
        """
        self.config = config
        # Convert paths
        self.raw_path = Path(config['data']['raw_path'])
        self.processed_path = Path(config['data']['processed_path'])
        self.tick_spacing = config['tick_spacing']
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw tick-level data from CSV files.
        
        Returns:
            DataFrame containing raw tick-level data
        """
        logger.info(f"Loading raw data from {self.raw_path}")
        try:
            # Get all CSV files in the raw directory
            csv_files = glob.glob(str(self.raw_path / "*.tick.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No .tick.csv files found in {self.raw_path}")
            
            # Load and concatenate all CSV files
            dfs = []
            for file in sorted(csv_files):
                logger.info(f"Loading {file}")
                df = pd.read_csv(file)
                dfs.append(df)
            
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Successfully loaded {len(df)} rows of data from {len(csv_files)} files")
            return df
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise
            
    def process_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw transaction data.
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            pd.DataFrame: Processed transactions
        """
        required_columns = [
            'tx_type', 'amount0', 'amount1', 'sqrtPriceX96',
            'current_tick', 'position_id', 'tick_lower', 'tick_upper',
            'total_liquidity_delta'
        ]
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Need: {required_columns}")
            
        processed = pd.DataFrame()
        processed['type'] = df['tx_type']
        processed['amount0'] = df['amount0'].astype(float)
        processed['amount1'] = df['amount1'].astype(float)
        processed['tick'] = df['current_tick']
        processed['position_id'] = df['position_id']
        processed['tick_lower'] = df['tick_lower']
        processed['tick_upper'] = df['tick_upper']
        processed['liquidity_delta'] = df['total_liquidity_delta']
        
        # Convert sqrtPriceX96 to price
        processed['price'] = (df['sqrtPriceX96'].astype(float) / 2**96)**2
        processed['bucket'] = df['current_tick'].div(self.tick_spacing).ffill()
        
        return processed
        
    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Save processed data to pickle file.
        
        Args:
            df: Processed DataFrame to save
        """
        logger.info(f"Saving processed data to {self.processed_path}")
        try:
            # Create directory if it doesn't exist
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(self.processed_path)
            logger.info("Successfully saved processed data")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
            
    def load_processed_data(self) -> pd.DataFrame:
        """
        Load processed data from pickle file.
        
        Returns:
            DataFrame containing processed data
        """
        logger.info(f"Loading processed data from {self.processed_path}")
        try:
            df = pd.read_pickle(self.processed_path)
            logger.info(f"Successfully loaded {len(df)} rows of processed data")
            return df
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise 