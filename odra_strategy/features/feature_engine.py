"""
Feature Engineering for ODRA Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Computes and normalizes features from Uniswap v3 tick-level data.
    Features include:
    - EWMA volume (filtered for non-arbitrage)
    - Center bucket
    - Wealth
    - Normalized time (t/T)
    - Pool price
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engine with configuration.
        
        Args:
            config: Dictionary containing:
                - alpha_ewma: EWMA decay factor
                - tick_spacing: Tick spacing for the pool
                - fee_tier: Pool fee tier (e.g. 0.3% = 3000)
                - price_impact_threshold: Max price impact to consider non-arbitrage
                - n_jobs: Number of parallel jobs (default: number of CPU cores)
                - token0_decimals: Number of decimals for token0 (e.g., 18 for ETH)
                - token1_decimals: Number of decimals for token1 (e.g., 6 for USDC)
        """
        self.config = config
        self.alpha = config['alpha_ewma']
        self.tick_spacing = config['tick_spacing']
        self.fee_tier = config.get('fee_tier', 3000)
        self.price_impact_threshold = config.get('price_impact_threshold', 0.001)
        self.n_jobs = config.get('n_jobs', mp.cpu_count())
        self.token0_decimals = config.get('token0_decimals', 18)
        self.token1_decimals = config.get('token1_decimals', 6)
        
    def _normalize_amount(self, amount: float, is_token0: bool) -> float:
        """Normalize token amount by its decimals.
        
        Args:
            amount: Raw token amount
            is_token0: Whether this is token0 (True) or token1 (False)
            
        Returns:
            Normalized amount in standard units
        """
        decimals = self.token0_decimals if is_token0 else self.token1_decimals
        return amount / (10 ** decimals)
        
    def compute_price_from_tick(self, tick: float) -> float:
        """Compute price from tick value.
        
        Args:
            tick: The tick value
            
        Returns:
            Price (token1/token0) at this tick
        """
        if pd.isna(tick):
            return 1.0
        
        # Price = 1.0001^tick * 10^(decimals1-decimals0)
        decimal_adjustment = 10 ** (self.token1_decimals - self.token0_decimals)
        return (1.0001 ** tick) * decimal_adjustment
        
    def is_non_arbitrage_swap(self, df_chunk: pd.DataFrame) -> np.ndarray:
        """Check if swaps are non-arbitrage using vectorized operations."""
        is_swap = (df_chunk['tx_type'] == 'SWAP').values
        if not any(is_swap):
            return np.zeros(len(df_chunk), dtype=bool)
            
        prices = np.array([self.compute_price_from_tick(tick) for tick in df_chunk['current_tick'].values])
        price_changes = np.abs(np.diff(prices, prepend=prices[0])) / (prices + 1e-10)
        
        return (is_swap) & (price_changes <= self.price_impact_threshold)
        
    def process_chunk(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of data to compute features."""
        # Clean up data first
        df_chunk = df_chunk.replace('', np.nan)  # Replace empty strings with NaN
        
        # Strip whitespace only for string columns
        string_cols = df_chunk.select_dtypes(include=['object']).columns
        for col in string_cols:
            df_chunk[col] = df_chunk[col].str.strip()
        
        # Convert numeric columns, create if missing
        numeric_cols = ['amount0_adjusted', 'amount1_adjusted', 'total_liquidity', 'total_liquidity_delta', 
                       'current_tick', 'tick_lower', 'tick_upper', 'liquidity']
        for col in numeric_cols:
            if col not in df_chunk.columns:
                logger.warning(f"Column {col} not found in data. Creating with NaN values.")
                df_chunk[col] = np.nan
            df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
        
        # Log column names for debugging
        logger.info(f"Available columns: {', '.join(df_chunk.columns)}")
        logger.info(f"First few rows of amount0_adjusted: {df_chunk['amount0_adjusted'].head()}")
        
        # Compute volumes using coalesced amounts
        amount0 = df_chunk['amount0_adjusted'].fillna(0).astype(float)
        amount1 = df_chunk['amount1_adjusted'].fillna(0).astype(float)
        volumes = np.maximum(np.abs(amount0), np.abs(amount1))
        
        # Rest of processing
        is_valid_swap = self.is_non_arbitrage_swap(df_chunk)
        
        # Volumes are already normalized by token decimals in DataLoader
        volumes[~is_valid_swap] = 0
        
        # Compute EWMA for chunk
        ewma = np.zeros_like(volumes)
        if len(volumes) > 0:
            ewma[0] = volumes[0]
            for i in range(1, len(volumes)):
                ewma[i] = self.alpha * volumes[i] + (1 - self.alpha) * ewma[i-1]
                
        # Compute center bucket and prices from ticks
        ticks = df_chunk['current_tick'].astype(float).fillna(0).values
        buckets = np.floor(ticks / self.tick_spacing).astype(int)
        prices = np.array([self.compute_price_from_tick(tick) for tick in ticks])
        
        # Compute wealth with normalized amounts
        wealth = np.zeros_like(volumes)
        cumulative_wealth = 0.0
        for i in range(len(df_chunk)):
            tx = df_chunk.iloc[i]
            if tx['tx_type'] in ['MINT', 'BURN', 'COLLECT']:
                amount0 = abs(float(tx['amount0_adjusted'])) if not pd.isna(tx['amount0_adjusted']) else 0
                amount1 = abs(float(tx['amount1_adjusted'])) if not pd.isna(tx['amount1_adjusted']) else 0
                # Amounts are already normalized by decimals
                cumulative_wealth += (amount0 + amount1) / 2
            wealth[i] = cumulative_wealth
            
        return pd.DataFrame({
            'ewma_volume': ewma,
            'center_bucket': buckets,
            'wealth': wealth,
            'price': prices
        }, index=df_chunk.index)
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for ODRA strategy using parallel processing."""
        logger.info(f"Computing features using {self.n_jobs} parallel jobs...")
        
        # Validate and preprocess input data
        logger.info(f"Input DataFrame shape: {df.shape}")
        logger.info(f"Input columns: {', '.join(df.columns)}")
        
        # Check if column names need cleaning
        df.columns = df.columns.str.strip().str.lower()
        logger.info(f"Cleaned columns: {', '.join(df.columns)}")
        
        # Ensure required columns exist
        required_cols = ['amount0_adjusted', 'amount1_adjusted', 'tx_type', 'current_tick']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Split data into chunks for parallel processing
        chunk_size = max(1, len(df) // (self.n_jobs * 4))
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        logger.info(f"Split data into {len(chunks)} chunks of size {chunk_size}")
        
        # Process chunks in parallel
        with mp.Pool(self.n_jobs) as pool:
            results = list(tqdm(
                pool.imap(self.process_chunk, chunks),
                total=len(chunks),
                desc="Processing chunks"
            ))
            
        # Combine results
        features = pd.concat(results)
        
        # Add normalized time
        features['t_T'] = np.linspace(0, 1, len(features))
        
        # Define column types
        numeric_cols = ['amount0_adjusted', 'amount1_adjusted', 'sqrtPriceX96', 'current_tick',
                       'liquidity', 'tick_lower', 'tick_upper']
        categorical_cols = ['tx_type', 'position_id']
        computed_cols = ['ewma_volume', 'center_bucket', 'wealth', 'price', 't_T']
        
        # Create a new DataFrame for numeric features only
        numeric_features = pd.DataFrame()
        
        # Add computed features
        for col in computed_cols:
            numeric_features[col] = features[col].astype('float32')
            
        # Add numeric columns from original data
        for col in numeric_cols:
            if col in df.columns:
                numeric_features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')
                
        # Store categorical columns separately
        categorical_features = pd.DataFrame()
        for col in categorical_cols:
            if col in df.columns:
                categorical_features[col] = df[col]
        
        # Combine numeric and categorical features
        final_features = pd.concat([numeric_features, categorical_features], axis=1)
        
        return final_features
        
    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using vectorized operations."""
        logger.info("Normalizing features...")
        normalized = features.copy()
        
        # Columns to exclude from normalization
        exclude_cols = ['t_T', 'tx_type', 'position_id']
        
        # Get only numeric columns for normalization
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        if not cols_to_normalize:
            logger.warning("No columns to normalize. Check if data contains numeric features.")
            return normalized
            
        logger.info(f"Normalizing {len(cols_to_normalize)} columns: {', '.join(cols_to_normalize)}")
        
        # Calculate statistics
        means = normalized[cols_to_normalize].mean()
        stds = normalized[cols_to_normalize].std()
        
        # Apply normalization with progress bar
        with tqdm(total=len(cols_to_normalize), desc="Normalizing columns") as pbar:
            for col in cols_to_normalize:
                if stds[col] > 1e-10:  # If there is variation
                    normalized[col] = (normalized[col] - means[col]) / (stds[col] + 1e-10)
                else:  # If no variation, center at 0 with tiny noise
                    normalized[col] = np.random.normal(0, 1e-10, len(normalized))
                pbar.update(1)
        
        logger.info("Feature normalization completed")
        return normalized 