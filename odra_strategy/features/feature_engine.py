"""
Feature Engineering for ODRA Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from pathlib import Path

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
        """
        self.config = config
        self.alpha = config['alpha_ewma']
        self.tick_spacing = config['tick_spacing']
        self.fee_tier = config.get('fee_tier', 3000)
        self.price_impact_threshold = config.get('price_impact_threshold', 0.001)
        
    def compute_price(self, sqrtPriceX96: float) -> float:
        """Compute price from sqrtPriceX96."""
        if pd.isna(sqrtPriceX96):
            return 1.0
        sqrtPriceX96 = int(sqrtPriceX96)
        return (sqrtPriceX96 / 2**96)**2
        
    def is_non_arbitrage_swap(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if a swap is non-arbitrage."""
        row = df.iloc[idx]
        if row['tx_type'] != 'SWAP':
            return False
            
        # First SWAP is always considered non-arbitrage
        if idx == 0:
            return True
            
        # Get previous SWAP
        prev_swaps = df[df['tx_type'] == 'SWAP'].iloc[:idx]
        if len(prev_swaps) == 0:
            return True
            
        prev_swap = prev_swaps.iloc[-1]
        
        # Check price impact
        price_before = self.compute_price(prev_swap['sqrtPriceX96'])
        price_after = self.compute_price(row['sqrtPriceX96'])
        
        if price_before == 0:
            return True
            
        price_impact = abs(price_after - price_before) / price_before
        if price_impact > self.price_impact_threshold:
            return False
            
        return True
        
    def compute_ewma_volume(self, df: pd.DataFrame) -> pd.Series:
        """Compute EWMA of volume for non-arbitrage swaps."""
        # Initialize EWMA series for all transactions
        ewma = pd.Series(0.0, index=df.index)
        
        # Get SWAP transactions only
        swap_df = df[df['tx_type'] == 'SWAP'].copy()
        
        if len(swap_df) > 0:
            # Process first SWAP
            first_idx = swap_df.index[0]
            ewma.loc[first_idx] = abs(float(swap_df.loc[first_idx, 'amount0']))
            last_value = ewma.loc[first_idx]
            
            # Process remaining SWAPs
            for idx in swap_df.index[1:]:
                if self.is_non_arbitrage_swap(df, df.index.get_loc(idx)):
                    volume = abs(float(swap_df.loc[idx, 'amount0']))
                    last_value = self.alpha * volume + (1 - self.alpha) * last_value
                ewma.loc[idx] = last_value
                
        # Forward fill for non-SWAP transactions
        ewma = ewma.fillna(method='ffill')
        
        return ewma
        
    def compute_center_bucket(self, df: pd.DataFrame) -> pd.Series:
        """Compute center bucket from tick."""
        # Initialize buckets for all transactions
        buckets = pd.Series(index=df.index)
        
        # Get SWAP transactions only
        swap_df = df[df['tx_type'] == 'SWAP'].copy()
        
        for idx in swap_df.index:
            tick = swap_df.loc[idx, 'current_tick']
            if not pd.isna(tick):
                buckets.loc[idx] = int(float(tick) // self.tick_spacing)
                
        # Forward fill for non-SWAP transactions
        buckets = buckets.fillna(method='ffill')
        
        return buckets
        
    def compute_wealth(self, df: pd.DataFrame) -> pd.Series:
        """Compute wealth from position changes."""
        wealth = pd.Series(0.0, index=df.index)
        
        # Process transactions sequentially
        cumulative_wealth = 0.0
        for idx in df.index:
            tx = df.loc[idx]
            if tx['tx_type'] in ['MINT', 'BURN']:
                amount0 = abs(float(tx['amount0'])) if not pd.isna(tx['amount0']) else 0
                amount1 = abs(float(tx['amount1'])) if not pd.isna(tx['amount1']) else 0
                cumulative_wealth += (amount0 + amount1) / 2  # Scale by 2 instead of 4
            elif tx['tx_type'] == 'COLLECT':
                amount0 = abs(float(tx['amount0'])) if not pd.isna(tx['amount0']) else 0
                amount1 = abs(float(tx['amount1'])) if not pd.isna(tx['amount1']) else 0
                cumulative_wealth += (amount0 + amount1) / 2  # Scale by 2 instead of 4
            wealth.loc[idx] = cumulative_wealth
            
        return wealth
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for ODRA strategy."""
        # Initialize features DataFrame
        features = pd.DataFrame(index=df.index)
        
        # Compute individual features
        features['ewma_volume'] = self.compute_ewma_volume(df)
        features['center_bucket'] = self.compute_center_bucket(df)
        features['wealth'] = self.compute_wealth(df)
        
        # Compute pool price
        features['pool_price'] = df['sqrtPriceX96'].apply(self.compute_price)
        
        # Compute normalized time
        features['t_T'] = np.linspace(0, 1, len(features))
        
        return features
        
    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using z-score normalization.
        
        Args:
            features: DataFrame containing raw features
            
        Returns:
            DataFrame with normalized features
        """
        normalized = features.copy()
        
        # Keep t_T unchanged
        for col in normalized.columns:
            if col != 't_T':
                mean = normalized[col].mean()
                std = normalized[col].std()
                if std > 1e-10:  # If there is variation
                    normalized[col] = (normalized[col] - mean) / std
                else:  # If no variation, center at 0 with tiny noise
                    normalized[col] = np.random.normal(0, 1e-10, len(normalized))
        
        return normalized 