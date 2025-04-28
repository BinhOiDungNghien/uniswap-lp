"""
Tests for UniswapDataLoader class
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import tempfile
import os

from odra_strategy.data.utils.data_loader import UniswapDataLoader

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        'data': {
            'raw_path': 'data/raw/ticks.csv',
            'processed_path': 'data/processed/odra_dataset.pkl'
        },
        'tick_spacing': 60
    }

@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    return pd.DataFrame({
        'block_number': [1, 1, 1, 1],
        'block_timestamp': ['2024-05-01 00:00:00'] * 4,
        'tx_type': ['SWAP', 'MINT', 'BURN', 'COLLECT'],
        'transaction_hash': ['0x123'] * 4,
        'pool_tx_index': [1, 1, 1, 1],
        'pool_log_index': [1, 1, 1, 1],
        'proxy_log_index': [1.0, 1.0, 1.0, 1.0],
        'sender': ['0x123'] * 4,
        'receipt': ['0x123'] * 4,
        'amount0': [1.0, 10.0, -5.0, 0.1],
        'amount1': [-1.0, -10.0, 5.0, 0.1],
        'total_liquidity': ['1000000000000000000'] * 4,
        'total_liquidity_delta': [0, 1000000, -500000, 0],
        'sqrtPriceX96': ['79228162514264337593543950336'] * 4,
        'current_tick': [0, None, None, None],
        'position_id': [None, '1', '1', '1'],
        'tick_lower': [None, -120, -120, -120],
        'tick_upper': [None, 120, 120, 120],
        'liquidity': [None, 1000000, 500000, None]
    })

def test_data_loader_initialization(sample_config):
    """Test data loader initialization."""
    loader = UniswapDataLoader(sample_config)
    assert loader.tick_spacing == 60
    assert str(loader.raw_path) == 'data/raw/ticks.csv'
    assert str(loader.processed_path) == 'data/processed/odra_dataset.pkl'

def test_process_transactions(sample_config, sample_raw_data):
    """Test transaction processing."""
    loader = UniswapDataLoader(sample_config)
    processed_df = loader.process_transactions(sample_raw_data)
    
    # Check if all transactions were processed
    assert len(processed_df) == 4
    
    # Check SWAP transaction
    swap_row = processed_df[processed_df['type'] == 'SWAP'].iloc[0]
    assert swap_row['price'] == 1.0  # (2^96/2^96)^2 = 1.0
    assert swap_row['bucket'] == 0  # tick 0 // 60 = 0
    assert swap_row['amount0'] == 1.0
    assert swap_row['amount1'] == -1.0
    
    # Check MINT transaction
    mint_row = processed_df[processed_df['type'] == 'MINT'].iloc[0]
    assert mint_row['tick_lower'] == -120
    assert mint_row['tick_upper'] == 120
    assert mint_row['position_id'] == '1'
    assert mint_row['amount0'] == 10.0
    assert mint_row['amount1'] == -10.0
    assert mint_row['liquidity_delta'] == 1000000
    
    # Check BURN transaction
    burn_row = processed_df[processed_df['type'] == 'BURN'].iloc[0]
    assert burn_row['tick_lower'] == -120
    assert burn_row['tick_upper'] == 120
    assert burn_row['position_id'] == '1'
    assert burn_row['amount0'] == -5.0
    assert burn_row['amount1'] == 5.0
    assert burn_row['liquidity_delta'] == -500000
    
    # Check COLLECT transaction
    collect_row = processed_df[processed_df['type'] == 'COLLECT'].iloc[0]
    assert collect_row['position_id'] == '1'
    assert collect_row['amount0'] == 0.1
    assert collect_row['amount1'] == 0.1
    assert collect_row['tick_lower'] == -120
    assert collect_row['tick_upper'] == 120

def test_save_and_load_processed_data(sample_config, sample_raw_data):
    """Test saving and loading processed data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update config with temporary paths
        sample_config['data']['processed_path'] = os.path.join(temp_dir, 'processed.pkl')
        
        loader = UniswapDataLoader(sample_config)
        processed_df = loader.process_transactions(sample_raw_data)
        
        # Save processed data
        loader.save_processed_data(processed_df)
        
        # Load processed data
        loaded_df = loader.load_processed_data()
        
        # Compare original and loaded data
        pd.testing.assert_frame_equal(processed_df, loaded_df)

def test_missing_required_columns(sample_config):
    """Test handling of missing required columns."""
    loader = UniswapDataLoader(sample_config)
    invalid_df = pd.DataFrame({'tx_type': ['SWAP']})  # Missing required columns
    
    with pytest.raises(ValueError):
        loader.process_transactions(invalid_df) 