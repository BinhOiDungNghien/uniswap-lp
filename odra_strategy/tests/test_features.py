"""
Tests for Feature Engineering
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from odra_strategy.features.feature_engine import FeatureEngine

@pytest.fixture
def sample_config():
    """Basic configuration for testing."""
    return {
        'alpha_ewma': 0.05,
        'tick_spacing': 60,
        'fee_tier': 3000,
        'price_impact_threshold': 0.001,
        'sandwich_window': 2
    }

@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    # Create timestamps
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(minutes=i) for i in range(7)]
    
    # Create transaction data
    data = {
        'timestamp': timestamps,
        'type': ['SWAP', 'MINT', 'SWAP', 'BURN', 'SWAP', 'COLLECT', 'COLLECT'],
        'amount0': [1.0, 10.0, 1.1, -5.0, 1.2, 0.1, 0.1],
        'amount1': [1.0, 10.0, -1.1, -5.0, -1.2, 0.1, 0.1],
        'sqrtPriceX96_before': [None, None, '79228162514264337593543950336', None, 
                               '79228162514264337593543950336', None, None],
        'sqrtPriceX96': ['79228162514264337593543950336', None, '79228162514264337593543950336',
                         None, '79228162514264337593543950336', None, None],
        'tick': [0, None, 60, None, 120, None, None],
        'position_id': [None, 'pos1', None, 'pos1', None, 'pos1', 'pos1'],
        'tick_lower': [None, -120, None, -120, None, -120, -120],
        'tick_upper': [None, 120, None, 120, None, 120, 120]
    }
    return pd.DataFrame(data).set_index('timestamp')

def test_price_computation(sample_config):
    """Test price computation from sqrtPriceX96."""
    engine = FeatureEngine(sample_config)
    
    # Test basic price computation
    sqrtPriceX96 = '79228162514264337593543950336'  # Should give price ≈ 1.0
    price = engine.compute_price(sqrtPriceX96)
    assert np.isclose(price, 1.0, rtol=1e-5)
    
    # Test NA handling
    assert engine.compute_price(None) == 1.0
    assert engine.compute_price(np.nan) == 1.0

def test_non_arbitrage_detection(sample_config, sample_transactions):
    """Test non-arbitrage swap detection."""
    engine = FeatureEngine(sample_config)
    
    # First SWAP should always be non-arbitrage
    assert engine.is_non_arbitrage_swap(sample_transactions, 0)
    
    # Second SWAP should be non-arbitrage (small price impact)
    assert engine.is_non_arbitrage_swap(sample_transactions, 2)
    
    # Non-SWAP transaction should return False
    assert not engine.is_non_arbitrage_swap(sample_transactions, 1)

def test_ewma_volume(sample_config, sample_transactions):
    """Test EWMA volume computation."""
    engine = FeatureEngine(sample_config)
    ewma = engine.compute_ewma_volume(sample_transactions)
    
    # Should only have values for SWAP transactions
    assert len(ewma) == 3  # Number of SWAP transactions
    
    # Check EWMA calculation
    swaps = sample_transactions[sample_transactions['type'] == 'SWAP']
    assert ewma.index.equals(swaps.index)
    assert ewma.iloc[0] == 1.0  # First SWAP amount
    assert ewma.iloc[1] > ewma.iloc[0]  # Should increase with volume

def test_center_bucket(sample_config, sample_transactions):
    """Test center bucket computation."""
    engine = FeatureEngine(sample_config)
    buckets = engine.compute_center_bucket(sample_transactions)
    
    # Should only have values for SWAP transactions
    assert len(buckets) == 3  # Number of SWAP transactions
    
    # Check bucket values
    swaps = sample_transactions[sample_transactions['type'] == 'SWAP']
    assert buckets.index.equals(swaps.index)
    assert buckets.iloc[0] == 0  # First tick = 0
    assert buckets.iloc[1] == 1  # Second tick = 60

def test_wealth_computation(sample_config, sample_transactions):
    """Test wealth computation."""
    engine = FeatureEngine(sample_config)
    wealth = engine.compute_wealth(sample_transactions)
    
    # Should have values for first 5 timestamps
    assert len(wealth) == 5
    
    # Check wealth accumulation
    assert wealth.iloc[0] == 0.0  # Initial wealth
    assert wealth.iloc[1] == 10.0  # After MINT
    assert wealth.iloc[3] == 15.0  # After BURN

def test_feature_computation(sample_config, sample_transactions):
    """Test full feature computation."""
    engine = FeatureEngine(sample_config)
    features = engine.compute_features(sample_transactions)
    
    # Check basic properties
    assert len(features) == 5  # First 5 timestamps
    assert all(col in features.columns for col in 
              ['ewma_volume', 'center_bucket', 'wealth', 't_T', 'pool_price'])
    
    # Check specific features
    assert np.isclose(features['pool_price'].iloc[0], 1.0, rtol=1e-5)
    assert features['t_T'].iloc[0] == 0.0
    assert features['t_T'].iloc[-1] == 1.0

def test_feature_normalization(sample_config, sample_transactions):
    """Test feature normalization."""
    engine = FeatureEngine(sample_config)
    features = engine.compute_features(sample_transactions)
    normalized = engine.normalize_features(features)
    
    # Check that t_T is unchanged
    assert np.array_equal(normalized['t_T'], features['t_T'])
    
    # Check that other features are normalized
    for col in normalized.columns:
        if col != 't_T':
            assert abs(normalized[col].mean()) < 1e-10  # Close to 0
            assert abs(normalized[col].std() - 1.0) < 1e-10 or normalized[col].std() < 1e-5 