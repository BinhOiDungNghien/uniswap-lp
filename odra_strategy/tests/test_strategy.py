"""
Tests for Strategy Simulator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from odra_strategy.strategy.strategy_simulator import StrategySimulator

@pytest.fixture
def sample_config():
    """Basic configuration for testing."""
    return {
        'tau': 3,
        'tick_spacing': 60,
        'initial_wealth': 1000.0,
        'min_position_size': 0.1,
        'fee_tier': 3000
    }

@pytest.fixture
def sample_transaction():
    """Create a sample SWAP transaction."""
    return pd.Series({
        'type': 'SWAP',
        'tick': 60,
        'amount0': 10.0,
        'amount1': -11.0,
        'sqrtPriceX96': '79228162514264337593543950336'  # Price ≈ 1.0
    })

@pytest.fixture
def sample_allocation():
    """Create sample allocation weights."""
    return np.array([0.0, 0.0, 0.3, 0.4, 0.3, 0.0, 0.0, 0.0])

def test_rebalance_trigger(sample_config):
    """Test rebalancing trigger logic."""
    simulator = StrategySimulator(sample_config)
    
    # First check should always trigger rebalance
    assert simulator.needs_rebalance(0)
    assert simulator.last_center_bucket is None
    
    # Set initial bucket
    simulator.last_center_bucket = 0
    
    # Small movement should not trigger
    assert not simulator.needs_rebalance(2)
    
    # Large movement should trigger
    assert simulator.needs_rebalance(4)

def test_bucket_range(sample_config):
    """Test bucket range computation."""
    simulator = StrategySimulator(sample_config)
    
    # Test center at 0
    lower, upper = simulator.compute_bucket_range(0)
    assert lower == -3
    assert upper == 3
    
    # Test positive center
    lower, upper = simulator.compute_bucket_range(5)
    assert lower == 2
    assert upper == 8
    
    # Test negative center
    lower, upper = simulator.compute_bucket_range(-5)
    assert lower == -8
    assert upper == -2

def test_swap_fee_computation(sample_config, sample_transaction):
    """Test swap fee computation."""
    simulator = StrategySimulator(sample_config)
    fee = simulator._compute_swap_fees(sample_transaction)
    
    # Fee should be amount0 * fee_rate
    expected_fee = abs(float(sample_transaction['amount0'])) * (3000 / 1_000_000)
    assert np.isclose(fee, expected_fee)

def test_liquidity_allocation(sample_config):
    """Test liquidity allocation."""
    simulator = StrategySimulator(sample_config)
    
    # Test with simple allocation
    weights = [0.5, 0.5]
    positions = simulator.allocate_liquidity(
        center_bucket=0,
        allocation_weights=weights,
        current_price=1.0
    )
    
    # Check position properties
    assert len(positions) == 2
    for pos_id, pos in positions.items():
        assert isinstance(pos_id, str)
        assert isinstance(pos, dict)
        assert all(k in pos for k in ['position_id', 'bucket', 'tick_lower', 
                                    'tick_upper', 'allocation', 'amount0', 'amount1'])
        assert pos['allocation'] == 0.5
        assert pos['amount0'] == simulator.wealth * 0.5 / 2
        assert pos['amount1'] == simulator.wealth * 0.5 / 2

def test_initial_step(sample_config, sample_transaction):
    """Test first simulation step."""
    simulator = StrategySimulator(sample_config)
    
    # Initial step with allocation
    result = simulator.simulate_step(
        sample_transaction,
        allocation_weights=[0.5, 0.5]
    )
    
    # Check result properties
    assert result['rebalanced']
    assert len(result['positions']) == 2
    assert result['wealth'] > simulator.initial_wealth  # Should collect fees
    assert result['fees_collected'] > 0

def test_rebalance_step(sample_config, sample_transaction, sample_allocation):
    """Test rebalancing step."""
    simulator = StrategySimulator(sample_config)
    
    # First step to establish position
    first_result = simulator.simulate_step(
        sample_transaction,
        allocation_weights=[0.5, 0.5]
    )
    initial_positions = set(simulator.current_positions.keys())
    
    # Move price to trigger rebalance
    modified_tx = sample_transaction.copy()
    modified_tx['tick'] = 240  # Move 4 tick spacings
    second_result = simulator.simulate_step(
        modified_tx,
        allocation_weights=sample_allocation
    )
    
    # Check rebalancing
    assert second_result['rebalanced']
    new_positions = set(simulator.current_positions.keys())
    assert new_positions != initial_positions
    assert len(second_result['new_positions']) == 3  # Based on sample_allocation

def test_collect_fees(sample_config):
    """Test fee collection."""
    simulator = StrategySimulator(sample_config)
    
    # Create COLLECT transaction
    collect_tx = pd.Series({
        'type': 'COLLECT',
        'amount0': 5.0,
        'amount1': 5.0
    })
    
    # Process COLLECT
    result = simulator.simulate_step(collect_tx)
    
    # Check fee collection
    assert result['fees_collected'] == 10.0
    assert result['wealth'] == simulator.initial_wealth + 10.0 