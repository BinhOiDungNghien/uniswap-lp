"""
Test cases for price and tick calculations in Uniswap v3.
"""

import pytest
import numpy as np
from odra_strategy.strategy.tick_math import (
    tick_to_sqrt_price_x96,
    sqrt_price_x96_to_tick,
    price_to_tick,
    tick_to_price,
    get_tick_spacing,
    MIN_TICK,
    MAX_TICK
)

def test_tick_to_price_basic():
    """Test basic tick to price conversions."""
    assert abs(tick_to_price(0) - 1.0) < 1e-10
    assert abs(tick_to_price(6932) - 2.0) < 1e-4
    assert abs(tick_to_price(-6932) - 0.5) < 1e-4

def test_price_to_tick_basic():
    """Test basic price to tick conversions."""
    assert price_to_tick(1.0) == 0
    assert abs(price_to_tick(2.0) - 6932) < 2  # Allow 1-2 tick deviation due to rounding
    assert abs(price_to_tick(0.5) + 6932) < 2

def test_tick_price_roundtrip():
    """Test that converting from tick to price and back preserves the tick."""
    test_ticks = [-6932, 0, 6932, 50000, -50000]
    for tick in test_ticks:
        price = tick_to_price(tick)
        recovered_tick = price_to_tick(price)
        assert abs(tick - recovered_tick) <= 1  # Allow 1 tick deviation due to rounding

def test_edge_cases():
    """Test edge cases and invalid inputs."""
    # Test NaN handling
    assert tick_to_price(np.nan) == 1.0
    
    # Test bounds
    assert tick_to_price(MIN_TICK) > 0
    assert tick_to_price(MAX_TICK) < float('inf')
    
    # Test invalid prices
    assert price_to_tick(-1.0) == 0
    assert price_to_tick(0) == 0

def test_sqrt_price_conversion():
    """Test conversion between tick and sqrtPriceX96."""
    test_ticks = [-6932, 0, 6932]
    for tick in test_ticks:
        sqrt_price = tick_to_sqrt_price_x96(tick)
        recovered_tick = sqrt_price_x96_to_tick(sqrt_price)
        assert abs(tick - recovered_tick) <= 1

def test_real_world_prices():
    """Test with real-world price examples from ETH/USDC pool."""
    # ETH price around $2000
    eth_price = 2000.0
    eth_tick = price_to_tick(eth_price)
    recovered_price = tick_to_price(eth_tick)
    assert abs(recovered_price - eth_price) / eth_price < 0.001  # Within 0.1%

def test_tick_spacing():
    """Test tick spacing for different fee tiers."""
    assert get_tick_spacing(100) == 1    # 0.01%
    assert get_tick_spacing(500) == 10   # 0.05%
    assert get_tick_spacing(3000) == 60  # 0.3%
    assert get_tick_spacing(10000) == 200  # 1%
    assert get_tick_spacing(1234) == 60  # Default to 0.3%

def test_price_monotonicity():
    """Test that prices increase monotonically with tick."""
    ticks = range(-50000, 50001, 1000)
    prices = [tick_to_price(tick) for tick in ticks]
    
    # Check that each price is larger than the previous
    for i in range(1, len(prices)):
        assert prices[i] > prices[i-1]

def test_batch_processing():
    """Test processing multiple ticks/prices in batch."""
    ticks = np.array([-6932, 0, 6932])
    prices = np.array([tick_to_price(t) for t in ticks])
    recovered_ticks = np.array([price_to_tick(p) for p in prices])
    
    np.testing.assert_array_almost_equal(ticks, recovered_ticks, decimal=0) 