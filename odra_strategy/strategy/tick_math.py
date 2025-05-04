"""
Utility functions for Uniswap v3 tick and price calculations.
Based on Uniswap v3 whitepaper and core contracts.
"""

import math
from decimal import Decimal
import numpy as np
from typing import Tuple

# Constants from Uniswap v3 core
MIN_TICK = -887272
MAX_TICK = 887272
Q96 = Decimal(2**96)

def tick_to_sqrt_price_x96(tick: int) -> Decimal:
    """Convert a tick value to a sqrtPriceX96 value."""
    if tick < MIN_TICK:
        tick = MIN_TICK
    elif tick > MAX_TICK:
        tick = MAX_TICK
        
    absTick = abs(tick)
    
    # Use the mathematical relationship:
    # sqrtPriceX96 = sqrt(1.0001^tick) * 2^96
    ratio = Decimal(math.sqrt(1.0001 ** absTick))
    
    if tick >= 0:
        sqrtPriceX96 = ratio * Q96
    else:
        sqrtPriceX96 = (Decimal(1) / ratio) * Q96
        
    return sqrtPriceX96

def sqrt_price_x96_to_tick(sqrt_price_x96: Decimal) -> int:
    """Convert a sqrtPriceX96 value to the corresponding tick."""
    # Convert to Decimal for precise math
    price = (Decimal(sqrt_price_x96) / Q96) ** 2
    
    # Use the mathematical relationship:
    # tick = log(price) / log(1.0001)
    if price > 0:
        tick = int(math.log(float(price)) / math.log(1.0001))
        return max(MIN_TICK, min(tick, MAX_TICK))
    return 0

def price_to_tick(price):
    """Convert a price value to the nearest tick.
    
    Args:
        price (float): The price to convert
        
    Returns:
        int: The nearest tick value
    """
    if price <= 0:
        return 0
        
    tick = int(math.log(price) / math.log(1.0001))
    return max(MIN_TICK, min(tick, MAX_TICK))

def tick_to_price(tick):
    """Convert a tick value to a price.
    
    Args:
        tick (int): The tick value to convert
        
    Returns:
        float: The price value
    """
    if np.isnan(tick):
        return 1.0
        
    tick = max(MIN_TICK, min(int(tick), MAX_TICK))
    return 1.0001 ** tick

def get_tick_spacing(fee_tier):
    """Get the tick spacing for a given fee tier.
    
    Args:
        fee_tier (int): The fee tier (e.g., 500, 3000, 10000)
        
    Returns:
        int: The tick spacing
    """
    # Standard Uniswap v3 tick spacings
    TICK_SPACINGS = {
        100: 1,    # 0.01%
        500: 10,   # 0.05%
        3000: 60,  # 0.3%
        10000: 200 # 1%
    }
    return TICK_SPACINGS.get(fee_tier, 60)  # Default to 0.3% tier spacing

def get_liquidity_for_amounts(
    sqrt_price_x96: Decimal,
    tick_lower: int,
    tick_upper: int,
    amount0: Decimal,
    amount1: Decimal
) -> Decimal:
    """Calculate liquidity amount for given token amounts and price range following Uniswap V3 formulas."""
    if tick_lower >= tick_upper:
        raise ValueError("Lower tick must be less than upper tick")
        
    sqrt_price_lower = tick_to_sqrt_price_x96(tick_lower)
    sqrt_price_upper = tick_to_sqrt_price_x96(tick_upper)
    
    # Calculate liquidity based on current price position
    if sqrt_price_x96 <= sqrt_price_lower:
        # Only token0 is used
        liquidity = amount0 * (sqrt_price_lower * sqrt_price_upper) / (sqrt_price_upper - sqrt_price_lower)
    elif sqrt_price_x96 >= sqrt_price_upper:
        # Only token1 is used
        liquidity = amount1 / (sqrt_price_upper - sqrt_price_lower)
    else:
        # Both tokens are used
        liquidity0 = amount0 * (sqrt_price_x96 * sqrt_price_upper) / (sqrt_price_upper - sqrt_price_x96)
        liquidity1 = amount1 / (sqrt_price_x96 - sqrt_price_lower)
        liquidity = min(liquidity0, liquidity1)
        
    return liquidity

def get_amounts_for_liquidity(
    sqrt_price_x96: Decimal,
    tick_lower: int,
    tick_upper: int,
    liquidity: Decimal
) -> Tuple[Decimal, Decimal]:
    """Calculate token amounts for given liquidity and price range using Uniswap V3 formulas."""
    if tick_lower >= tick_upper:
        raise ValueError("Lower tick must be less than upper tick")
        
    sqrt_price_lower = tick_to_sqrt_price_x96(tick_lower)
    sqrt_price_upper = tick_to_sqrt_price_x96(tick_upper)
    
    if sqrt_price_x96 <= sqrt_price_lower:
        # Only token0
        amount0 = liquidity * (sqrt_price_upper - sqrt_price_lower) / (sqrt_price_lower * sqrt_price_upper)
        amount1 = Decimal(0)
    elif sqrt_price_x96 >= sqrt_price_upper:
        # Only token1
        amount0 = Decimal(0)
        amount1 = liquidity * (sqrt_price_upper - sqrt_price_lower)
    else:
        # Both tokens
        amount0 = liquidity * (sqrt_price_upper - sqrt_price_x96) / (sqrt_price_x96 * sqrt_price_upper)
        amount1 = liquidity * (sqrt_price_x96 - sqrt_price_lower)
        
    return amount0, amount1

def calculate_fee_growth_inside(
    tick_lower: int,
    tick_upper: int,
    current_tick: int,
    fee_growth_global: Decimal,
    fee_growth_below_lower: Decimal,
    fee_growth_above_upper: Decimal
) -> Decimal:
    """Calculate fees accumulated inside a specific range."""
    if current_tick < tick_lower:
        return fee_growth_global - fee_growth_below_lower
    elif current_tick >= tick_upper:
        return fee_growth_global - fee_growth_above_upper
    else:
        return fee_growth_global - fee_growth_below_lower - fee_growth_above_upper 