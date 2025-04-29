"""
Utility functions for Uniswap v3 tick and price calculations.
Based on Uniswap v3 whitepaper and core contracts.
"""

import math
import numpy as np
from typing import Tuple

# Constants from Uniswap v3 core
MIN_TICK = -887272
MAX_TICK = 887272
Q96 = 2**96

def tick_to_sqrt_price_x96(tick):
    """Convert a tick value to a sqrtPriceX96 value.
    
    Args:
        tick (int): The tick value to convert
        
    Returns:
        int: The sqrtPriceX96 value
    """
    if tick < MIN_TICK:
        tick = MIN_TICK
    elif tick > MAX_TICK:
        tick = MAX_TICK
        
    absTick = abs(tick)
    
    # Use the mathematical relationship:
    # sqrtPriceX96 = sqrt(1.0001^tick) * 2^96
    ratio = math.sqrt(1.0001 ** absTick)
    
    if tick >= 0:
        sqrtPriceX96 = int(ratio * Q96)
    else:
        sqrtPriceX96 = int((1.0 / ratio) * Q96)
        
    return sqrtPriceX96

def sqrt_price_x96_to_tick(sqrt_price_x96):
    """Convert a sqrtPriceX96 value to the corresponding tick.
    
    Args:
        sqrt_price_x96 (int): The sqrtPriceX96 value to convert
        
    Returns:
        int: The tick value
    """
    # Convert to float for easier math
    price = (sqrt_price_x96 / Q96) ** 2
    
    # Use the mathematical relationship:
    # tick = log(price) / log(1.0001)
    if price > 0:
        tick = int(math.log(price) / math.log(1.0001))
        
        # Ensure tick is within bounds
        tick = max(MIN_TICK, min(tick, MAX_TICK))
        return tick
    else:
        return 0  # Default to 0 for invalid prices

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
    sqrt_price_x96: int,
    tick_lower: int,
    tick_upper: int,
    amount0: float,
    amount1: float
) -> float:
    """Calculate liquidity amount for given token amounts and price range."""
    if tick_lower >= tick_upper:
        raise ValueError("Lower tick must be less than upper tick")
        
    sqrt_price_lower = tick_to_sqrt_price_x96(tick_lower)
    sqrt_price_upper = tick_to_sqrt_price_x96(tick_upper)
    
    # Calculate liquidity from both tokens
    if sqrt_price_x96 <= sqrt_price_lower:
        # Only token0 is used
        liquidity = amount0 * sqrt_price_lower
    elif sqrt_price_x96 >= sqrt_price_upper:
        # Only token1 is used
        liquidity = amount1 / sqrt_price_upper
    else:
        # Both tokens are used
        liquidity0 = amount0 * sqrt_price_x96
        liquidity1 = amount1 / sqrt_price_x96
        liquidity = min(liquidity0, liquidity1)
        
    return liquidity

def get_amounts_for_liquidity(
    sqrt_price_x96: int,
    tick_lower: int,
    tick_upper: int,
    liquidity: float
) -> Tuple[float, float]:
    """Calculate token amounts for given liquidity and price range."""
    if tick_lower >= tick_upper:
        raise ValueError("Lower tick must be less than upper tick")
        
    sqrt_price_lower = tick_to_sqrt_price_x96(tick_lower)
    sqrt_price_upper = tick_to_sqrt_price_x96(tick_upper)
    
    if sqrt_price_x96 <= sqrt_price_lower:
        # Only token0
        amount0 = liquidity / sqrt_price_lower
        amount1 = 0
    elif sqrt_price_x96 >= sqrt_price_upper:
        # Only token1
        amount0 = 0
        amount1 = liquidity * sqrt_price_upper
    else:
        # Both tokens
        amount0 = liquidity / sqrt_price_x96
        amount1 = liquidity * sqrt_price_x96
        
    return amount0, amount1 