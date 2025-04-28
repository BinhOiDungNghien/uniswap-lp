"""
Tick Math Utilities for Uniswap V3 Calculations
Based on: https://github.com/Uniswap/v3-core/blob/main/contracts/libraries/TickMath.sol
"""

import math
from typing import Tuple

# Constants
MIN_TICK = -887272
MAX_TICK = 887272
MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

def tick_to_sqrt_price_x96(tick: int) -> int:
    """Convert a tick to its corresponding sqrtPriceX96."""
    if tick < MIN_TICK or tick > MAX_TICK:
        raise ValueError(f"Tick {tick} out of bounds")
        
    abs_tick = abs(tick)
    ratio = 1.0001 ** (abs_tick / 2)
    
    if tick >= 0:
        sqrt_price_x96 = int(ratio * (2 ** 96))
    else:
        sqrt_price_x96 = int((2 ** 96) / ratio)
        
    return max(MIN_SQRT_RATIO, min(MAX_SQRT_RATIO, sqrt_price_x96))

def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
    """Convert sqrtPriceX96 to its corresponding tick."""
    if sqrt_price_x96 < MIN_SQRT_RATIO or sqrt_price_x96 > MAX_SQRT_RATIO:
        raise ValueError(f"sqrtPriceX96 {sqrt_price_x96} out of bounds")
        
    ratio = sqrt_price_x96 / (2 ** 96)
    if ratio >= 1:
        tick = int(math.log(ratio, 1.0001) * 2)
    else:
        tick = -int(math.log(1/ratio, 1.0001) * 2)
        
    return max(MIN_TICK, min(MAX_TICK, tick))

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