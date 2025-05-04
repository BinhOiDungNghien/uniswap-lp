"""
Strategy Simulator for ODRA
"""

import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Optional, List, Tuple, Union
import uuid
import logging
from .tick_math import (
    tick_to_sqrt_price_x96,
    sqrt_price_x96_to_tick,
    get_liquidity_for_amounts,
    get_amounts_for_liquidity,
    calculate_fee_growth_inside
)
from ..utils.logging_utils import log_simulation_step, plot_price_ranges

logger = logging.getLogger(__name__)

class Position:
    """Represents a liquidity position in Uniswap V3."""
    def __init__(self, 
                 position_id: str,
                 tick_lower: int,
                 tick_upper: int,
                 liquidity: Decimal,
                 amount0: Decimal,
                 amount1: Decimal):
        self.position_id = position_id
        self.tick_lower = tick_lower
        self.tick_upper = tick_upper
        self.liquidity = liquidity
        self.amount0 = amount0
        self.amount1 = amount1
        self.fees_earned = Decimal(0)
        self.fee_growth_inside_last = Decimal(0)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary representation."""
        return {
            'position_id': self.position_id,
            'tick_lower': self.tick_lower,
            'tick_upper': self.tick_upper,
            'liquidity': float(self.liquidity),
            'amount0': float(self.amount0),
            'amount1': float(self.amount1),
            'fees_earned': float(self.fees_earned)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create position from dictionary representation."""
        position = cls(
            position_id=data['position_id'],
            tick_lower=data['tick_lower'],
            tick_upper=data['tick_upper'],
            liquidity=Decimal(data['liquidity']),
            amount0=Decimal(data['amount0']),
            amount1=Decimal(data['amount1'])
        )
        position.fees_earned = Decimal(data.get('fees_earned', 0))
        return position

class StrategySimulator:
    """
    Simulates strategy execution for ODRA.
    Handles rebalancing, position management, fee collection, and wealth tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simulator with configuration.
        
        Args:
            config: Dictionary containing:
                - tau: Rebalancing threshold in buckets
                - tick_spacing: Tick spacing for the pool
                - initial_wealth: Starting wealth
                - min_position_size: Minimum position size
                - fee_tier: Pool fee tier (e.g. 0.3% = 3000)
        """
        self.config = config
        self.tau = config['tau']
        self.tick_spacing = config['tick_spacing']
        self.initial_wealth = Decimal(config.get('initial_wealth', 1.0))
        self.min_position_size = Decimal(config.get('min_position_size', 0.1))
        self.fee_tier = config.get('fee_tier', 3000)
        
        # Token decimals
        self.token0_decimals = config.get('token0_decimals', 18)
        self.token1_decimals = config.get('token1_decimals', 18)
        
        # State variables
        self.wealth = self.initial_wealth
        self.current_positions: Dict[str, Position] = {}
        self.current_price = None
        self.current_tx = None
        self.last_center_bucket = None
        self.fees_collected = Decimal(0)
        
        # Fee growth tracking
        self.fee_growth_global = Decimal(0)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _adjust_for_decimals(self, amount: float, token_decimals: int) -> Decimal:
        """Adjust token amount for decimals."""
        return Decimal(amount) * Decimal(10) ** -token_decimals

    def _get_current_bucket(self) -> int:
        """Get current bucket based on current price or tick."""
        if self.current_tx is None:
            return 0
            
        if 'current_tick' in self.current_tx and not pd.isna(self.current_tx['current_tick']):
            tick = float(self.current_tx['current_tick'])
            bucket = int(tick // self.tick_spacing)
            logger.debug(f"Current bucket from tick {tick}: {bucket}")
            return bucket
        elif self.current_price is not None and not pd.isna(self.current_price):
            try:
                tick = sqrt_price_x96_to_tick(int(self.current_price))
                bucket = tick // self.tick_spacing
                logger.debug(f"Current bucket from price {self.current_price}: {bucket}")
                return bucket
            except (ValueError, TypeError):
                logger.warning(f"Could not convert price {self.current_price} to tick")
                return 0
        return 0

    def needs_rebalance(self, current_bucket: Optional[int] = None) -> bool:
        """Check if rebalancing is needed based on price/tick movement."""
        if self.last_center_bucket is None:
            logger.debug("First rebalance needed (no last center bucket)")
            return True
            
        bucket_to_check = current_bucket if current_bucket is not None else self._get_current_bucket()
        distance = abs(bucket_to_check - self.last_center_bucket)
        logger.debug(f"Bucket distance: {distance} (current: {bucket_to_check}, last: {self.last_center_bucket}, tau: {self.tau})")
        return distance >= self.tau

    def _get_bucket_range(self) -> Tuple[int, int]:
        """Get range of buckets for current center."""
        center = self._get_current_bucket()
        return center - self.tau, center + self.tau

    def _calculate_positions(self, allocation: np.ndarray, bucket_range: Tuple[int, int]) -> np.ndarray:
        """Calculate new positions based on allocation vector.
        
        Args:
            allocation: Vector of allocation weights
            bucket_range: (min_bucket, max_bucket) tuple
            
        Returns:
            Array of position amounts
        """
        min_bucket, max_bucket = bucket_range
        n_buckets = max_bucket - min_bucket + 1
        
        if len(allocation) != n_buckets + 2:  # +2 for outside pool positions
            raise ValueError(f"Allocation vector length {len(allocation)} does not match number of buckets {n_buckets + 2}")
            
        # Convert allocation weights to actual position amounts
        # Skip first and last elements which represent outside pool positions
        position_amounts = np.zeros(n_buckets)
        for i in range(n_buckets):
            position_amounts[i] = allocation[i + 1] * self.total_wealth
            
        return position_amounts

    def _create_position(self, 
                        amount: float, 
                        tick_lower: int, 
                        tick_upper: int) -> Position:
        """Create a new position with given parameters."""
        position_id = str(uuid.uuid4())
        
        # Calculate optimal token amounts based on current price
        if self.current_price is not None:
            sqrt_price_x96 = int(self.current_price)
            liquidity = get_liquidity_for_amounts(
                sqrt_price_x96,
                tick_lower,
                tick_upper,
                amount/2,  # Split amount for initial estimation
                amount/2
            )
            amount0, amount1 = get_amounts_for_liquidity(
                sqrt_price_x96,
                tick_lower,
                tick_upper,
                liquidity
            )
        else:
            # Fallback to equal split if no price available
            liquidity = amount
            amount0 = amount1 = amount/2
            
        return Position(
            position_id=position_id,
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            liquidity=liquidity,
            amount0=amount0,
            amount1=amount1
        )

    def _allocate_positions(self, weights: List[float]) -> Dict[str, Position]:
        """Convert allocation weights to positions."""
        positions = {}
        current_tick = self._get_current_bucket() * self.tick_spacing
        
        # Convert weights to Decimal
        weights = [Decimal(str(w)) for w in weights]  # Use str() for exact conversion
        
        if len(weights) == 2:  # Simple in/out pool allocation
            if weights[0] > 0:  # In-pool position
                tick_lower = current_tick - self.tick_spacing
                tick_upper = current_tick + self.tick_spacing
                position = self._create_position(
                    amount=weights[0] * self.wealth,
                    tick_lower=tick_lower,
                    tick_upper=tick_upper
                )
                positions[position.position_id] = position
                
            if weights[1] > 0:  # Out-pool position
                tick_lower = current_tick - 100 * self.tick_spacing
                tick_upper = current_tick + 100 * self.tick_spacing
                position = self._create_position(
                    amount=weights[1] * self.wealth,
                    tick_lower=tick_lower,
                    tick_upper=tick_upper
                )
                positions[position.position_id] = position
                
        else:  # Full bucket allocation
            base_tick = current_tick - (len(weights) // 2) * self.tick_spacing
            for i, weight in enumerate(weights):
                if weight > 0:
                    tick_lower = base_tick + i * self.tick_spacing
                    tick_upper = tick_lower + self.tick_spacing
                    position = self._create_position(
                        amount=weight * self.wealth,
                        tick_lower=tick_lower,
                        tick_upper=tick_upper
                    )
                    positions[position.position_id] = position
                    
        return positions

    def _validate_sqrt_price(self, price_val: Union[float, str]) -> Optional[Decimal]:
        """
        Validate and convert sqrtPriceX96 to Decimal.
        
        Args:
            price_val: The price value to validate
            
        Returns:
            Optional[Decimal]: Valid Decimal price or None if invalid
        """
        try:
            # Convert to string first for exact decimal conversion
            price_str = str(float(price_val))
            price_dec = Decimal(price_str)
            
            # Check for valid price (must be positive)
            if price_dec <= 0:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Invalid price value (<=0): {price_dec}")
                return None
            
            return price_dec
        except (ValueError, TypeError, InvalidOperation) as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Error converting price {price_val} to Decimal: {e}")
            return None

    def _calculate_swap_fees(self, tx: pd.Series) -> Decimal:
        """Calculate fees from swap following Uniswap V3 formula."""
        try:
            # Adjust amounts for decimals
            amount0 = self._adjust_for_decimals(abs(float(tx['amount0_adjusted'])), self.token0_decimals)
            amount1 = self._adjust_for_decimals(abs(float(tx['amount1_adjusted'])), self.token1_decimals)
            
            # Calculate fee based on token being swapped in
            if float(tx['amount0_adjusted']) < 0:  # Swap token0 -> token1
                fee_amount = amount0 * Decimal(self.fee_tier) / Decimal(1_000_000)
            else:  # Swap token1 -> token0
                # Try both camelCase and lowercase versions of the column name
                sqrt_price = None
                for col_name in ['sqrtPriceX96', 'sqrtpricex96']:
                    if col_name in tx and not pd.isna(tx[col_name]):
                        sqrt_price = tx[col_name]
                        break
                
                # If no valid sqrt price, use amount0 for fee calculation
                if sqrt_price is None:
                    fee_amount = amount0 * Decimal(self.fee_tier) / Decimal(1_000_000)
                else:
                    price_dec = self._validate_sqrt_price(sqrt_price)
                    if price_dec is None:
                        # Fallback to amount0-based fee if price is invalid
                        fee_amount = amount0 * Decimal(self.fee_tier) / Decimal(1_000_000)
                    else:
                        current_price = price_dec / (Decimal(2**96))**2
                        fee_amount = amount1 * current_price * Decimal(self.fee_tier) / Decimal(1_000_000)
            
            # Update fee growth global if total liquidity is available
            if 'total_liquidity' in tx and not pd.isna(tx['total_liquidity']):
                try:
                    total_liquidity = Decimal(str(float(tx['total_liquidity'])))
                    if total_liquidity > 0:
                        self.fee_growth_global += fee_amount / total_liquidity
                    elif logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Invalid total_liquidity (<=0): {total_liquidity}")
                except (ValueError, TypeError, InvalidOperation) as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Error updating fee growth global: {e}")
            
            return fee_amount
        
        except Exception as e:
            logger.error(f"Error calculating swap fees: {e}")
            return Decimal(0)

    def process_collect(self, tx: pd.Series) -> Decimal:
        """Process COLLECT transaction and return value in terms of wealth."""
        amount0 = self._adjust_for_decimals(float(tx['amount0_adjusted']), self.token0_decimals)
        amount1 = self._adjust_for_decimals(float(tx['amount1_adjusted']), self.token1_decimals)
        
        if self.current_price is not None:
            price = Decimal(self.current_price) / (Decimal(2**96))**2
            value = amount0 + (amount1 * price)
            return value
        return Decimal(0)

    def compute_bucket_range(self, center_bucket: int) -> Tuple[int, int]:
        """
        Compute range of buckets to allocate liquidity.
        
        Args:
            center_bucket: Center bucket for range
            
        Returns:
            Tuple[int, int]: (lower_bucket, upper_bucket)
        """
        lower = center_bucket - self.tau
        upper = center_bucket + self.tau
        return lower, upper
        
    def _compute_swap_fees(self, transaction: pd.Series) -> float:
        """
        Compute fees earned from a swap.
        
        Args:
            transaction: Swap transaction data
            
        Returns:
            float: Fee amount in base currency
        """
        amount = abs(float(transaction['amount0']))
        fee_rate = self.fee_tier / 1_000_000  # Convert from bps
        return amount * fee_rate
        
    def allocate_liquidity(self, 
                          center_bucket: int, 
                          allocation_weights: List[float], 
                          current_price: float) -> Dict[str, Dict[str, Any]]:
        """
        Allocate liquidity across buckets.
        
        Args:
            center_bucket: Center bucket for allocation
            allocation_weights: List of weights for each bucket
            current_price: Current pool price
            
        Returns:
            Dict[str, Dict[str, Any]]: Map of position_id -> position details
        """
        lower, upper = self.compute_bucket_range(center_bucket)
        
        # Normalize weights
        weights = np.array(allocation_weights)
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        
        # Create positions
        positions = {}
        for i, bucket in enumerate(range(lower, upper + 1)):
            if i < len(weights) and weights[i] >= self.min_position_size:
                position_id = str(uuid.uuid4())
                tick_lower = bucket * self.tick_spacing
                tick_upper = (bucket + 1) * self.tick_spacing
                
                positions[position_id] = {
                    'position_id': position_id,
                    'bucket': bucket,
                    'tick_lower': tick_lower,
                    'tick_upper': tick_upper,
                    'allocation': float(weights[i]),
                    'amount0': self.wealth * weights[i] / 2,
                    'amount1': self.wealth * weights[i] / 2
                }
                
        return positions
        
    def _expand_allocation(self, weights: List[float]) -> np.ndarray:
        """Expand 2-element allocation (in/out) to full bucket range."""
        if len(weights) != 2:
            raise ValueError("Initial allocation must be [in_pool, out_pool]")
            
        # For initial allocation, put everything in center bucket
        expanded = np.zeros(2 * self.tau + 1)
        if weights[0] > 0:  # If allocating to pool
            center_idx = self.tau  # Middle bucket
            expanded[center_idx] = weights[0]
        return expanded

    def simulate_step(self, tx: pd.Series, allocation_weights: Optional[List[float]] = None) -> dict:
        """Simulate a single step of the strategy.
        
        Args:
            tx: Transaction data
            allocation_weights: Optional allocation weights for rebalancing
            
        Returns:
            Dictionary containing step results
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Processing transaction: {tx.to_dict()}")
        
        self.current_tx = tx
        
        # Update current price if available - try both camelCase and lowercase versions
        sqrt_price = None
        for col_name in ['sqrtPriceX96', 'sqrtpricex96']:
            if col_name in tx and not pd.isna(tx[col_name]):
                sqrt_price = tx[col_name]
                break
                
        if sqrt_price is not None:
            price_dec = self._validate_sqrt_price(sqrt_price)
            if price_dec is not None:
                self.current_price = price_dec
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Updated price to: {self.current_price}")
            else:
                # Keep previous price if new price is invalid
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Could not validate price, keeping previous price")
        
        # Process transaction - handle different column names for transaction type
        tx_type = None
        for col_name in ['tx_type', 'type', 'transaction_type']:
            if col_name in tx:
                tx_type = tx[col_name]
                break
                
        if tx_type is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Transaction type column not found. Using default 'SWAP'.")
            tx_type = 'SWAP'
            
        # Process transaction based on type
        if tx_type == 'SWAP':
            # Calculate and collect fees
            fees = self._calculate_swap_fees(tx)
            self.fees_collected += fees
            self.wealth += fees
            
        elif tx_type == 'COLLECT':
            collected_value = self.process_collect(tx)
            self.wealth += collected_value
            
        elif tx_type in ['MINT', 'BURN']:
            try:
                # Update position amounts with decimal adjustment
                amount0 = self._adjust_for_decimals(
                    abs(float(tx['amount0_adjusted'])) if not pd.isna(tx['amount0_adjusted']) else 0,
                    self.token0_decimals
                )
                amount1 = self._adjust_for_decimals(
                    abs(float(tx['amount1_adjusted'])) if not pd.isna(tx['amount1_adjusted']) else 0,
                    self.token1_decimals
                )
                
                # Calculate value based on current price
                if self.current_price is not None:
                    price_scale = self.current_price / (Decimal(2**96))**2
                    position_value = amount0 + (amount1 * price_scale)
                else:
                    # Fallback: use simple sum if price not available
                    position_value = amount0 + amount1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("No valid price available, using simple sum for position value")
                
                if tx_type == 'MINT':
                    self.wealth -= position_value
                else:  # BURN
                    self.wealth += position_value
                
            except Exception as e:
                logger.error(f"Error processing {tx_type} transaction: {e}")
                return {'wealth': float(self.wealth)}  # Return current wealth even on error
        
        # Check if rebalancing is needed
        current_bucket = self._get_current_bucket()
        if self.needs_rebalance(current_bucket):
            if allocation_weights is not None:
                # Create new positions based on allocation
                self.current_positions = self._allocate_positions(allocation_weights)
                self.last_center_bucket = current_bucket
                
        # Return step results
        return {
            'wealth': float(self.wealth),
            'fees_collected': float(self.fees_collected),
            'active_positions': len(self.current_positions),
            'current_bucket': current_bucket
        }

    def get_position_stats(self) -> Dict[str, Any]:
        """Get statistics about current positions."""
        current_tick = self._get_current_bucket() * self.tick_spacing
        return {
            'total_positions': len(self.current_positions),
            'active_positions': sum(
                1 for pos in self.current_positions.values()
                if pos.tick_lower <= current_tick <= pos.tick_upper
            ),
            'total_liquidity': sum(pos.liquidity for pos in self.current_positions.values()),
            'total_fees_earned': sum(pos.fees_earned for pos in self.current_positions.values())
        } 