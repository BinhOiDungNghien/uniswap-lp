"""
Strategy simulator for ODRA
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import uuid
import numpy as np
import pandas as pd
from .tick_math import (
    tick_to_sqrt_price_x96,
    sqrt_price_x96_to_tick,
    get_liquidity_for_amounts,
    get_amounts_for_liquidity
)
from utils.logging_utils import log_simulation_step, plot_price_ranges

logger = logging.getLogger(__name__)

# ... existing code ... 