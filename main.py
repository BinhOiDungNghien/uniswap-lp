"""
Main entry point for ODRA strategy
"""

import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple

from features.feature_engine import FeatureEngine
from strategy.strategy_simulator import StrategySimulator
from model.network import ODRANetwork
from model.trainer import ODRATrainer
from model.evaluator import ODRAEvaluator
from model.loss import EntropyRegularizedLoss, TransactionCostLoss
from utils.logging_utils import setup_logging

# ... existing code ... 