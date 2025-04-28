# ODRA Strategy: Optimal Dynamic Reset Allocation for Uniswap v3

This repository implements the ODRA (Optimal Dynamic Reset Allocation) strategy for Uniswap v3 liquidity providers, using real tick-level data and deep reinforcement learning.

## Overview

ODRA is a dynamic liquidity provision strategy that:
- Optimizes position ranges based on price movement and volume patterns
- Uses deep reinforcement learning to learn optimal rebalancing policies
- Incorporates real tick-level data from Uniswap v3 pools
- Considers transaction costs and competitive LP behavior

## Project Structure

```
odra_strategy/
│
├── config/                   # Configuration files
├── data/                    # Data handling
│   ├── raw/                 # Raw tick-level data
│   ├── processed/           # Processed datasets
│   └── utils/              # Data utilities
├── features/               # Feature engineering
├── strategy/              # Core strategy logic
├── model/                 # Neural network models
├── simulator/             # Tick-level simulator
├── outputs/               # Results & artifacts
├── notebooks/             # Analysis notebooks
└── tests/                 # Test suite
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/odra-strategy.git
cd odra-strategy
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare data:
   - Place tick-level data files in `data/raw/`
   - Run data processing:
   ```bash
   python -m odra_strategy.main --config config/odra_config.yml --mode process
   ```

2. Train model:
```bash
python -m odra_strategy.main --config config/odra_config.yml --mode train
```

3. Evaluate strategy:
```bash
python -m odra_strategy.main --config config/odra_config.yml --mode evaluate
```

## Configuration

Key parameters in `config/odra_config.yml`:
- `tau`: Reset threshold (in ticks)
- `alpha_ewma`: EWMA decay factor
- `tick_spacing`: Pool tick spacing
- Model architecture parameters
- Training hyperparameters

## Features

- **Tick-Level Data Processing**: Handles SWAP, MINT, BURN, and COLLECT events
- **Feature Engineering**: Computes EWMA volume, center bucket, wealth metrics
- **Neural Network**: 5-layer architecture with ReLU activation
- **Strategy Simulation**: Realistic LP behavior modeling
- **Performance Evaluation**: CARA utility-based assessment

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uniswap v3 team for the protocol design
- Academic research on automated market making
- Open-source DeFi analytics tools

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{odra2024,
  author = {Your Name},
  title = {ODRA: Optimal Dynamic Reset Allocation for Uniswap v3},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/uniswap-lp}
}
```

## Project Structure

```
odra_strategy/
├── config/
│   └── odra_config.yml     # Configuration parameters
├── data/
│   ├── raw/                # Tick-level data
│   └── data_loader.py      # Data loading utilities
├── features/
│   └── feature_engine.py   # Feature computation
├── model/
│   ├── network.py          # Neural network architecture
│   └── trainer.py          # Training logic
├── strategy/
│   └── strategy_simulator.py # Strategy simulation
└── notebooks/
    └── train_odra_colab.ipynb # Colab training notebook
```

## Requirements

See `requirements.txt` for full list of dependencies.

## Usage

1. Data Processing:
```bash
python main.py process-data
```

2. Training:
```bash
python main.py train
```

3. Evaluation:
```bash
python main.py evaluate
```

## Development

- Follow the coding guidelines in `.cursor/rules/uniswap.mdc`
- Run tests: `pytest tests/`
- Use Jupyter notebooks in `notebooks/` for exploration

## License

MIT License 