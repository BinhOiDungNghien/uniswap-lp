"""
Setup file for ODRA strategy package
"""

from setuptools import setup, find_packages

setup(
    name="odra_strategy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        "seaborn",
        "pyyaml",
    ],
    python_requires=">=3.8",
    author="Le Ngoc Binh",
    author_email="lengocbinh2001@gmail.com",
    description="Optimal Dynamic Reset Allocation Strategy for Uniswap v3",
    keywords="defi, uniswap, liquidity-provision, reinforcement-learning",
) 