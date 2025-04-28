import os
import yaml
from pathlib import Path

def is_running_on_kaggle():
    """Check if the code is running on Kaggle."""
    return os.path.exists('/kaggle')

def adjust_paths_for_kaggle(config_path):
    """Adjust paths in config file for Kaggle environment."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if is_running_on_kaggle():
        # Adjust paths for Kaggle
        kaggle_base = '/kaggle/working/uniswap-lp/odra_strategy'
        
        # Update data paths
        config['data']['raw_path'] = os.path.join(kaggle_base, 'data/raw')
        config['data']['processed_path'] = os.path.join(kaggle_base, 'data/processed/odra_dataset.pkl')
        
        # Update output paths
        config['logging']['log_dir'] = os.path.join(kaggle_base, 'outputs/logs')
        config['logging']['model_dir'] = os.path.join(kaggle_base, 'outputs/models')
        config['logging']['plot_dir'] = os.path.join(kaggle_base, 'outputs/plots')
        
        # Create necessary directories
        for path in [config['data']['raw_path'], 
                    os.path.dirname(config['data']['processed_path']),
                    config['logging']['log_dir'],
                    config['logging']['model_dir'],
                    config['logging']['plot_dir']]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    return config

def setup_kaggle_environment():
    """Setup the Kaggle environment with necessary directories and configurations."""
    if not is_running_on_kaggle():
        return
    
    # Create base directories
    kaggle_base = '/kaggle/working/uniswap-lp/odra_strategy'
    for dir_name in ['data/raw', 'data/processed', 'outputs/logs', 'outputs/models', 'outputs/plots']:
        Path(os.path.join(kaggle_base, dir_name)).mkdir(parents=True, exist_ok=True)
    
    print("Kaggle environment setup completed.") 