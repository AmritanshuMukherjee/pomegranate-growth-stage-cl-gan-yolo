import yaml
from pathlib import Path


def load_yaml(config_path):
    """
    Load a YAML configuration file.

    Args:
        config_path (str or Path): Path to the YAML file

    Returns:
        dict: Parsed YAML content
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
