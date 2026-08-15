import os
import yaml
from pathlib import Path
from typing import Any, Dict

# Paths 

# Absolute path to the project root (two levels up from this file:
#   utils/file_utils.py  →  utils/  →  project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Canonical location of the main YAML configuration file
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yml"


# Loaders 

def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load any YAML file and return its contents as a dict.

    Args:
        file_path (str): Absolute or relative path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML contents.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> Dict[str, Any]:
    """
    Load and return the project-wide configuration from ``config/config.yml``.

    The path is resolved relative to the project root, so this works regardless
    of the current working directory the script is launched from.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If ``config/config.yml`` does not exist.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found at expected location: {CONFIG_PATH}"
        )
    return load_yaml(str(CONFIG_PATH))
