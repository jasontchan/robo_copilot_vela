import importlib.util
from pathlib import Path

from diffusion_policy.utils import DiffusionPolicyConfig


def load_config(config_path: str) -> DiffusionPolicyConfig:
    """
    Load a configuration file given its file path.

    Args:
        config_path (str): File path to the config file.

    Returns:
        DiffusionPolicyConfig: The configuration object defined in the config file.
    """
    config_path_obj: Path = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_path_obj.stem, str(config_path_obj))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config
