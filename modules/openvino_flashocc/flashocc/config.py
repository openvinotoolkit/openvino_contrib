import runpy
from pathlib import Path


def load_config(path):
    cfg_path = Path(path)
    namespace = runpy.run_path(str(cfg_path))
    if "model" not in namespace:
        raise KeyError(f"No 'model' entry found in config: {cfg_path}")
    return namespace
