import json
import os
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    """Load a configuration file that can be JSON or YAML.

    - .json => json.load
    - .yaml/.yml => yaml.safe_load (requires PyYAML)
    Returns a plain dict compatible with existing args usage.
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            return json.load(f)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Reading YAML requires PyYAML. Please install it: pip install pyyaml"
                ) from e
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config extension: {ext}")
