import argparse
import json
from typing import Dict

from trainer import train


def _parse_override_pairs(pairs) -> Dict[str, object]:
    if not pairs:
        return {}

    def _convert(value: str):
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower in {"null", "none"}:
            return None

        try:
            if value.startswith("[") or value.startswith("{"):
                return json.loads(value)
        except json.JSONDecodeError:
            pass

        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        return value

    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' is not in key=value format")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override key in '{pair}'")
        overrides[key] = _convert(value.strip())
    return overrides


def main():
    namespace = setup_parser().parse_args()
    config_path = namespace.config
    cfg = load_json(config_path)

    cli_args = vars(namespace)
    override_pairs = cli_args.pop("override", None)

    for key, value in cli_args.items():
        if key in {"config", "override"}:
            continue
        cfg[key] = value

    overrides = _parse_override_pairs(override_pairs)
    cfg.update(overrides)
    cfg["config"] = config_path

    train(cfg)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument(
        '--override',
        nargs='+',
        help='Override config entries via key=value pairs. '
             'Example: --override optimizer_type=sam sam_rho=0.05 flat_eval=true',
    )
    return parser

if __name__ == '__main__':
    main()
