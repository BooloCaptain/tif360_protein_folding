import os
import argparse
import yaml


def load_config(path=None):
    """Load a YAML config file.

    If `path` is None, checks the CONFIG_NAME env var and looks under `configs/`.
    """
    if path is None:
        env_name = os.environ.get("CONFIG_NAME")
        if env_name:
            candidate = os.path.join("configs", env_name)
            if os.path.exists(candidate):
                path = candidate
            else:
                # allow passing just a basename with .yaml
                candidate_yaml = candidate if candidate.endswith(".yaml") else candidate + ".yaml"
                if os.path.exists(candidate_yaml):
                    path = candidate_yaml
        if path is None:
            path = os.path.join("configs", "example.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Training entrypoint (minimal CLI).")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    return parser.parse_args()


def get_config_from_cli_or_env():
    args = parse_args()
    if args.config:
        return load_config(args.config)
    return load_config(None)
