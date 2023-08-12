import yaml
from easydict import EasyDict

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    return config