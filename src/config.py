from pathlib import Path
import yaml

CONFIG_PATH=Path(__file__).resolve().parent.parent/"configs"/"config.yaml"
# print(CONFIG_PATH)

def load_config(path: Path=CONFIG_PATH)->dict:
    with open(path,'r') as f:
        return yaml.safe_load(f)
    
CONFIG= load_config()