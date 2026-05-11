# Run this once in a Python shell to see what CONFIG returns
from src.config import CONFIG
print(repr(CONFIG["simulator"]["min_replicas"]))   # should be int 1, not str "1"
print(repr(CONFIG["simulator"]["max_replicas"]))   # should be int 50, not str "50"