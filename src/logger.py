import logging
import sys
from pathlib import Path


LOG_DIR= Path(__file__).resolve().parent.parent/"results"/"logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name:str, level: int=logging.INFO)-> logging.Logger:
    logger=logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    formatter=logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler=logging.StreamHandler(sys.stdout)
    if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler=logging.FileHandler(LOG_DIR/"autoscaler.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger