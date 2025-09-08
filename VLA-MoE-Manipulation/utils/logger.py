# utils/logger.py
"""
Logger utilities for VLA-MoE Manipulation
-----------------------------------------

Features
- get_logger(name, log_file): create a logger that prints to console and file
- log levels: INFO, DEBUG, WARNING, ERROR
- simple AverageMeter class to track metrics
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str = "vla_moe", log_file: str = None, level: int = logging.INFO):
    """
    Create a logger with console + optional file handlers.

    Args:
        name     : logger name
        log_file : optional file path to save logs
        level    : log level (default INFO)

    Returns:
        logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # formatter with timestamp
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss/metrics across batches.
    """
    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} (avg:{avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    # quick self-test
    logger = get_logger("demo", "outputs/logs/demo.log")
    logger.info("This is an info message")
    logger.warning("This is a warning")

    meter = AverageMeter("loss")
    for i in range(1, 6):
        meter.update(i * 0.5)
        logger.info(str(meter))
