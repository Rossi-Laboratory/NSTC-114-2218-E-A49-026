# acp/utils/logger.py
import logging, sys, os
def get_logger(name='acp', log_file=None, level=logging.INFO):
    logger = logging.getLogger(name); logger.setLevel(level); logger.propagate=False
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
