import logging
from datetime import datetime

def setup_logger():
    """Set up logging to a timestamped file (no console)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"backtest_{timestamp}.log"

    logger = logging.getLogger("bt_mods")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Add file handler
    file_handler = logging.FileHandler(log_filename, mode="w")
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
