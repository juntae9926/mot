import os
from datetime import datetime
from loguru import logger

def setup_logger():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = os.path.join("runs", current_time)
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, "result.log")
    logger.add(log_path, format="{time} {level} {message}", level="INFO", rotation="10 MB", compression="zip")
    return log_folder