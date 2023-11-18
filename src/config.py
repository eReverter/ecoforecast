import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(log_file=None):
    logger = logging.getLogger('ForecastLogger')
    logger.setLevel(logging.INFO)

    # Remove all handlers that might have been added previously
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()  # It's important to close the handler.

    # Setup the log file path
    if log_file is None:
        current_dir = os.path.dirname(__file__)
        project_dir = os.path.abspath(os.path.join(current_dir, '..'))
        log_file = os.path.join(project_dir, 'main.log')

    # Add new handler
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=30)  # 10 MB limit, keep 30 old logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger