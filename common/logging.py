import logging
import sys

def setup_logger(name: str = "face_ar_system", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    # If the logger already has handlers, do not add more (to avoid duplicate logs)
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Define a clean, consistent format
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler using stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Disable propagation to root logger to avoid double printing in some environments
    logger.propagate = False
    
    return logger
