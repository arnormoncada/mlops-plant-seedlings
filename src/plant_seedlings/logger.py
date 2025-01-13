import os
import sys
from datetime import datetime
from loguru import logger

def setup_logger(
    log_dir: str,
    filename: str,
    rotation: str = None,
    retention: str = None,
    level: str = "INFO",
    backtrace: bool = False,
    colorize: bool = False,
):
    """
    Configures Loguru to save logs under `log_dir` with a timestamped filename.
    Also sets up an optional console sink that can be colorized and display full backtrace.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"{timestamp}_{filename}")

    # Optionally remove any default sinks if you want full control:
    # logger.remove()

    #File sink 
    logger.add(
        log_file_path,
        rotation=rotation,
        retention=retention,
        level=level,
        backtrace=backtrace,
        diagnose=True,  
        colorize=False,
    )

    # 2. Console sink
    logger.add(
        sys.stdout,
        level=level,
        backtrace=backtrace,
        diagnose=True,
        colorize=colorize,
    )

    logger.info(f"Logger set up. Logs will be saved at: {log_file_path}")
