import logging


def get_logger() -> logging.Logger:
    """
    Create a logger
    Returns:  A customized Logger
    """
    logger = logging.getLogger("vietac-logger")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger()
