from logger import Logger

logger = Logger()

def log(*args, **kwargs):
    logger.log(*args, **kwargs)
