from ...neurorad import config as neurorad_config
from .config import Configuration

config = Configuration()
paths = config.paths
neurorad_config.paths = paths
