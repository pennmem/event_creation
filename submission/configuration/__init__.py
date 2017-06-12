import neurorad.config
from .config import Configuration

config = Configuration()
paths = config.paths
neurorad.config.paths = paths
