from config import Configuration

config = Configuration()

paths = config.paths

import neurorad.config

neurorad.config.paths = paths
