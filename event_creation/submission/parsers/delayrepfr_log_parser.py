import numpy as np
import pandas as pd
from . import dtypes
from .repfr_log_parser import RepFRSessionParser

class DelayRepFRSessionLogParser(RepFRSessionParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(DelayRepFRSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files)

        with open(files["wordpool"]) as f:
            self.categories = np.unique([line.split("\t")[0].rstrip() for line in f])
        with open(files["wordpool"]) as f:
            self.wordpool = [line.split("\t")[1].rstrip() for line in f]

