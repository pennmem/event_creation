from readers.eeg_reader import get_eeg_reader
from submission.transferer import DATA_ROOT
import glob
import os

subject = raw_input('subject: ')

filenames = glob.glob(os.path.join(DATA_ROOT, subject, 'raw', '*', '*.edf')) + \
            glob.glob(os.path.join(DATA_ROOT, subject, 'raw', '*', '*.EEG'))

for filename in filenames:
    reader = get_eeg_reader(filename)
    print os.path.dirname(filename), reader.get_start_time_string()