import os
import numpy as np

class EGI_Aligner:
    """

    """
    TASK_TIME_FIELD = 'mstime'       # Field in events structure describing time on task laptop
    EEG_TIME_FIELD = 'eegoffset'     # field in events structure describing sample number on eeg system
    EEG_FILE_FIELD = 'eegfile'       # Field in events structure defining split eeg file in use at that time

    STARTING_ALIGNMENT_WINDOW = 100  # Tries to align these many sync pulses to start with
    ALIGNMENT_WINDOW_STEP = 10       # Reduces window by this amount if it cannot align
    MIN_ALIGNMENT_WINDOW = 5         # Does not drop below this many aligned pulses
    ALIGNMENT_THRESHOLD = 10         # This many ms may differ between sync pulse times

    def __init__(self, events, files):
        """
        Constructor
        :param events: The events structure to be aligned (np.recarray)
        :param files:  The output of a Transferer -- a dictionary mapping file name to file location.
        """
        self.task_pulse_file = files['eeg_log']
        self.eeg_pulse_file = ''
        self.eeg_dir = os.path.dirname(self.eeg_pulse_file)
        self.events = events

        # Determine sample rate and EEG data format from params file
        with open(files['eeg_params']) as eeg_params_file:
            params_text = [line.split() for line in eeg_params_file.readlines()]
        self.sample_rate = int(params_text[0][1])
        self.eeg_data_fmt = params_text[1][1]

    def align(self):
