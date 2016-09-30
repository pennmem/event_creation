import os
import numpy as np
from logging import log
from warnings import warn

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
        self.behav_files = files['eeg_log']
        self.pulse_files = files['sync_pulses']
        self.eeg_dir = os.path.dirname(self.eeg_pulse_file)
        self.behav_ms = None
        self.pulse_ms = None
        self.pulses = None
        self.events = events

        # Determine sample rate and EEG data format from params file
        with open(files['eeg_params']) as eeg_params_file:
            params_text = [line.split() for line in eeg_params_file.readlines()]
        self.sample_rate = int(params_text[0][1])
        self.eeg_data_fmt = params_text[1][1]

    def align(self):
        """

        """
        log('Aligning...')

        # Determine which sync pulse file to use and get the indices of the samples that contain sync pulses
        self.get_pulse_sync()
        if not self.pulses:
            warn('No sync pulse file could be found. Unable to align behavioral and EEG data.', Warning)
            return False

        # Get the behavioral sync data and create eeg.eeglog.up if necessary
        if len(self.behav_files) > 0:
            self.get_behav_sync()
        else:
            warn('No eeg pulse log could be found. Unable to align behavioral and EEG data.', Warning)
            return False




    def get_pulse_sync(self):
        """
        Determines which type of sync pulse file to use for alignment when multiple are present. When this is the case,
        .D255 files take precedence, followed by .DI15, and then .DIN1 files. Once the correct sync file has been found,
        extract the indices of the samples that contain sync pulses, then calculate the mstimes of the pulses using
        the sample rate.
        """
        log('Acquiring EEG sync pulses...')
        for file_type in ('.D255', '.DI15', '.DIN1'):
            for f in self.pulse_files:
                if f.endswith(file_type):
                    pulse_sync_file = f
                    self.pulses = np.where(np.fromfile(pulse_sync_file, 'int16') > 0)
                    self.pulse_ms = self.pulses * 1000 / self.sample_rate
                    log('Done.')
                    return

    def get_behav_sync(self):
        """
        Checks if eeg.eeglog.up already exists. If it does not, create it by extracting the up pulses from the eeg log.
        Then set the eeg log file for alignment to be the eeg.eeglog.up file. Also gets the mstimes of the behavioral
        sync pulses.
        """
        log('Acquiring behavioral sync pulse times...')
        for f in self.behav_files:
            if f.endswith('.up'):
                self.behav_ms = np.loadtxt(f, dtype=int, usecols=[0])
        if not self.behav_ms:
            log('No eeg.eeglog.up file detected. Extracting up pulses from eeg.eeglog...')
            self.behav_ms = extract_up_pulses(self.behav_files[0])
        log('Done.')


def extract_up_pulses(eeg_log):
    """
    Extracts all up pulses from an eeg.eeglog file and writes them to an eeg.eeglog.up file.
    :param eeg_log: The filepath for the eeg.eeglog file.
    :return: Numpy array containing the mstimes for the up pulses
    """
    # Load data from the eeg.eeglog file and get all rows for up pulses
    data = np.loadtxt(eeg_log, dtype=str, skiprows=1, usecols=(0, 1, 2))
    up_pulses = data[data[:, 2] == 'UP']
    # Save the up pulses to eeg.eeglog.up
    np.savetxt(eeg_log + '.up', up_pulses, fmt='%s %s %s')
    # Return the mstimes from all up pulses
    return up_pulses[:, 0].astype(int)
