import os
import numpy as np
from loggers import logger
from ptsa.data.align import find_needle_in_haystack


class LTPAligner:
    """
    Used for aligning the EEG data from the ephys computer with the task events from the behavioral computer.
    """
    ALIGNMENT_WINDOW = 100  # Tries to align this many sync pulses
    ALIGNMENT_THRESH = 10  # This many milliseconds may differ between sync pulse times during matching

    def __init__(self, events, files, behav_dir):
        """
        Constructor for the EGI aligner.

        :param events: The events structure to be aligned (np.recarray).
        :param files:  The output of a Transferer -- a dictionary mapping file name to file location.

        DATA FIELDS:
        behav_files: The list of sync pulse logs from the behavioral computer (eeg.eeglog, eeg.eeglog.up).
        pulse_files: The list of sync pulse logs from the ephys computer (.D255, .DI15, .DIN1 files).
        noreref_dir: The path to the EEG noreref directory.
        reref_dir: The path to the EEG reref directory.
        num_samples: The number of EEG samples in the sync channel file.
        pulses: A numpy array containing the indices of EEG samples that contain sync pulses.
        ephys_ms: The mstimes of the sync pulses received by the ephys computer.
        behav_ms: The mstimes of the sync pulses sent by the behavioral computer.
        ev_ms: The mstimes of all task events.
        events: The events structure for the experimental session.
        basename: The basename of a participant's split EEG files (without the .### extension).
        sample_rate: The sample rate of the EEG recording (typically 500).
        gain: The amplifier gain. Not used by alignment, but is loaded from params.txt for later use by the artifact
        detection system, since we're accessing the file anyway.
        system: EGI or Biosemi, depending on which EEG system was used for the session. Like gain, it is not used for
        alignment, but will be necessary for artifact detection afterwards.
        """
        self.behav_files = files['eeg_log'] if 'eeg_log' in files else []
        self.noreref_dir = os.path.join(os.path.dirname(os.path.dirname(behav_dir)), 'ephys', 'current_processed', 'noreref')
        self.reref_dir = os.path.join(os.path.dirname(self.noreref_dir), 'reref')
        self.pulse_files = []
        for f in os.listdir(self.noreref_dir):
            if f.endswith(('.DIN1', '.DI15', '.D255')):
                self.pulse_files.append(f)
        self.num_samples = -999
        self.pulses = None
        self.ephys_ms = None
        self.behav_ms = None
        self.ev_ms = events.view(np.recarray).mstime
        self.events = events.view(np.recarray)
        self.basename = os.path.splitext(self.pulse_files[0])[0] if len(self.pulse_files) > 0 else ''
        # Determine sample rate from the params.txt file
        if 'eeg_params' in files:
            with open(files['eeg_params']) as eeg_params_file:
                params_text = [line.split() for line in eeg_params_file.readlines()]
            self.sample_rate = int(params_text[0][1])
            self.gain = float(params_text[2][1])
            self.system = params_text[3][1]
        else:
            self.sample_rate = -999
            self.gain = 1
            self.system = ''

    def align(self):
        """
        Aligns the times at which sync pulses were sent by the behavioral computer with the times at which they were
        received by the ephys computer. This enables conversions to be made between the mstimes of behavioral events
        and the indices of EEG data samples that were taken at the same time. Behavioral sync pulse times come from
        the eeg.eeglog and eeg.eeglog.up files, located in the main session directory. Ephys sync pulses are identified
        from either a .D255, .DI15, or .DIN1 file located in the eeg.noreref directory. A linear regression is run on
        the behavioral and ephys sync pulse times in order to calculate the EEG sample number that corresponds to each
        behavioral event. The events structure is then updated with this information.

        :return: The updated events structure, now filled with eegfile and eegoffset information.
        """
        logger.debug('Aligning...')

        # Determine which sync pulse file to use and get the indices of the samples that contain sync pulses
        self.get_ephys_sync()
        if self.pulses is None:
            logger.warn('No sync pulse file could be found. Unable to align behavioral and EEG data.')
            return self.events

        # Get the behavioral sync data and create eeg.eeglog.up if necessary
        if len(self.behav_files) > 0:
            self.get_behav_sync()
        else:
            logger.warn('No eeg pulse log could be found. Unable to align behavioral and EEG data.')
            return self.events

        # Remove sync pulses that occurred less than 100 ms after the preceding pulse
        mask = np.where(np.diff(self.ephys_ms) < 100)[0] + 1
        self.ephys_ms = np.delete(self.ephys_ms, mask)
        self.pulses = np.delete(self.pulses, mask)

        # Calculate the eeg offset for each event using PTSA's alignment system
        logger.debug('Calculating EEG offsets...')
        try:
            eeg_offsets, s_ind, e_ind = times_to_offsets(self.behav_ms, self.ephys_ms, self.ev_ms, self.sample_rate,
                                                         window=self.ALIGNMENT_WINDOW, thresh_ms=self.ALIGNMENT_THRESH)
            logger.debug('Done.')

            # Add eeg offset and eeg file information to the events
            logger.debug('Adding EEG file and offset information to events structure...')
            oob = 0  # Counts the number of events that are out of bounds of the start and end sync pulses
            for i in range(self.events.shape[0]):
                if 0 <= eeg_offsets[i] <= self.num_samples:
                    self.events[i].eegoffset = eeg_offsets[i]
                    self.events[i].eegfile = self.basename
                else:
                    oob += 1
            logger.debug('Done.')

            if oob > 0:
                logger.warn(str(oob) + ' events are out of bounds of the EEG files.')

        except ValueError as e:
            logger.warn(e)
            logger.warn('Unable to align events with EEG data!')

        return self.events

    def get_ephys_sync(self):
        """
        Determines which type of sync pulse file to use for alignment when multiple are present. When this is the case,
        .D255 files take precedence, followed by .DI15, and then .DIN1 files. Once the correct sync file has been found,
        extract the indices of the samples that contain sync pulses, then calculate the mstimes of the pulses using
        the sample rate.
        """
        logger.debug('Acquiring EEG sync pulses...')
        if not hasattr(self.pulse_files, '__iter__'):
            self.pulse_files = [self.pulse_files]
        for file_type in ('.D255', '.DI15', '.DIN1'):
            for f in self.pulse_files:
                if f.endswith(file_type):
                    pulse_sync_file = f
                    eeg_samples = np.fromfile(os.path.join(self.noreref_dir, pulse_sync_file), 'int16')
                    self.num_samples = len(eeg_samples)
                    self.pulses = np.where(eeg_samples > 0)[0]
                    self.ephys_ms = self.pulses * 1000 / self.sample_rate
                    self.basename = f[:-5]
                    logger.debug('Done.')
                    return

    def get_behav_sync(self):
        """
        Checks if eeg.eeglog.up already exists. If it does not, create it by extracting the up pulses from the eeg log.
        Then set the eeg log file for alignment to be the eeg.eeglog.up file. Also gets the mstimes of the behavioral
        sync pulses.
        """
        logger.debug('Acquiring behavioral sync pulse times...')
        if not hasattr(self.pulse_files, '__iter__'):
            self.pulse_files = [self.pulse_files]
        for f in self.behav_files:
            if f.endswith('.up'):
                self.behav_ms = np.loadtxt(f, dtype=int, usecols=[0])
        if self.behav_ms is None:
            logger.debug('No eeg.eeglog.up file detected. Extracting up pulses from eeg.eeglog...')
            self.behav_ms = self.extract_up_pulses(self.behav_files[0])
        logger.debug('Done.')

    @staticmethod
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


def times_to_offsets(behav_ms, ephys_ms, ev_ms, samplerate, window=100, thresh_ms=10):
    """
    Slightly modified version of the times_to_offsets_old function in PTSA's alignment systems. Runs a regression on the
    behavioral and ephys sync pulse times in order to calculate which EEG sample number corresponds to each task event's
    mstime.

    :param behav_ms: The mstimes for all sync pulses sent by the behavioral computer.
    :param ephys_ms: The mstimes for all sync pulses received by the ephys computer.
    :param ev_ms: The mstimes for all task events.
    :param samplerate: The sample rate of the EEG recording (typically 500).
    :param window: The number of sync pulses to match between the behavioral and ephys computers' logs at the beginning
    and end of the recording.
    :param thresh_ms: The magnitude of discrepancy permitted when matching behavioral and ephys sync pulses.

    :return offsets: A numpy array containing the EEG sample number corresponding to the onset of each task event.
    :return s_ind: The index of the EEG sample that matches the beginning of the behavioral pulse syncs.
    :return e_ind: The index of the EEG sample that matches the end of the behavioral pulse syncs.
    """
    s_ind = None
    e_ind = None

    # Determine which range of samples in the ephys computer's pulse log matches the behavioral computer's sync pulse timings
    # Determine which ephys sync pulses correspond to the beginning behavioral sync pulses
    for i in xrange(len(ephys_ms) - window):
        s_ind = find_needle_in_haystack(np.diff(ephys_ms[i:i + window]), np.diff(behav_ms), thresh_ms)
        if s_ind is not None:
            start_ephys_vals = ephys_ms[i:i + window]
            start_behav_vals = behav_ms[s_ind:s_ind + window]
            break
    if s_ind is None:
        raise ValueError("Unable to find a start window.")

    # Determine which ephys sync pulses correspond with the ending behavioral sync pulses
    for i in xrange(len(ephys_ms) - window):
        e_ind = find_needle_in_haystack(np.diff(ephys_ms[::-1][i:i + window]), np.diff(behav_ms[::-1]), thresh_ms)
        if e_ind is not None:
            e_ind = len(behav_ms) - e_ind - window
            i = len(ephys_ms) - i - window
            end_ephys_vals = ephys_ms[i:i + window]
            end_behav_vals = behav_ms[e_ind:e_ind + window]
            break
    if e_ind is None:
        raise ValueError("Unable to find an end window.")

    # Perform a regression on the corresponding behavioral and ephys sync pulse times to enable a conversion between
    # event mstimes and EEG offsets.
    x = np.r_[start_behav_vals, end_behav_vals]
    y = np.r_[start_ephys_vals, end_ephys_vals]
    m, c = np.linalg.lstsq(np.vstack([x - x[0], np.ones(len(x))]).T, y)[0]
    c = c - x[0] * m

    eeg_start_ms = round((1-c)/m)

    # Use the regression to convert task event mstimes to EEG offsets
    offsets = np.int64(np.round((m*ev_ms + c)*samplerate/1000.))
    eeg_start = np.round(eeg_start_ms*samplerate/1000.)

    return offsets, s_ind, e_ind
