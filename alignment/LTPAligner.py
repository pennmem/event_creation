import os
import numpy as np
from loggers import logger
from ptsa.data.align import find_needle_in_haystack
from helpers.butter_filt import butter_filt

class LTPAligner:
    """
    Used for aligning the EEG data from the ephys computer with the task events from the behavioral computer. ALSO
    IDENTIFIES ARTIFACTS in the EEG data and adds info about them to the events structure.
    """
    ######### ALIGNMENT PARAMS #########
    ALIGNMENT_WINDOW = 100  # Tries to align this many sync pulses
    ALIGNMENT_THRESH = 10  # This many milliseconds may differ between sync pulse times during matching

    ##### ARTIFACT DETECTION PARAMS #####
    EOG_CHANS = [(25, 127), (8, 126)]
    BLINK_THRESH = 100

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
        self.events = events
        self.basename = os.path.splitext(self.pulse_files[0])[0] if len(self.pulse_files) > 0 else ''
        # Determine sample rate from the params.txt file
        if 'eeg_params' in files:
            with open(files['eeg_params']) as eeg_params_file:
                params_text = [line.split() for line in eeg_params_file.readlines()]
            self.sample_rate = int(params_text[0][1])
            self.gain = float(params_text[2][1])
        else:
            self.sample_rate = -999
            self.gain = 1

    '''
    ===============================================================
    ====== ALIGNMENT ==============================================
    ===============================================================
    '''

    def align(self):
        """
        Aligns the times at which sync pulses were sent by the behavioral computer with the times at which they were
        received by the ephys computer. This enables conversions to be made between the mstimes of behavioral events
        and the indices of EEG data samples that were taken at the same time. Behavioral sync pulse times come from
        the eeg.eeglog and eeg.eeglog.up files, located in the main session directory. Ephys sync pulses are identified
        from either a .D255, .DI15, or .DIN1 file located in the eeg.noreref directory. A linear regression is run on
        the behavioral and ephys sync pulse times in order to calculate the EEG sample number that corresponds to each
        behavioral event. The events structure is then updated with this information.

        After alignment, blinks and other artifacts occurring within the EEG signal are identified, and their info is
        also added into the events structure. Once all artifact info has been added, return the events structure.

        :return: The updated events structure, now filled with eegfile, eegoffset, and artifact information.
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

        """ ARTIFACT DETECTION """
        logger.debug('Searching for blinks in EOG channels...')
        # Identify blinks and other artifacts, and add information about them to the events that occurred during those
        # artifacts
        self.process_eog(self.EOG_CHANS)
        logger.debug('Searching for bad events...')
        # Do not look for additional artifacts in these channels. Includes EOG channels and several peripheral sites
        # that often have poor signals
        ignore_chans = np.array([25, 8, 127, 126, 14, 21, 107, 56, 49, 113, 99, 63, 1, 32, 44, 114, 17])
        self.find_bad_events(129, ignore_chans, duration=3200, offset=-200, buff=1000, filtfreq=[[58, 62]])

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

    '''
    ===============================================================
    ====== BLINK DETECTION ========================================
    ===============================================================
    '''

    def process_eog(self, eog_chans):
        """
        Locates blinks/artifacts in the EOG channels, then identifies which artifacts occurred during which events and
        adds this info to the existing events structure.

        :param eog_chans: A list of integers and/or tuples denoting which channels should be used for identifying blinks
        """
        logger.debug('Identifying blinks in the EOG channels...')
        # Use rereferenced data
        eeg_path = os.path.join(self.reref_dir, self.basename)
        # Load and scan the eeg data from each EOG channel. If the channel is a binary channel (represented as a tuple),
        # load both sub-channels and subtract one from the other before searching for blinks.
        artifact_mask = None
        for i in range(len(eog_chans)):
            ch = eog_chans[i]
            if isinstance(ch, tuple):
                logger.debug('Identifying blinks in binary channel ' + str(ch) + '...')
                eeg1 = np.fromfile(eeg_path + '.{:03}'.format(ch[0]), 'int16') * self.gain
                eeg2 = np.fromfile(eeg_path + '.{:03}'.format(ch[1]), 'int16') * self.gain
                eeg = eeg1 - eeg2
            else:
                logger.debug('Identifying blinks in channel ' + str(ch) + '...')
                eeg = np.fromfile(eeg_path + '.{:03}'.format(ch), 'int16') * self.gain

            # Instantiate artifact_mask once we know how many EEG samples there are. Note that this assumes all channels
            # have the same number of samples
            if artifact_mask is None:
                artifact_mask = np.empty((len(eog_chans), len(eeg)))

            # Find the blinks recorded by each EOG channel
            artifact_mask[i] = self.find_blinks(eeg, self.BLINK_THRESH)
            logger.debug('Done.')
        logger.debug('Blink identification complete.')

        # Get a list of the indices for all samples that contain a blink
        blinks = np.where(np.any(artifact_mask, 0))[0]

        logger.debug('Aligning blinks with events...')
        # Check for blinks that occurred during each event
        for i in range(len(self.events)):
            # Skip the event if it has no eegfile or no eegoffset, as it will not be possible to align artifacts with it
            if self.events[i].eegfile == '' or self.events[i].eegoffset < 0:
                continue

            # If on the last event or next event has no eegdata, look for artifacts occurring up to 1600 ms after the event onset
            if i == len(self.events) - 1 or self.events[i+1].eegoffset <= 0 :
                # FIXME: In original MATLAB, there is no handling for the final event's length. May want to introduce it here.
                ev_len = 2000 * self.sample_rate / 1000  # Placeholder - considers the 2 seconds following event onset
                ev_blink = blinks[np.where(np.logical_and(self.events[i].eegoffset <= blinks, blinks <= self.events[i].eegoffset + ev_len))[0]]
            # Otherwise, use the next event as the upper bound for aligning artifacts with the current event
            else:
                ev_blink = blinks[np.where(np.logical_and(self.events[i].eegoffset <= blinks, blinks < self.events[i+1].eegoffset))[0]]
                ev_len = self.events[i+1].eegoffset - self.events[i].eegoffset

            # Calculate and add artifact info to the current event, if any artifacts were present
            if ev_blink.size > 0:
                # Calculate how many milliseconds after the event it was that the blink onset occurred
                self.events[i].artifactMS = (ev_blink[0] - self.events[i].eegoffset) * 1000 / self.sample_rate
                # Calculate the number of samples during the event with artifacts in them
                self.events[i].artifactNum = len(ev_blink)
                # Calculate the porportion of samples during the event that have artifacts in them
                self.events[i].artifactFrac = float(len(ev_blink)) / ev_len
                # Calculate the average number of milliseconds after the event that artifacts occurred
                self.events[i].artifactMeanMS = (np.mean(ev_blink) - self.events[i].eegoffset) * 1000 / self.sample_rate
            else:
                continue
        logger.debug('Events successfully updated with artifact information.')

    @staticmethod
    def find_blinks(data, thresh):
        """
        Locates the blinks in an EEG signal. It does so by maintaining two running averages - one that changes quickly
        and one that changes slowly.

        The "slow" running average gives each new sample a weight of .025 and the current running average a weight of
        .975, then adds them to produce the new average. This average is used as a type of baseline measure against
        which the "fast" running average is calculated.

        The fast average tracks the divergence of the signal voltage from the slow/baseline average, in order to detect
        large spikes in voltage. In calculating each new average, it gives the new sample a weight of .5 and the
        current fast running average a weight of .5, allowing it to be heavily influenced by rapid voltage changes.

        Blinks are identified as occurring on any sample where the fast average exceeds the given threshold, i.e.
        whenever voltage displays a large and rapid divergence from baseline.

        :param data: A numpy array containing the EEG data that will be searched for blinks.
        :param thresh: The uV threshold used for determining when a blink has occurred.
        :return: A numpy array containing one boolean per EEG sample, indicating whether that sample contains a blink.
        """
        # Set weights for the fast-changing average (a, b) and the slow-changing average (c, d)
        a, b = (.5, .5)
        c, d = (.975, .025)

        # Create the arrays for the two averages
        num_samples = len(data)
        fast = np.empty(num_samples)
        slow = np.empty(num_samples)

        # Calculate starting "fast" and "slow" averages
        start_mean = np.mean(data[0:10])
        fast[0] = b * (data[0] - start_mean)
        slow[0] = c * start_mean + d * data[0]

        # Track the running averages across all samples
        for i in range(1, num_samples):
            fast[i] = a * fast[i-1] + b * (data[i] - slow[i-1])
            slow[i] = c * slow[i-1] + d * data[i]

        return abs(fast) >= thresh

    def find_bad_events(self, num_chans, ignore_chans, duration, offset, buff, filtfreq):
        """
        Determines which channels contain artifacts on each word presentation event. This is done by finding the mean
        and standard deviation of the voltage across all channels and events, and then labeling samples that are more
        than 4 standard deviations above or below the mean as bad samples. Any channel that contains at least one bad
        sample during a given event will be marked on the event as a bad channel. Any event with one or more bad
        channels is marked as a bad event.

        :param num_chans: The total number of EEG channels.
        :param ignore_chans: A numpy array of channels that should not be searched for artifacts.
        :param duration: How many milliseconds of EEG data will be loaded for each event.
        :param offset: If negative, is the number of samples to include before the onset of the word presentation. If
        positive, is the number of samples the skip immdiately after the word presentation.
        :param buff: The number of samples before {offset} that should also be included.
        :param filtfreq: The frequencies on which to filter when lodaing EEG data.

        data: A channels x events x samples matrix of EEG data.
        bad_evchans: A channels x events matrix
        bad_events: Each pres event has a row of booleans denoting whether each channel is bad (1 == bad, 0 == good)
        """
        # Get the indices of all word presentation events with eeg data
        pres_ev_ind = np.where(np.logical_and(self.events.type == 'WORD', self.events.eegfile != ''))[0]
        # Get a list of the channels to search for artifacts
        chans = np.array(range(1, num_chans + 1))
        chans = np.setdiff1d(chans, ignore_chans)
        # Calculate the number of samples to read from each event based on the duration in ms
        ev_length = int(duration * self.sample_rate / 1000)

        # Create a channels x events x samples matrix containing the samples from each channel during each event
        logger.debug('Loading reref data for word presentation events...')
        data = np.zeros((len(chans), len(pres_ev_ind), ev_length))
        for i in range(len(chans)):
            c = chans[i]
            data[i] = self.get_event_eeg(c, self.events[pres_ev_ind], ev_length, offset, buff, filtfreq)
        logger.debug('Done.')

        # Calculate mean and standard deviation across all samples from all presentation events
        avg = data.mean()
        stddev = data.std()

        # Set artifact threshold to 4 standard deviations away from the mean
        pos_thresh = avg + 4 * stddev
        neg_thresh = avg - 4 * stddev

        # Find artifacts by looking for samples that are greater than 4 standard deviations above or below the mean
        logger.debug('Finding artifacts during word presentations...')
        data = np.logical_or(data > pos_thresh, data < neg_thresh)

        # Get matrix of channels x events where entries are True if one or more bad samples occur on a channel during
        # an event
        bad_evchans = np.any(data, 2)
        # Get an array of booleans denoting whether each event is bad
        bad_events = np.any(bad_evchans, 0)
        # Mark each bad event with a list of the channels containing bad samples during the event
        logger.debug('Logging badEventChannel info...')
        for i in np.where(bad_events)[0]:
            self.events[i].badEvent = True
            self.events[i].badEventChannel = np.where(bad_evchans[:, i])

    def get_event_eeg(self, chan, events, ev_length, offset, buff, filtfreq):
        """
        Loads the eeg data from a single channel for each event. For each event in events, gets {ev_length} samples
        from channel {chan}, beginning with the sample defined by the event's eegoffset field. Runs a first-order,
        bandstop Butterworth filter on the data from each event upon loading.

        :param chan: The channel from which data will be loaded.
        :param events: The list of events whose data will be loaded.
        :param ev_length: The integer number of samples to load for each event.
        :param offset: If negative, is the number of samples to include before the onset of the word presentation. If
        positive, is the number of samples the skip immdiately after the word presentation.
        :param buff: The number of samples before {offset} that should also be included.
        :param filtfreq:
        :return: A matrix where each row is the array of data samples for one event.
        """
        data = np.zeros((len(events), ev_length))
        # Calculate the index of the first byte for each event's starting sample (16-bit data == 2 bytes per sample)
        byte_offsets = (offset - buff + events.eegoffset) * 2
        # Load each event's data from its reref file, while filtering the data from each event
        for i in range(len(events)):
            with open(os.path.join(self.reref_dir, events[i].eegfile) + '.{:03}'.format(chan), 'rb') as f:
                f.seek(byte_offsets[i])
                data[i] = np.fromfile(f, 'int16', ev_length)
                data[i] = butter_filt(data[i], filtfreq, self.sample_rate, filt_type='bandstop', order=1)
        # Multiply all data by the gain
        data *= self.gain
        return data


'''
===============================================================
====== ALIGNMENT HELPERS ======================================
===============================================================
'''


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
