import os
import mne
import numpy as np
from ..log import logger


class ArtifactDetector:
    """
    Runs scripts for blink and artifact detection. Parameters differ depending on whether the session was conducted
    using EGI or Biosemi. After detection processes are run, the events structure is filled with artifact data.
    """

    def __init__(self, events, eeg, ephys_dir):
        """
        :param events: The events structure (a recarray) for the session
        :param eeg: A dictionary matching the basename of each EEG recording to its data (designed for cases with 
        multiple recordings from a single session).
        """
        self.events = events
        self.basename = None  # Used for tracking the basename of the recording currently being processed
        self.eeg = eeg
        self.ephys_dir = ephys_dir

        self.eog_chans = None
        self.bad_chans = None
        self.blink_thresh = None

    def run(self):
        """
        Runs blink detection on the EOG channels, then artifact detection on the EEG data from all channels during all
        events. Currently supports EGI and Biosemi data.

        :return: The events structure updated with artifact and blink info.
        """
        if self.events.shape == () or len(self.eeg) == 0:
            logger.warn('Skipping artifact detection due to there being no events or invalid EEG parameter info.')
        else:
            for self.basename in self.eeg:
                # Prepare settings depending on the EEG system that was used
                if 'GSN-HydroCel-129' in self.eeg[self.basename].info['description']:
                    self.eog_chans = [('E25', 'E127'), ('E8', 'E126')]  # 127 is left, 126 is right
                    # These channels were excluded by the old MATLAB pipeline when calculating the mean and stddev
                    # during artifact detection
                    self.bad_chans = ['E1', 'E8', 'E14', 'E17', 'E21', 'E25', 'E32', 'E44', 'E49', 'E56', 'E63', 'E99', 'E107', 'E113', 'E114', 'E126', 'E127']
                    self.blink_thresh = .0001
                elif 'biosemi128' in self.eeg[self.basename].info['description']:
                    self.eog_chans = [('EXG3', 'EXG1'), ('EXG4', 'EXG2')]  # EXG1 and 3 are left, EXG2 and 4 are right
                    self.bad_chans = ['EXG1', 'EXG2', 'EXG3', 'EXG4']
                    self.blink_thresh = .0001
                else:
                    logger.warn('Unidentifiable EEG system detected in file %s' % self.basename)
                    continue

                # Find blinks in the current file
                self.process_eog()

                # Apply the common average reference before searching for artifacts
                self.eeg[self.basename].apply_proj()
                self.find_bad_events()

        return self.events

    def process_eog(self):
        """
        Locates blinks/artifacts in the EOG channels using find_blinks(), then identifies which artifacts occurred
        during which events and adds this info to the existing events structure. Fills the artifactMS, artifactNum,
        artifactFrac, and artifactMeanMS fields of the structure. Blinks are attached to an event if they occurred after
        the onset of that event and before the onset of the next event. If no onset can be determined for the next
        event, any blinks occurring up to 3 seconds after the event onset are attached to the event.
        """
        logger.debug('Identifying blinks in the EOG channels...')
        artifact_mask = None

        # Select the EEG data for the recording currently being processed
        eeg = self.eeg[self.basename]

        # Identify blinks in each EOG channel. For binary EOG channels (represented as tuples), get both EOG
        # sub-channels and subtract one from the other before searching for blinks
        for i in range(len(self.eog_chans)):
            ch = self.eog_chans[i]
            if isinstance(ch, tuple):
                logger.debug('Identifying blinks in binary channel ' + str(ch) + '...')
                # Get a 1D numpy array containing the data from an individual channel
                eog1 = eeg.get_data(picks=mne.pick_channels(eeg.ch_names, include=[ch[0]]))[0]
                eog2 = eeg.get_data(picks=mne.pick_channels(eeg.ch_names, include=[ch[1]]))[0]
                eog = eog1 - eog2
            else:
                logger.debug('Identifying blinks in channel ' + str(ch) + '...')
                eog = eeg.get_data(picks=mne.pick_channels(eeg.ch_names, include=[ch]))[0]

            # Instantiate artifact_mask once we know how many EEG samples there are. Note that this assumes all channels
            # have the same number of samples. The artifact_mask will be used to track which events have a blink on each
            # of the EOG channels. It has one row for each EOG channel and one column for each EEG sample.
            if artifact_mask is None:
                artifact_mask = np.empty((len(self.eog_chans), len(eog)))

            # Find the blinks recorded by each EOG channel and log them in artifact_mask
            artifact_mask[i] = self.find_blinks(eog, self.blink_thresh)

            logger.debug('Done.')
        logger.debug('Blink identification complete.')

        # Get a list of the indices for all samples that contain a blink on any EOG channel
        blinks = np.where(np.any(artifact_mask, 0))[0]

        logger.debug('Aligning blinks with events...')
        # Check each event to see if any blinks occurred during it
        for i in range(len(self.events)):
            # Skip the event if it has no eegfile or no eegoffset, as it will not be possible to align artifacts with it
            if self.events[i].eegfile != self.basename or self.events[i].eegoffset < 0:
                continue

            # If on the last event or next event has no eegdata, look for artifacts occurring up to 3000 ms after the
            # event onset
            if i == len(self.events) - 1 or self.events[i + 1].eegoffset <= 0:
                ev_len = 3000 * eeg.info['sfreq'] / 1000
                ev_blink = blinks[np.where(
                    np.logical_and(self.events[i].eegoffset <= blinks, blinks <= self.events[i].eegoffset + ev_len))[0]]
            # Otherwise, use the next event as the upper time bound for aligning artifacts with the current event
            else:
                ev_blink = blinks[
                    np.where(np.logical_and(self.events[i].eegoffset <= blinks,
                                            blinks < self.events[i + 1].eegoffset))[0]]
                ev_len = self.events[i + 1].eegoffset - self.events[i].eegoffset

            # Calculate and add artifact info to the current event, if any artifacts were present
            if ev_blink.size > 0:
                # Calculate how many milliseconds after the event it was that the blink onset occurred
                self.events[i].artifactMS = (ev_blink[0] - self.events[i].eegoffset) * 1000 / eeg.info['sfreq']
                # Calculate the number of samples during the event with artifacts in them
                self.events[i].artifactNum = len(ev_blink)
                # Calculate the porportion of samples during the event that have artifacts in them
                self.events[i].artifactFrac = float(len(ev_blink)) / ev_len
                # Calculate the average number of milliseconds after the event that artifacts occurred
                self.events[i].artifactMeanMS = (np.mean(ev_blink) - self.events[i].eegoffset) * 1000 / eeg.info['sfreq']
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

        :param data: A numpy array containing the EEG signal that will be searched for blinks.
        :param thresh: The mV threshold used for determining when a blink has occurred.
        :return: A numpy array containing one boolean per EEG sample, indicating whether that sample contains a blink.
        """
        # Create an array for each of the two running averages
        num_samples = len(data)
        fast = np.empty(num_samples)
        slow = np.empty(num_samples)

        # Calculate the starting "fast" and "slow" averages
        start_mean = np.mean(data[0:10])
        fast[0] = .5 * (data[0] - start_mean)
        slow[0] = .975 * start_mean + .025 * data[0]

        # Track the running averages across all samples
        for i in range(1, num_samples):
            fast[i] = .5 * fast[i - 1] + .5 * (data[i] - slow[i - 1])
            slow[i] = .975 * slow[i - 1] + .025 * data[i]

        # Mark whether each sample's "fast average" exceeded the threshold
        return abs(fast) >= thresh

    def find_bad_events(self):
        """
        DOCSTRING NEEDS UPDATE
        
        Determines which channels contain artifacts on each word presentation event. This is done by finding the mean
        and standard deviation of the voltage across all channels and events, and then labeling samples that are more
        than 4 standard deviations above or below the mean as bad samples. Any channel that contains at least one bad
        sample during a given event will be marked on the event as a badEventChannel. Any event with one or more bad
        channels is marked as a bad event.

        data: A channels x events x samples matrix of EEG data.
        bad_evchans: A channels x events matrix denoting which channels are bad on each event
        bad_events: An array denoting whether each event has at least one bad channel (1 == bad, 0 == good)
        """
        eeg = self.eeg[self.basename]
        picks_eeg = mne.pick_types(eeg.info, eeg=True, meg=False)

        # Get event numbers for all events aligned with the current EEG recording
        ev_mask = np.where(self.events.eegfile == self.basename)[0]
        # Get event numbers for all word presentation events aligned with the current EEG recording
        pres_mask = np.where(np.logical_and(self.events.type == 'WORD', self.events.eegfile == self.basename))[0]

        # Construct an mne-formatted event onset array to easily get the data from specified events. First column
        # holds the EEG sample number of the event onset, second column is ignored, third column holds event ID number.
        # Here we will mark word presentation events from the current recording with a 2, all other events from the
        # current recording with a 1, and events from other files with a 0
        ev_markers = np.zeros((len(self.events), 3), dtype=int)
        ev_markers[:, 0] = self.events.eegoffset[ev_mask]
        ev_markers[ev_mask, 2] = 1
        ev_markers[pres_mask, 2] = 2

        # Create dictionary of event types we want to load
        ev_ids = dict()
        if 1 in ev_markers[:, 2]:
            ev_ids['nonpres'] = 1
        if 2 in ev_markers[:, 2]:
            ev_ids['pres'] = 2

        # Get EEG data from 200 ms before through 3000 ms after each event from the current recording
        ev_data = mne.Epochs(eeg, ev_markers, event_id=ev_ids, tmin=-.2, tmax=3, baseline=None, preload=True)

        try:
            # Load the mean and std deviation if they have already been calculated. This way math events generation can
            # calculate the voltage thresholds without access to word presentation events.
            with open(os.path.join(self.ephys_dir, '%s_mean_stddev.txt' % self.basename), 'r') as f:
                thresh_data = f.readlines()
            avg = float(thresh_data[0])
            stddev = float(thresh_data[1])
            assert not np.isnan(avg)
            assert not np.isnan(stddev)
            logger.debug('Loaded average voltage from file.')
        except (IOError, AssertionError):
            logger.debug('Calculating average voltage...')

            # Get list of good EEG channels to use for calculating the artifact threshold
            good_chans = mne.pick_channels(eeg.ch_names, include=[], exclude=self.bad_chans)
            chans_to_use = np.intersect1d(picks_eeg, good_chans)

            # Get an events x channels x samples array containing the data from all good EEG channels during all word
            # presentation events
            pres_data = ev_data['pres'].get_data()[:, chans_to_use, :]

            # Calculate mean and standard deviation of voltage in good channels during presentation events. This will
            # be used for calculating the artifact threshold
            avg = pres_data.mean()
            stddev = pres_data.std()

            logger.debug('Saving average voltage...')
            with open(os.path.join(self.ephys_dir, '%s_mean_stddev.txt' % self.basename), 'w') as f:
                f.write(str(avg) + '\n' + str(stddev))

        # Set artifact threshold to 4 standard deviations away from the mean
        logger.debug('Calculating artifact thresholds...')
        pos_thresh = avg + 4 * stddev
        neg_thresh = avg - 4 * stddev

        # Find artifacts by looking for samples that are greater than 4 standard deviations above or below the mean
        logger.debug('Finding artifacts during all events...')

        # Convert ev_data from an mne.Epochs object to an events x EEG channels x samples numpy array of voltages
        ev_data = ev_data.get_data()[:, picks_eeg, :]

        # Replace ev_data with booleans indicating whether each sample exceeds the artifact threshold
        ev_data = np.logical_or(ev_data > pos_thresh, ev_data < neg_thresh)

        # Get array of events x EEG channels where bad_evchans(i, j) is True if one or more bad samples occur on channel
        # j during event i
        bad_evchans = ev_data.any(axis=2)

        # Get an array of booleans denoting whether each event has at least one bad EEG channel (not EOG channels)
        bad_events = bad_evchans.any(axis=1)

        # Mark each event with the channels that contain artifacts during that event.
        # Note that we use self.events[ev_mask[i]] instead of self.events[i] because ev_data only contains data for
        # events aligned with the current EEG recording. We need to index over events from this EEG file, not over all
        # events.
        logger.debug('Logging badEventChannel info...')
        for i in np.where(bad_events)[0]:
            self.events[ev_mask[i]].badEvent = True
            badEventChannel = np.array(eeg.ch_names)[picks_eeg[np.where(bad_evchans[i, :])]]  # Names of bad EEG channels during event i
            self.events[ev_mask[i]].badEventChannel = np.append(badEventChannel, np.array(['' for x in range(len(self.events[i].badEventChannel) - len(badEventChannel))]))
