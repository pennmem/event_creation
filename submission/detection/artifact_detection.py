import os
import mne
import numpy as np
import scipy.stats as ss
from ..log import logger


class ArtifactDetector:
    """
    Runs scripts for Scalp Lab blink and artifact detection. Parameters differ depending on whether the session was
    conducted using EGI or Biosemi. After detection processes are run, the events structure is filled with artifact data.
    """

    def __init__(self, events, eeg, ephys_dir):
        """
        :param events: The events structure (a recarray) for the session
        :param eeg: A dictionary matching the basename of each EEG recording to its data (designed for cases with 
        multiple recordings from a single session).
        """
        self.events = events
        self.eegfile = None  # Used for tracking the path to the recording which is currently being processed
        self.eeg = eeg
        self.ephys_dir = ephys_dir

        self.chans = None
        self.n_chans = None
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
            for self.eegfile in self.eeg:
                self.eeg[self.eegfile].pick_types(eeg=True, eog=True)  # Drop miscellaneous/sync pulse channel(s)
                # Prepare settings depending on the EEG system that was used
                if 'GSN-HydroCel-129' in self.eeg[self.eegfile].info['description']:
                    self.left_eog = ['E25', 'E127']
                    self.right_eog = ['E8', 'E126']
                    # These channels were excluded by the old MATLAB pipeline when calculating the mean and stddev during artifact detection
                    # self.bad_chans = ['E1', 'E8', 'E14', 'E17', 'E21', 'E25', 'E32', 'E44', 'E49', 'E56', 'E63', 'E99', 'E107', 'E113', 'E114', 'E126', 'E127']

                elif 'biosemi128' in self.eeg[self.eegfile].info['description']:
                    self.left_eog = ['EXG3', 'EXG1']
                    self.right_eog = ['EXG4', 'EXG2']
                    # self.bad_chans = ['EXG1', 'EXG2', 'EXG3', 'EXG4']

                else:
                    logger.warn('Unidentifiable EEG system detected in file %s' % self.eegfile)
                    continue

                self.chans = self.eeg[self.eegfile].ch_names
                self.n_chans = len(self.chans)
                if self.n_chans != 130:
                    logger.warn('Artifact detection expected 128 EEG + 4 EOG channels (excluding , but got %i for file %s! Skipping...' % (self.n_chans, self.eegfile))
                    continue

                # Set bipolar reference for EOG channels. Note that the resulting channels will be anode - cathode
                self.eeg[self.eegfile] = mne.set_bipolar_reference(self.eeg[self.eegfile], anode=[self.left_eog[0], self.right_eog[0]], cathode=[self.left_eog[1], self.right_eog[1]])
                # Record the indices of the bipolar EOG channels, as positioned in the mne object
                self.leog_ind = self.eeg[self.eegfile].ch_names.index(self.left_eog[0] + '-' + self.left_eog[1])
                self.reog_ind = self.eeg[self.eegfile].ch_names.index(self.right_eog[0] + '-' + self.right_eog[1])

                self.mark_bad_channels()
                self.mark_bad_epochs()

        return self.events


    def mark_bad_channels(self):
        """
        TBA
        :return:
        """
        logger.debug('Identifying bad channels for %s' % self.eegfile)

        # Method 1: High voltage offset from the reference channel
        ref_offset = self.eeg[self.eegfile].mean(axis=1)

        # Method 2: Low correlation with other channels
        corr = np.corrcoef(self.eeg[self.eegfile]._data)
        corr = np.abs(corr)
        corr = corr.mean(axis=0)

        # Method 3: High/Low variance
        var = np.var(self.eeg[self.eegfile]._data, axis=1)

        # Method 4: High/Low log-transformed variance
        log_var = np.log10(var)

        # Make automated estimates of which channels are bad
        bad = np.where(np.logical_or.reduce((np.abs(ref_offset) > .05, ss.zscore(corr) < -3,
                                             np.abs(ss.zscore(var)) > 3, np.abs(ss.zscore(log_var)) > 3)))
        badch = np.zeros(self.n_chans, dtype=int)
        badch[bad] = True

        # Save a TSV file with the scores from each of the detection methods
        bad_chan_file = os.path.join(self.ephys_dir, os.path.splitext(os.path.basename(self.eegfile))[0] + '_bad_chan.tsv')
        with open(bad_chan_file, 'w') as f:
            f.write('name\tref_offset\tcorr\tvar\tlog_var\tbad\n')
            for i, ch in enumerate(self.chans):
                f.write('%s\t%f\t%f\t%f\t%f\t%i\n' % (ch, ref_offset[i], corr[i], var[i], log_var[i], badch[i]))


    def mark_bad_epochs(self):
        """
        TBA
        :return:
        """
        ##########
        #
        # Create EEG epochs from raw data
        #
        ##########

        logger.debug('Identifying bad epochs for %s' % self.eegfile)

        # Create an mne events array with one row for each event of all types that appears in ev_ids (currently just
        # presentation events). The first column indicates the sample number of the event's onset, the second column is
        # ignored, and the third column indicates the event type as defined by the ev_ids dictionary.
        ev_ids = dict(
            WORD=0
        )

        offsets = [o for i,o in enumerate(self.events.eegoffset) if self.events.type[i] in ev_ids and self.events.eegfile[i] == self.eegfile]
        ids = [ev_ids[self.events.type[i]] for i,o in enumerate(self.events.eegoffset) if self.events.type[i] in ev_ids and self.events.eegfile[i] == self.eegfile]
        mne_evs = np.zeros((len(offsets), 3), dtype=int)
        mne_evs[:, 0] = offsets
        mne_evs[:, 2] = ids

        # Load data from all presentation events into an mne.Epochs object & baseline correct using each event's average voltage
        ep = mne.Epochs(self.eeg[self.eegfile], mne_evs, event_id=ev_ids, tmin=0., tmax=1.6, baseline=None, preload=True)

        ##########
        #
        # Individual-channel and all-channel bad epoch detection
        #
        ##########

        # Apply baseline correction on epoch data before analyzing individual channels across events
        ep.apply_baseline((0, None))

        # Method 1: High variance on individual channels during event
        variance = np.var(ep._data, axis=2)
        avg_variance = variance.mean(axis=1)

        # Method 2: High median slope for individual channels during event
        gradient = np.gradient(ep._data, axis=2)
        gradient = np.median(gradient, axis=2)

        # Method 3: High voltage range on individual channels during event
        amp_range = ep._data.max(axis=2) - ep._data.min(axis=2)
        avg_amp_range = amp_range.mean(axis=1)

        # Method 4: Large deviation of voltage from interquartile range on individual channels during event
        # Find the interquartile range of each channel, across time and across all events
        p75 = np.percentile(ep._data, 75, axis=[2, 0])
        p25 = np.percentile(ep._data, 25, axis=[2, 0])
        iqr = p75 - p25
        # Find the max and min of each channel during each event, then determine how many IQRs outside the IQR they fall
        amp_max_iqr = (ep._data.max(axis=2) - p75) / iqr
        amp_max_iqr[amp_max_iqr < 0] = 0
        amp_min_iqr = (ep._data.min(axis=2) - p25) / iqr
        amp_min_iqr[amp_min_iqr > 0] = 0

        # Create mask to select only EEG channels (not EOG)
        eeg_mask = np.ones(self.n_chans, dtype=bool)
        eeg_mask[self.reog_ind] = False
        eeg_mask[self.leog_ind] = False

        # Mark entire events as bad if they have a high voltage range or variance across channels
        bad_epoch = np.logical_or(ss.zscore(avg_amp_range) > 3, ss.zscore(avg_variance) > 3)

        # Create events x channels matrices of booleans indicating whether each EEG channel is bad during each event
        eeg_art = np.logical_or.reduce((ss.zscore(variance, axis=0) > 3, ss.zscore(gradient, axis=0) > 3,
                                    ss.zscore(amp_range, axis=0) > 3), amp_max_iqr > 3, amp_min_iqr < -3)[eeg_mask]

        # Use only method 4 to search for blinks/eye movements in each EOG channel
        right_eog_art = np.logical_or(amp_max_iqr[self.right_eog] > 3, amp_min_iqr[self.right_eog] < -3)
        left_eog_art = np.logical_or(amp_max_iqr[self.left_eog] > 3, amp_min_iqr[self.left_eog] < -3)

        ##########
        #
        # Artifact Information Logging
        #
        ##########

        # Mark channels with artifacts during each presentation event
        logger.debug('Marking events with artifact info...')
        for i in range(len(self.events)):
            if self.events[i].type not in ev_ids:
                continue
            else:
                # badEpoch is 1 if abnormally high range or variance occurs across EEG channels, else 0
                self.events[i].badEpoch = bad_epoch[i]
                # artifactChannels is a 128-item array indicating whether each EEG channel is bad during each event
                self.events[i].artifactChannels = eeg_art[i]

                # variance is a 128-item array indicating the variance of each channel during the event
                self.events[i].variance = variance[i]
                # medGradiant is a 128-item array indicating the median gradient of each channel during the event
                self.events[i].medGradient = gradient[i]
                # ampRange is a 128-item array indicating the amplitude range of each channel during the event
                self.events[i].ampRange = amp_range[i]
                # iqrDevMax is a 128-item array how many IQRs above the 75th %ile each channel reaches during the event
                self.events[i].iqrDevMax = amp_max_iqr[i]
                # iqrDevMin is a 128-item array how many IQRs below the 25th %ile each channel reaches during the event
                self.events[i].iqrDevMin = amp_min_iqr[i]

                # eogArtifact = 3 if an artifact was detected in both EOG channels during the event
                if right_eog_art[i] and left_eog_art[i]:
                    self.events[i].eogArtifact = 3
                # eogArtifact = 2 if an artifact was detected only in the right EOG channel during the event
                elif right_eog_art[i]:
                    self.events[i].eogArtifact = 2
                # eogArtifact = 1 if an artifact was detected only in the left EOG channel during the event
                elif left_eog_art[i]:
                    self.events[i].eogArtifact = 1
                # eogArtifact = 0 if no artifact was detected in either EOG channel during the event
                else:
                    self.events[i].eogArtifact = 0

        logger.debug('Events marked with artifact info for %s' % self.eegfile)
