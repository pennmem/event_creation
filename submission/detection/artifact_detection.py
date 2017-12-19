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
        :param ephys_dir: The path to the current_processed ephys folder for the current session.
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
        Adds artifact testing information to the events structure and records bad channels and bad-channel testing
        scores in a TSV file for each EEG recording in the session.

        :return: The events structure updated with artifact and blink information.
        """
        if self.events.shape == () or len(self.eeg) == 0 or np.sum(self.events.type == 'WORD') == 0:
            logger.warn('Skipping artifact detection due to there being no word presentation events or no EEG data!')
        else:
            for self.eegfile in self.eeg:
                # Drop miscellaneous/sync pulse channel(s)
                self.eeg[self.eegfile].pick_types(eeg=True, eog=True)

                # Prepare settings depending on the EEG system that was used
                if self.eegfile.endswith('.bdf'):
                    self.left_eog = ['EXG3', 'EXG1']
                    self.right_eog = ['EXG4', 'EXG2']
                elif self.eegfile.endswith('.mff') or self.eegfile.endswith('.raw'):
                    self.left_eog = ['E25', 'E127']
                    self.right_eog = ['E8', 'E126']
                else:
                    logger.warn('Unidentifiable EEG system detected in file %s' % self.eegfile)
                    continue

                # Set bipolar reference for EOG channels. Note that the resulting channels will be anode - cathode
                self.eeg[self.eegfile] = mne.set_bipolar_reference(self.eeg[self.eegfile], anode=[self.left_eog[0],
                                                    self.right_eog[0]], cathode=[self.left_eog[1], self.right_eog[1]])

                # Get a list of the channels names, and make sure we have 130 channels as intended (128 +
                self.chans = np.array(self.eeg[self.eegfile].ch_names)
                self.n_chans = len(self.chans)
                if (self.eegfile.endswith('.mff') or self.eegfile.endswith('.raw')) and self.n_chans != 126:
                    logger.warn('Artifact detection expected 124 EEG + 2 bipolar EOG channels for EGI but got %i for '
                                'file %s! Skipping...' % (self.n_chans, self.eegfile))
                    continue
                elif self.eegfile.endswith('.bdf') and self.n_chans != 130:
                    logger.warn(
                        'Artifact detection expected 128 EEG + 2 bipolar EOG channels for BioSemi but got %i for '
                        'file %s! Skipping...' % (self.n_chans, self.eegfile))
                    continue

                # Record the indices of the bipolar EOG channels, as positioned in the mne object
                self.leog_ind = self.eeg[self.eegfile].ch_names.index(self.left_eog[0] + '-' + self.left_eog[1])
                self.reog_ind = self.eeg[self.eegfile].ch_names.index(self.right_eog[0] + '-' + self.right_eog[1])

                # Create mask to select only EEG channels (not EOG)
                self.eeg_mask = np.ones(self.n_chans, dtype=bool)
                self.eeg_mask[self.reog_ind] = False
                self.eeg_mask[self.leog_ind] = False

                # Run artifact detection
                self.mark_bad_channels()
                self.mark_bad_epochs()

        return self.events


    def mark_bad_channels(self):
        """
        Runs several bad channel detection tests and records the test scores and automatically marked bad channels in
        a TSV file. The detection methods are as follows:

        1) Average voltage offset from the reference channel. This corresponds to the electrode offset screen in
        BioSemi's ActiView (though also appears to be effective for EGI sessions), and can be used to identify channels
        with poor connection to the scalp.

        2) Low average correlation with all other channels. As signals on the scalp are rather diffuse, a properly
        functioning channel should have some degree of correlation many other channels. Therefore, a channel with
        extremely low average correlation is likely broken or not actually connected to the scalp. Note that the
        absolute values of the correlation coefficients are used, such that the average correlation is always
        nonnegative and low z-scores indicate low correlations rather than strong negative correlations.

        3) Variance of the channel. Extremely high variance indicates a noisy channel, while extremely low variance
        indicates a flat channel.

        4) Log-transformed variance of the channel. Testing of method 3 revealed that variance is distributed in such
        a way that it is nearly impossible for channels to have highly negative z-scored variance. Flat channels are
        detected much more reliably after the variance values have been log-transformed.

        Channels are automatically marked as bad if the average reference offset is greater than 50 millivolts, if the
        z-scored average correlation is less than -3, or if the z-scored variance or log-transformed variance is
        greater than 3 or less than -3. Z-scores are calculated using the mean and standard deviation across all EEG
        channels (EOG channels are excluded). Note that methods 2 and 3 are adapted from the "FASTER" method by Nolan,
        Whelan, and Reilly (2010).

        Afterwards, a tab-separated values (.tsv) file called <eegfile_basename>_bad_chan.tsv is created where the
        scores for each EEG channel are recorded, including a column for whether each channel has been automatically
        determined to be a bad channel.

        :return: None
        """
        logger.debug('Identifying bad channels for %s' % self.eegfile)

        # Method 1: High voltage offset from the reference channel
        ref_offset = self.eeg[self.eegfile]._data[self.eeg_mask, :].mean(axis=1)

        # Method 2: Low correlation with other channels
        corr = np.corrcoef(self.eeg[self.eegfile]._data[self.eeg_mask, :])
        corr = np.abs(corr)
        corr = corr.mean(axis=0)

        # Method 3: High/Low variance
        var = np.var(self.eeg[self.eegfile]._data[self.eeg_mask, :], axis=1)

        # Method 4: High/Low log-transformed variance
        log_var = np.log10(var)

        # Make automated estimates of which channels are bad
        bad = np.where(np.logical_or.reduce((np.abs(ref_offset) > .05, ss.zscore(corr) < -3,
                                             np.abs(ss.zscore(var)) > 3, np.abs(ss.zscore(log_var)) > 3)))
        badch = np.zeros(self.n_chans, dtype=int)
        badch[bad] = True

        # Save a TSV file with the scores from each of the detection methods for each EEG channel
        bad_chan_file = os.path.join(self.ephys_dir, os.path.splitext(os.path.basename(self.eegfile))[0] + '_bad_chan.tsv')
        with open(bad_chan_file, 'w') as f:
            f.write('name\tref_offset\tcorr\tvar\tlog_var\tbad\n')
            for i, ch in enumerate(self.chans[self.eeg_mask]):
                    f.write('%s\t%f\t%f\t%f\t%f\t%i\n' % (ch, ref_offset[i], corr[i], var[i], log_var[i], badch[i]))


    def mark_bad_epochs(self):
        """
        Runs several bad epoch detection tests -- some on individual channels and others across channels. The test
        scores and automatically marked bad epochs/bad channels on each epoch are marked in the events structure. The
        detection methods for cross-channel bad epochs are as follows:

        1) Variance during the event, calculated for each channel then averaged across channels. This method is
        designed to identify events which have large amounts of noise across many or all channels.

        2) Amplitude range during the event, calculated for each channel then averaged across channels. This method is
        designed to identify events contaminated with high-amplitude artifacts or large baseline shifts across many or
        all channels.

        Entire presentation events are automatically marked as bad if the average variance across channels (z-scored
        across events) is greater than 3 or if the average amplitude range across channels (z-scored across events) is
        greater than 3. Note that both of these methods are adapted from the "FASTER" method by Nolan, Whelan, and
        Reilly (2010).

        The detection methods for marking individual channels as bad during each event are as follows:

        1) Variance of the channel during the event. Extremely high variance indicates that the channel was noisy during
        the event.

        2) Median slope of the channel during the event. A high median slope is characteristic of high-frequency
        artifacts that often originate from muscle activity.

        3) Amplitude range of the channel during the event. A high amplitude range may be indicative of a high-amplitude
        artifact or a large shift in baseline.

        4) Max/min deviation of the voltage from the channel's interquartile range. The interquartile range is
        calculated for each channel across all time points and presentation events. Deviation is measured as the maximum
        number of interquartile ranges above the 75th percentile or below the 25th percentile that the channel's voltage
        reaches during an event. This is an effective method for identifying high-amplitude artifacts, including blinks.

        Individual channels on each event are marked as bad if the variance of the channel (z-scored across events) is
        greater than 3, if the median slope of the channel during the event (z-scored across events) is greater than 3,
        if the amplitude range of the channel (z-scored across events) is greater than 3, and if the channel's voltage
        reaches 3 IQRs above its 75th percentile/below its 25th percentile at any point during the event. Methods 1, 2,
        and 3 were based on the methods used in "FASTER".

        Blink/eye movement detection is performed by applying method 4 to each EOG channel. Events where an EOG channel
        exceeds the 3*IQR threshold are marked as having an EOG artifact.

        The scores and automatic artifact markings for each word presentation are saved into the events structure
        directly.

        Bad epoch detection does not currently support event types other than word presentations, but could potentially
        be modified to do so.

        :return: None
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

        offsets = [o for i,o in enumerate(self.events.eegoffset) if self.events.type[i] in ev_ids and self.events.eegfile[i].endswith(self.eegfile)]
        ids = [ev_ids[self.events.type[i]] for i,o in enumerate(self.events.eegoffset) if self.events.type[i] in ev_ids and self.events.eegfile[i].endswith(self.eegfile)]
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
        avg_variance = variance[:, self.eeg_mask].mean(axis=1)

        # Method 2: High median slope for individual channels during event
        gradient = np.gradient(ep._data, axis=2)
        gradient = np.median(gradient, axis=2)

        # Method 3: High voltage range on individual channels during event
        amp_range = ep._data.max(axis=2) - ep._data.min(axis=2)
        avg_amp_range = amp_range[:, self.eeg_mask].mean(axis=1)

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

        # Mark entire events as bad if they have a high voltage range or variance across channels
        bad_epoch = np.logical_or(ss.zscore(avg_amp_range) > 3, ss.zscore(avg_variance) > 3)

        # Create events x channels matrices of booleans indicating whether each EEG channel is bad during each event
        eeg_art = np.logical_or.reduce((ss.zscore(variance, axis=0) > 3, ss.zscore(gradient, axis=0) > 3,
                                    ss.zscore(amp_range, axis=0) > 3, amp_max_iqr > 3, amp_min_iqr < -3))

        # Use only method 4 to search for blinks/eye movements in each EOG channel
        right_eog_art = np.logical_or(amp_max_iqr[:, self.reog_ind] > 3, amp_min_iqr[:, self.reog_ind] < -3)
        left_eog_art = np.logical_or(amp_max_iqr[:, self.leog_ind] > 3, amp_min_iqr[:, self.leog_ind] < -3)

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
                # badEpoch is True if abnormally high range or variance occurs across EEG channels
                self.events[i].badEpoch = bad_epoch[i]
                # artifactChannels is a 128-item array indicating whether each EEG channel is bad during each event
                self.events[i].artifactChannels[:self.n_chans-2] = eeg_art[i, self.eeg_mask]

                # variance is a 128-item array indicating the variance of each channel during the event
                self.events[i].variance[:self.n_chans-2] = variance[i, self.eeg_mask]
                # medGradiant is a 128-item array indicating the median gradient of each channel during the event
                self.events[i].medGradient[:self.n_chans-2] = gradient[i, self.eeg_mask]
                # ampRange is a 128-item array indicating the amplitude range of each channel during the event
                self.events[i].ampRange[:self.n_chans-2] = amp_range[i, self.eeg_mask]
                # iqrDevMax is a 128-item array how many IQRs above the 75th %ile each channel reaches during the event
                self.events[i].iqrDevMax[:self.n_chans-2] = amp_max_iqr[i, self.eeg_mask]
                # iqrDevMin is a 128-item array how many IQRs below the 25th %ile each channel reaches during the event
                self.events[i].iqrDevMin[:self.n_chans-2] = amp_min_iqr[i, self.eeg_mask]

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
