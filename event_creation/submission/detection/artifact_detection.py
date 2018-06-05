import mne
import numpy as np
from ..log import logger


class ArtifactDetector:
    """
    Runs scripts for Scalp Lab blink and artifact detection. Parameters differ depending on whether the session was
    conducted using EGI or Biosemi. After detection processes are run, the events structure is filled with artifact data.
    """

    def __init__(self, events, eeg, ephys_dir, experiment):
        """
        :param events: The events structure (a recarray) for the session
        :param eeg: A dictionary matching the basename of each EEG recording to its data (designed for cases with 
        multiple recordings from a single session).
        :param ephys_dir: The path to the current_processed ephys folder for the current session.
        :param experiment: The name of the experiment. This can be used to modify event types and timings for different
        experiments.
        """
        self.events = events
        self.eegfile = None  # Used for tracking the path to the recording which is currently being processed
        self.eeg = eeg
        self.ephys_dir = ephys_dir
        self.experiment = experiment

        self.system = None
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
                    self.system = 'bio'
                    self.left_eog = ['EXG3', 'EXG1']
                    self.right_eog = ['EXG4', 'EXG2']
                elif self.eegfile.endswith('.mff') or self.eegfile.endswith('.raw'):
                    self.system = 'egi'
                    self.left_eog = ['E25', 'E127']
                    self.right_eog = ['E8', 'E126']
                else:
                    logger.warn('Unidentifiable EEG system detected in file %s' % self.eegfile)
                    continue

                # Set bipolar reference for EOG channels. Note that the resulting channels will be anode - cathode
                self.eeg[self.eegfile] = mne.set_bipolar_reference(self.eeg[self.eegfile], anode=[self.left_eog[0],
                                                    self.right_eog[0]], cathode=[self.left_eog[1], self.right_eog[1]])

                # Get a list of the channels names, and make sure we have the proper number of channels
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
                self.mark_bad_epochs()

        return self.events

    def mark_bad_epochs(self):
        """
        Blink/eye movement detection is performed by applying the following method to each EOG channel. The
        interquartile range of voltages on a given channel is calculated across all time points during all presentation
        events. Positive and negative voltage thresholds are then set at 3 interquartile ranges above the 75th
        percentile and 3 interquartile ranges below the 25th percentile. An event is marked as having a blink if the
        voltage on any EOG channel exceeds either threshold.

        Blink detection does not currently support event types other than word presentations, but could potentially
        be modified to do so.

        :return: None
        """
        ##########
        #
        # Create EEG epochs from raw data
        #
        ##########

        logger.debug('Finding blinks for %s' % self.eegfile)

        SETTINGS = {
            'ltpFR': {'WORD': (0., 3.0)},
            'ltpFR2': {'WORD': (0., 1.6)},
            'VFFR': {'WORD': (0., 1.2), 'REC_WORD': (-1.0, -0.1)}
        }

        ev_types = SETTINGS[self.experiment]
        for t in ev_types:
            # Get the eeg offsets of all events of the target type
            offsets = [o for i, o in enumerate(self.events.eegoffset) if self.events.type[i] == t and
                       self.events.eegfile[i].endswith(self.eegfile)]
            # Skip to the next event type if there were no events of the target type
            if len(offsets) == 0:
                continue

            # Create MNE events using eeg offsets
            mne_evs = np.zeros((len(offsets), 3), dtype=int)
            mne_evs[:, 0] = offsets

            # Determine tmin and tmax using settings dictionary
            tmin = ev_types[t][0]
            tmax = ev_types[t][1]

            # Remove any events that run beyond the bounds of the EEG file
            truncated_events_pre = 0
            truncated_events_post = 0
            while mne_evs[0, 0] + self.eeg[self.eegfile].info['sfreq'] * tmin < 0:
                mne_evs = mne_evs[1:]
                truncated_events_pre += 1
            while mne_evs[-1, 0] + self.eeg[self.eegfile].info['sfreq'] * tmax >= self.eeg[self.eegfile].n_times:
                mne_evs = mne_evs[:-1]
                truncated_events_post += 1

            # Load data from all presentation events into an mne.Epochs object & baseline correct using each event's average voltage
            ep = mne.Epochs(self.eeg[self.eegfile], mne_evs, tmin=tmin, tmax=tmax, baseline=None, preload=True)

            ##########
            #
            # Individual-channel and all-channel bad epoch detection
            #
            ##########

            # Apply baseline correction on epoch data before analyzing individual channels across events
            ep.apply_baseline((0, None))

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

            # Search for blinks/eye movements in each EOG channel
            right_eog_art = np.logical_or(amp_max_iqr[:, self.reog_ind] > 3, amp_min_iqr[:, self.reog_ind] < -3)
            left_eog_art = np.logical_or(amp_max_iqr[:, self.leog_ind] > 3, amp_min_iqr[:, self.leog_ind] < -3)

            ##########
            #
            # Artifact Information Logging
            #
            ##########

            # Mark channels with artifacts during each presentation event
            logger.debug('Marking events with blink info...')

            # Skip event types which have not been tested with artifact detection, and those aligned to other recordings
            event_mask = np.where([ev.type == t and ev.eegfile.endswith(self.eegfile) for ev in self.events])[0]

            # Also skip any events that run beyond the bounds of the EEG file
            event_mask = event_mask[truncated_events_pre:]
            event_mask = event_mask[:-truncated_events_post] if truncated_events_post > 0 else event_mask

            # Set eogArtifact to 1 if an artifact was detected only on the left, 2 if only on the right, and 3 if both
            self.events.eogArtifact[event_mask] = 0
            self.events.eogArtifact[event_mask[left_eog_art]] += 1
            self.events.eogArtifact[event_mask[right_eog_art]] += 2

            logger.debug('Events marked with blink info for %s' % self.eegfile)
