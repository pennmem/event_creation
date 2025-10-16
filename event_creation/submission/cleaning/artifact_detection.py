import mne
import numpy as np
from ..log import logger


class ArtifactDetector:
    """
    Runs scripts for Scalp Lab blink and artifact detection.
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
        self.eeg = eeg
        self.ephys_dir = ephys_dir
        self.experiment = experiment
        self.eegfile = None  # Tracks the path to the recording which is currently being processed
        self.system = None  # Will automatically be set to 'bio' for BioSemi sessions and 'egi' for EGI sessions

    def run(self):
        """
        Preprocesses EOG channels by applying a bipolar reference for each eye, then bandpass filtering the data to
        reduce irrelevant noise. Following this preprocessing, EOG artifact detection is performed by the
        mark_bad_epochs() method and artifact information is added to the events.

        :return: The events structure updated with artifact and blink information.
        """
        if self.events.shape == () or len(self.eeg) == 0 or np.sum(self.events.type == 'WORD') == 0:
            logger.warn('Skipping artifact detection due to there being no word presentation events or no EEG data!')
        else:
            for self.eegfile in self.eeg:

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

                # Pick only the EOG channels - need to be labeled as EEG
                # channels to be rereferenced.
                eog = self.eeg[self.eegfile].copy()
                eog.pick_channels(self.left_eog + self.right_eog).set_channel_types(
                        {c:'eeg' for c in self.left_eog+self.right_eog})

                # Set bipolar reference for EOG channels. Note that the resulting channels will be anode - cathode
                eog = mne.set_bipolar_reference(eog, anode=[self.left_eog[0], self.right_eog[0]],
                                                cathode=[self.left_eog[1], self.right_eog[1]])

                # Apply a 1-10 Hz bandpass filter on the EOG data to reduce irrelevant noise
                eog.filter(1, 10, filter_length='10s', phase='zero-double',
                           fir_window='hann', fir_design='firwin2')

                # Record the indices of the bipolar EOG channels, as positioned in the mne object
                self.leog_ind = eog.ch_names.index(self.left_eog[0] + '-' + self.left_eog[1])
                self.reog_ind = eog.ch_names.index(self.right_eog[0] + '-' + self.right_eog[1])

                # Run artifact detection
                self.detect_eog_artifacts(eog)

        return self.events

    def detect_eog_artifacts(self, eog):
        """
        Detects eye movement artifacts on the EOG channels and logs this information in the events. Events with
        artifacts detected on both eyes will be marked with a 3. Events with artifacts detected only on the left eye
        will be marked with a 1. Events with artifacts detected only on the right eye will be marked with a 2. Events
        with no EOG artifacts will be marked with a 0. Untested events will have a value of -1.

        Note that which event types blink detection is performed on -- as well as the onset/offset times for those
        events -- must be set individually for each experiment in SETTINGS.

        Blink/eye movement detection is performed for each event type by applying the following method to each EOG
        channel:
        1) Calculate the interquartile range of voltages on that channel across all time points during all events of
            that type.
        2) Set positive and negative voltage thresholds at 3 interquartile ranges above the 75th percentile and 3
            interquartile ranges below the 25th percentile.
        3) Mark every event where the voltage on that channel exceeds either threshold as having an EOG artifcat.

        :param eog: An MNE Raw object containing the EOG channel data to be searched for artifacts.
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
            'VFFR': {'WORD': (0., 1.2), 'REC_WORD': (-1.0, -0.1)},
            'prelim': {'WORD': (0., 1.6)},
            'RepFR': {'WORD': (0., 1.6)},
            'ltpRepFR': {'WORD': (0., 1.6)},
            'NiclsCourierReadOnly': {'WORD': (0, 1.6)},
            'NiclsCourierClosedLoop': {'WORD': (0, 1.6)},
            'ltpDelayRepFRReadOnly': {'WORD': (0., 1.6)},
            'CourierReinstate1': {'WORD': (0, 3.0)},
            'ValueCourier': {'WORD': (0, 3.0), 'VALUE_RECALL': (0, 3.0)}
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
            while mne_evs[0, 0] + eog.info['sfreq'] * tmin < 0:
                mne_evs = mne_evs[1:]
                truncated_events_pre += 1
            while mne_evs[-1, 0] + eog.info['sfreq'] * tmax >= eog.n_times:
                mne_evs = mne_evs[:-1]
                truncated_events_post += 1

            # Load data from all presentation events into an mne.Epochs object & baseline correct using each event's average voltage
            ep = mne.Epochs(eog, mne_evs, tmin=tmin, tmax=tmax, baseline=(None, None), preload=True)

            ##########
            #
            # EOG Artifact Detection
            #
            ##########

            # Look for large deviations of voltage from interquartile range on individual channels during event
            # Find the interquartile range of each channel, across time and across all events
            p75 = np.percentile(ep._data, 75, axis=[2, 0])
            p25 = np.percentile(ep._data, 25, axis=[2, 0])
            iqr = p75 - p25

            # Find the max and min of each channel in each event, then determine how many IQRs outside the IQR they fall
            amp_max_iqr = (ep._data.max(axis=2) - p75) / iqr
            amp_min_iqr = (ep._data.min(axis=2) - p25) / iqr

            # Search for blinks/eye movements in each EOG channel
            left_eog_art = np.logical_or(amp_max_iqr[:, self.leog_ind] > 3, amp_min_iqr[:, self.leog_ind] < -3)
            right_eog_art = np.logical_or(amp_max_iqr[:, self.reog_ind] > 3, amp_min_iqr[:, self.reog_ind] < -3)

            ##########
            #
            # Artifact Information Logging
            #
            ##########

            logger.debug('Marking events with blink info...')

            # Skip event types which have not been tested with artifact detection, and those aligned to other recordings
            event_mask = np.where([ev.type == t and ev.eegfile.endswith(self.eegfile) for ev in self.events])[0]

            # Also skip any events that run beyond the bounds of the EEG file (as determined previously)
            event_mask = event_mask[truncated_events_pre:]
            event_mask = event_mask[:-truncated_events_post] if truncated_events_post > 0 else event_mask

            # Set eogArtifact to 1 if an artifact was detected only on the left, 2 if only on the right, and 3 if both
            self.events.eogArtifact[event_mask] = 0
            self.events.eogArtifact[event_mask[left_eog_art]] += 1
            self.events.eogArtifact[event_mask[right_eog_art]] += 2

            logger.debug('Events marked with blink info for %s' % self.eegfile)
