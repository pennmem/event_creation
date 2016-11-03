import os
import glob
import numpy as np
from loggers import logger
from helpers.butter_filt import butter_filt


class ArtifactDetector:

    def __init__(self, events, system, basename, reref_dir, sample_rate, gain):

        self.events = events
        self.basename = basename
        self.reref_dir = reref_dir
        self.sample_rate = sample_rate
        self.gain = gain
        self.known_sys = True

        if system == 'EGI':
            self.num_chans = 129
            self.eog_chans = [('025', '127'), ('008', '126')]
            self.weak_chans = np.array(['001', '008', '014', '017', '021', '025', '032', '044', '049', '056', '063',
                                        '099', '107', '113', '114', '126', '127'])
            self.blink_thresh = 100
        elif system == 'Biosemi':
            self.num_chans = 137
            self.eog_chans = [('EXG1', 'EXG2'), ('EXG3', 'EXG4'), ('EXG5', 'EXG6'), ('EXG7', 'EXG8')]
            self.weak_chans = np.array([])
            self.blink_thresh = 100
        else:
            logger.warn('Unknown EEG system \"%s\" detected while attempting to run artifact detection!' % self.system)
            self.known_sys = False

    def run(self):
        """
        Runs blink detection on the EOG channels and artifact detection on the EEG data from all events. Currently
        supports EGI and Biosemi data.

        :return: The events structure updated with artifact and blink info.
        """
        if self.known_sys:
            self.process_eog()
            self.find_bad_events(duration=3200, offset=-100, buff=500, filtfreq=[[58, 62]])
        return self.events

    def process_eog(self):
        """
        Locates blinks/artifacts in the EOG channels, then identifies which artifacts occurred during which events and
        adds this info to the existing events structure.
        """
        logger.debug('Identifying blinks in the EOG channels...')
        # Use rereferenced data
        eeg_path = os.path.join(self.reref_dir, self.basename)
        # Load and scan the eeg data from each EOG channel. If the channel is a binary channel (represented as a tuple),
        # load both sub-channels and subtract one from the other before searching for blinks.
        artifact_mask = None
        for i in range(len(self.eog_chans)):
            ch = self.eog_chans[i]
            if isinstance(ch, tuple):
                logger.debug('Identifying blinks in binary channel ' + str(ch) + '...')
                eeg1 = np.fromfile(eeg_path + '.' + ch[0], 'int16') * self.gain
                eeg2 = np.fromfile(eeg_path + '.' + ch[1], 'int16') * self.gain
                eeg = eeg1 - eeg2
            else:
                logger.debug('Identifying blinks in channel ' + str(ch) + '...')
                eeg = np.fromfile(eeg_path + '.' + ch, 'int16') * self.gain

            # Instantiate artifact_mask once we know how many EEG samples there are. Note that this assumes all channels
            # have the same number of samples
            if artifact_mask is None:
                artifact_mask = np.empty((len(self.eog_chans), len(eeg)))

            # Find the blinks recorded by each EOG channel
            artifact_mask[i] = self.find_blinks(eeg, self.blink_thresh)
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

            # If on the last event or next event has no eegdata, look for artifacts occurring up to 3000 ms after the
            # event onset
            if i == len(self.events) - 1 or self.events[i + 1].eegoffset <= 0:
                # FIXME: In original MATLAB, there is no handling for the final event's length. May want to introduce it here.
                ev_len = 3000 * self.sample_rate / 1000  # Placeholder - considers the 3 seconds following event onset
                ev_blink = blinks[np.where(
                    np.logical_and(self.events[i].eegoffset <= blinks, blinks <= self.events[i].eegoffset + ev_len))[0]]
            # Otherwise, use the next event as the upper bound for aligning artifacts with the current event
            else:
                ev_blink = blinks[
                    np.where(np.logical_and(self.events[i].eegoffset <= blinks,
                                            blinks < self.events[i + 1].eegoffset))[0]]
                ev_len = self.events[i + 1].eegoffset - self.events[i].eegoffset

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
            fast[i] = a * fast[i - 1] + b * (data[i] - slow[i - 1])
            slow[i] = c * slow[i - 1] + d * data[i]

        return abs(fast) >= thresh

    def find_bad_events(self, duration, offset, buff, filtfreq):
        """
        Determines which channels contain artifacts on each word presentation event. This is done by finding the mean
        and standard deviation of the voltage across all channels and events, and then labeling samples that are more
        than 4 standard deviations above or below the mean as bad samples. Any channel that contains at least one bad
        sample during a given event will be marked on the event as a bad channel. Any event with one or more bad
        channels is marked as a bad event.

        :param duration: How many milliseconds of EEG data will be loaded for each event.
        :param offset: If negative, is the number of samples to include before the onset of the word presentation. If
        positive, is the number of samples the skip immdiately after the word presentation.
        :param buff: The number of samples before {offset} that should also be included.
        :param filtfreq: The frequencies on which to filter when lodaing EEG data.

        data: A channels x events x samples matrix of EEG data.
        bad_evchans: A channels x events matrix denoting which channels are bad on each event
        bad_events: An array denoting whether each event has at least one bad channel (1 == bad, 0 == good)
        """
        # Get a list of the channels to search for artifacts. Look for all reref files that begin with the basename,
        # then get their extension (minus the preceding period)
        all_chans = np.array([os.path.splitext(f)[1][1:] for f in glob.glob(os.path.join(self.reref_dir, self.basename + '.*'))])
        # Calculate the number of samples to read from each event based on the duration in ms
        ev_length = int(duration * self.sample_rate / 1000)

        has_eeg = np.where(self.events.eegfile != '')[0]
        # Create a channels x events_with_eeg x samples matrix containing the samples from each channel during each event
        logger.debug('Loading reref data for word presentation events...')
        data = np.zeros((self.num_chans, len(has_eeg), ev_length))
        for i in range(len(all_chans)):
            data[i] = self.get_event_eeg(all_chans[i], self.events[has_eeg], ev_length, offset, buff, filtfreq)
        logger.debug('Done.')

        # Get the indices of all word presentation events with eeg data
        pres_ev_ind = np.where(np.logical_and(self.events.type == 'WORD', self.events.eegfile != ''))[0]
        chans_to_use = np.where(np.array([(chan not in self.weak_chans) for chan in all_chans]))[0]

        # Calculate mean and standard deviation across all samples in non-weak channels from all presentation events
        data_to_use = data[np.ix_(chans_to_use, pres_ev_ind, )]
        avg = data_to_use.mean()
        stddev = data_to_use.std()

        # Set artifact threshold to 4 standard deviations away from the mean
        pos_thresh = avg + 4 * stddev
        neg_thresh = avg - 4 * stddev

        print avg
        print stddev
        print pos_thresh
        print neg_thresh

        # Find artifacts by looking for samples that are greater than 4 standard deviations above or below the mean
        logger.debug('Finding artifacts during word presentations...')
        data = np.logical_or(data > pos_thresh, data < neg_thresh)

        # Get matrix of channels x events where entries are True if one or more bad samples occur on a channel during
        # an event
        bad_evchans = data.any(2)
        # Get an array of booleans denoting whether each event is bad
        bad_events = bad_evchans.any(0)
        # Mark each bad event with a list of the channels containing bad samples during the event
        logger.debug('Logging badEventChannel info...')
        for i in np.where(bad_events)[0]:
            self.events[has_eeg[i]].badEvent = True
            self.events[has_eeg[i]].badEventChannel = all_chans[np.where(bad_evchans[:, i])[0]]

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
        :param buff: Additional samples to be read on either side of the event which are used in filtering, but are
        removed before returning. 1 * buff additional samples are read before the event and 2 * buff additional samples
        are read after it.
        :param filtfreq: The frequencies on which to filter the data.
        :return: A matrix where each row is the array of data samples for one event.
        """
        len_with_buffer = ev_length + 3 * buff
        data = np.zeros((len(events), len_with_buffer))
        # Calculate the index of the first byte for each event's starting sample (16-bit data == 2 bytes per sample)
        byte_offsets = (offset - buff + events.eegoffset) * 2
        # Load each event's data from its reref file, while filtering the data from each event
        for i in range(len(events)):
            with open(os.path.join(self.reref_dir, events[i].eegfile) + '.' + str(chan), 'rb') as f:
                f.seek(byte_offsets[i])
                data[i] = np.fromfile(f, 'int16', len_with_buffer)
            data[i] = butter_filt(data[i], filtfreq, self.sample_rate, filt_type='bandstop', order=1)
        # Multiply all data by the gain
        data *= self.gain
        return data[:, buff:buff + ev_length]