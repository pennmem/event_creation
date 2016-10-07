import numpy as np
import scipy.stats
import os

from nose.tools import raises
import json
from loggers import logger


class UnAlignableEEGException(Exception):
    pass


class System1Aligner:
    """
    Alignment for System1 data.
    Uses a file that marks the time at which pulses were sent on the behavioral system, and a text file that marks
    the samples at which the sync pulses were received in the EEG to determine an EEG time for each behavioral time.

    NOTE: At the moment, can only handle a single eeg source -- cannot align multiple recordings for a single session
    """

    TASK_TIME_FIELD = 'mstime'      # Field in events structure describing time on task laptop
    EEG_TIME_FIELD = 'eegoffset'    # field in events structure describing sample number on eeg system
    EEG_FILE_FIELD = 'eegfile'      # Field in events structure defining split eeg file in use at that time

    STARTING_ALIGNMENT_WINDOW = 100 # Tries to align these many sync pulses to start with
    ALIGNMENT_WINDOW_STEP = 10      # Reduces window by this amount if it cannot align
    MIN_ALIGNMENT_WINDOW = 5        # Does not drop below this many aligned pulses
    ALIGNMENT_THRESHOLD = 10        # This many ms may differ between sync pulse times

    def __init__(self, events, files):
        """
        Constructor
        :param events: The events structure to be aligned (np.recarray)
        :param files:  The output of a Transferer -- a dictionary mapping file name to file location.
                       the files 'eeg_log', 'sync_pulses', and 'eeg_source' must be defined
        """
        self.task_pulse_file = files['eeg_log']
        self.eeg_pulse_file = files['sync_pulses']
        eeg_sources = json.load(open(files['eeg_sources']))
        if len(eeg_sources) != 1:
            raise UnAlignableEEGException('Cannot align EEG with %d sources' % len(eeg_sources))
        self.eeg_file_stem = eeg_sources.keys()[0]
        eeg_source = eeg_sources.values()[0]
        eeg_dir = os.path.dirname(self.eeg_pulse_file)
        self.samplerate = eeg_source['sample_rate']
        self.n_samples = eeg_source['n_samples']
        self.events = events

    def align(self):
        """
        Performs the actual alignment, using task and eeg pulse times to set eegoffset in events structure
        :return: events that have been aligned
        """
        task_times, eeg_times = self.get_task_pulses(), self.get_eeg_pulses()
        # Get the coefficients mapping the task times to the eeg times
        slope, intercept = self.get_coefficient(task_times, eeg_times)

        # Apply to the events structure
        self.events[self.EEG_TIME_FIELD] = (self.events[self.TASK_TIME_FIELD] * slope + intercept) * self.samplerate/1000.
        self.events[self.EEG_FILE_FIELD] = self.eeg_file_stem

        # Check to make sure that all of the events aren't beyond the end or beginning of the recording
        out_of_range = np.logical_or(self.events[self.EEG_TIME_FIELD] < 0,
                                     self.events[self.EEG_TIME_FIELD] > self.n_samples)
        n_out_of_range = np.count_nonzero(out_of_range)
        if n_out_of_range == self.events.size:
            raise UnAlignableEEGException('Could not align any events.')
        elif n_out_of_range:
            logger.warn('{} events out of range of eeg'.format(n_out_of_range))

        # For files that are out of range, mark them as having no eegfile
        self.events[self.EEG_FILE_FIELD][out_of_range] = ''
        self.events[self.EEG_TIME_FIELD][out_of_range] = -1
        return self.events

    def get_task_pulses(self):
        """
        Gets the lines from eeg.eeglog that mark that a sync pulse has been sent
        :return: list of the mstimes at which the sync pulses were sent.
        """
        split_lines = [line.split() for line in open(self.task_pulse_file).readlines()]
        times = [float(line[0]) for line in split_lines if line[2]=='CHANNEL_0_UP']
        return times

    def get_eeg_pulses(self):
        """
        Gets the samples at which sync pulses were received by the eeg system.
        Removes pulses that occur too close together (less than 100 samples)
        :return:
        """
        times = np.array([float(line) for line in open(self.eeg_pulse_file).readlines()])
        dp = np.diff(times)
        times = times[:-1][dp > 100]
        return times * 1000. / self.samplerate

    @classmethod
    def get_coefficient(cls, task_pulse_ms, eeg_pulse_ms):
        """
        Gets the coefficient that maps the task pulse times onto the eeg pulse times
        :param task_pulse_ms: the times at which sync pulses were sent in mstime
        :param eeg_pulse_ms:  the times at which sync pulses werew received in samples
        :return: slope, intercept
        """
        task_pulse_ms = np.array(task_pulse_ms)
        eeg_pulse_ms = np.array(eeg_pulse_ms)
        matching_task_times, matching_eeg_times, max_residual= \
            cls.get_matching_pulses(task_pulse_ms, eeg_pulse_ms)
        slope, intercept, r, p, err = scipy.stats.linregress(matching_task_times, matching_eeg_times)
        prediction = matching_task_times * slope + intercept
        residual = matching_eeg_times - prediction
        logger.debug('Slope: {0}\nIntercept: {1}\nStd. Err: {2}\nMax residual: {3}\nMin residual: {4}'\
            .format(slope, intercept, err, max(residual), min(residual)))
        return slope, intercept


    @classmethod
    def get_matching_pulses(cls, task_pulse_ms, eeg_pulse_ms):
        """
        Finds the pulses in the EEG recording which correspond to the times at which pulses were send on the
        task laptop
        :param task_pulse_ms: Times at which pulses were sent on task
        :param eeg_pulse_ms:  Samples at which pulses were received on eeg system
        :return: matching task pulses, matching eeg pulses, max residual from fit
        """

        # Going to find differences between pulse times that match between task and eeg
        task_diff = np.diff(task_pulse_ms)
        eeg_diff = np.diff(eeg_pulse_ms)

        # We match the beginning and the end separately, then draw a line between them
        logger.debug('Scanning for start window')
        task_start_range, eeg_start_range = cls.find_matching_window(eeg_diff, task_diff, True)

        logger.debug('Scanning for end window')
        task_end_range, eeg_end_range = cls.find_matching_window(eeg_diff, task_diff, False)

        # This whole next part was just for confirming that the fit is good,
        # However, it was never really implemented...
        [slope_start, intercept_start, _, _, _] = cls.get_fit(task_pulse_ms[task_start_range[0]: task_start_range[1]],
                                                              eeg_pulse_ms[eeg_start_range[0]: eeg_start_range[1]])
        [slope_end, intercept_end, _, _, _] = cls.get_fit(task_pulse_ms[task_end_range[0]: task_end_range[1]],
                                                          eeg_pulse_ms[eeg_end_range[0]: eeg_end_range[1]])

        prediction_start = slope_start * task_pulse_ms[task_start_range[0]: task_start_range[1]] + intercept_start
        residuals_start = eeg_pulse_ms[eeg_start_range[0]: eeg_start_range[1]] - prediction_start
        max_residual_start = max(abs(residuals_start))
        logger.debug('Max residual start %.1f' % max_residual_start)

        prediction_end = slope_end * task_pulse_ms[task_end_range[0]: task_end_range[1]] + intercept_end
        residuals_end = eeg_pulse_ms[eeg_end_range[0]: eeg_end_range[1]] - prediction_end
        max_residual_end = max(abs(residuals_end))
        logger.debug('Max residual end %.1f;' % max_residual_end)

        max_residual = max(max_residual_start, max_residual_end)

        # Join the beginning and the end
        task_range = np.union1d(range(task_start_range[0], task_start_range[1]),
                               range(task_end_range[0], task_end_range[1]))
        eeg_range = np.union1d(range(eeg_start_range[0], eeg_start_range[1]),
                                 range(eeg_end_range[0], eeg_end_range[1]))

        # Return the times that were used
        task_pulse_out = task_pulse_ms[task_range]
        eeg_pulse_out = eeg_pulse_ms[eeg_range]

        return task_pulse_out, eeg_pulse_out, max_residual

    @staticmethod
    def get_fit(x,y):
        """
        Simple wrapper method to get a fit between two data sets, in case method is changed
        :param x:
        :param y:
        :return: slope, intercept
        """
        return scipy.stats.linregress(x, y)

    @classmethod
    def find_matching_window(cls, eeg_diff, task_diff, from_front=True, alignment_window=None):
        """
        Finds the window in which the differences between the eeg pulses and the differences between the task pulses
        are the same as one another.
        :param eeg_diff: Differences in samples between eeg pulses
        :param task_diff: Differences in times between task pulses
        :param from_front: Whether to try to match from the front or the back
        :param alignment_window: How much of a window to attempt to align
        :return: (task start index, task end index), (eeg start index, eeg end index)
        """
        if alignment_window is None:
            alignment_window = cls.STARTING_ALIGNMENT_WINDOW

        task_start_ind = None
        eeg_start_ind = None

        # Define the indices of interest, and the direction of iteration
        start_i = 0 if from_front else len(eeg_diff) - alignment_window
        end_i = len(eeg_diff) - alignment_window if from_front else 0
        step_i = 1 if from_front else -1

        # Slide the window, looking for one that fits the differences
        for i in range(start_i, end_i, step_i):
            task_start_ind = cls.get_best_offset(eeg_diff[i:i + alignment_window],
                                            task_diff,
                                            cls.ALIGNMENT_THRESHOLD)
            # If it finds an offset, we can stop looking
            if task_start_ind:
                eeg_start_ind = i
                break

        # If it didn't find an offset, reduce the window and try again
        if not task_start_ind:
            # If the window gets too small, raise an error
            if alignment_window - cls.ALIGNMENT_WINDOW_STEP < cls.MIN_ALIGNMENT_WINDOW:
                raise UnAlignableEEGException("Could not align window")
            else:
                logger.warn('Reducing align window to {}'.format(alignment_window - cls.ALIGNMENT_WINDOW_STEP))
                return cls.find_matching_window(eeg_diff, task_diff, from_front,
                                                alignment_window - cls.ALIGNMENT_WINDOW_STEP)

        return (task_start_ind, task_start_ind + alignment_window), \
               (eeg_start_ind, eeg_start_ind + alignment_window)

    @staticmethod
    def get_best_offset(eeg_diff, task_diff, delta):
        """
        Finds the index of eeg_diff at which the pattern of differences in task_diff occurs
        :param eeg_diff: differences between samples of received sync pulses
        :param task_diff: differences between ms of sent sync pulses
        :param delta: threshold under which is considered a match for differences in times
        :return:  the offset of eeg_diff at which it begins matching with task_diff
        """

        # Find any differences that match
        ind = np.where(abs(task_diff - eeg_diff[0]) < delta)

        # For each difference, find the indices that still match
        for i, this_eeg_diff in enumerate(eeg_diff):
            ind = np.intersect1d(ind, np.where(abs(task_diff-this_eeg_diff) < delta)[0] - i)
            # If there are no indices left, return None
            if len(ind) == 0:
                return None

        # Raise an exception if we've found more than one
        if len(ind) > 1:
            raise UnAlignableEEGException("Multiple matching windows. Lower threshold or increase window.")
        return ind[0]