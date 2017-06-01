import itertools
import json
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from .system1 import UnAlignableEEGException
from ..readers.eeg_reader import NSx_reader
from ..readers.eeg_reader import read_jacksheet
from ..loggers import logger
from ..parsers.system2_log_parser import System2LogParser


def System2Aligner(events, files, plot_save_dir=None):
    """
    Wrapper function which returns an instance of the Task Aligner, which aligns task to NP, or the Host Aligner,
    which aligns just Host to NP
    :param events: The events to be aligned
    :param files: Dictionary of file names to locations. must include 'host_logs', 'eeg_sources'
    :param plot_save_dir: Where to save plots describing fits
    :return: instance of aligner object
    """
    if 'session_log' in files:
        return System2TaskAligner(events, files, plot_save_dir)
    else:
        return System2HostAligner(events, files, plot_save_dir)


class System2TaskAligner(object):
    """
    Aligns pre-existing events with the host PC and neuroport based on host log files
    Optionally merges stimulation events from the host log into the pre-existing events as well
    """

    TASK_TIME_FIELD = 'mstime'   # Field which describes time on task PC
    NP_TIME_FIELD = 'eegoffset'  # Field which describes sample on EG system
    EEG_FILE_FIELD = 'eegfile'   # Field containing name of eeg file in events structure

    PLOT_SAVE_FILE_EXT = '.png'

    def __init__(self, events, files, plot_save_dir=None):
        """
        Constructor
        :param events: The events structure to be aligned
        :param files: Dictionary of file name -> locations. Must include 'host_logs', 'eeg_sources',
                      optionally 'jacksheet'
        :param plot_save_dir:
        """
        self.files = files

        # There can be multiple host logs, so always place in list
        if isinstance(files['host_logs'], list):
            self.host_log_files = sorted(files['host_logs'])
        else:
            self.host_log_files = [files['host_logs']]

        # Read the jacksheet into a dictionary
        if 'contacts' in files:
            self.jacksheet = read_jacksheet(files['contacts'])
        elif 'jacksheet' in files:
            self.jacksheet = read_jacksheet(files['jacksheet'])
        else:
            self.jacksheet = None

        self.plot_save_dir = plot_save_dir
        self.events = events
        self.merged_events = events

        # Store the name of the eeg files in the sources structure. Just makes things easier later
        nsx_info = json.load(open(files['eeg_sources']))
        for name in nsx_info:
            nsx_info[name]['name'] = name
        # Get the nsx files in order of their start time
        self.all_nsx_info = sorted([v for v in nsx_info.values()], key=lambda v: v['start_time_ms'])

        # Get the coefficients that map task to host
        self.task_to_host_coefs, self.host_time_task_starts, _ = \
            self.get_coefficients_from_host_log(self.get_task_host_coefficient,
                                                self.plot_save_dir)

        # Get the coefficients that map host to neuroport
        self.host_to_np_coefs, self.host_time_np_starts, self.host_time_np_ends = \
            self.get_coefficients_from_host_log(self.get_host_np_coefficient,
                                                self.all_nsx_info[0]['source_file'],
                                                self.plot_save_dir)

        # Get the length of each of the neuroport recordings as seen by the log on the host PC
        self.np_log_lengths = np.array(self.host_time_np_ends) - np.array(self.host_time_np_starts)

    def add_stim_events(self, event_template, persistent_fields=lambda *_: tuple()):
        """
        Adds events to the record array that correspond to when STIM occurred in the host log file
        :param event_template: Gives default values for events that are to be created
        :param persistent_fields: Function that accepts event and returns the fields from the event
                                  which should persist into a stim event. E.g., 'list' should be maintained.
        :return: events with stim merged in
        """

        # Merge in the stim events
        s2lp = System2LogParser(self.host_log_files, self.jacksheet)
        self.merged_events = s2lp.merge_events(self.events, event_template, self.stim_event_to_mstime,
                                               persistent_fields)

        # Have to manually correct subject and session due to events appearing before start of session
        self.merged_events['subject'] = self.events[-1].subject
        self.merged_events['session'] = self.events[-1].session
        return self.merged_events

    def align(self, start_type=None):
        """
        Do the actual alignment between the task and neuroport
        :param start_type: Don't care about events that occur before this event. Typically SESS_START.
        :return: events with updated eegoffset and eegfile fields
        """

        aligned_events = deepcopy(self.merged_events)

        host_times = self.align_source_to_dest(aligned_events[self.TASK_TIME_FIELD],
                                               self.task_to_host_coefs,
                                               self.host_time_task_starts)
        if start_type:
            starting_entries= np.where(aligned_events['type'] == start_type)[0]
            # Don't have to align until one after the starting type (which is normally SESS_START)
            starts_at = starting_entries[0] + 1 if len(starting_entries) > 0 else 0
        else:
            starts_at = 0

        np_times = self.align_source_to_dest(host_times,
                                             self.host_to_np_coefs,
                                             self.host_time_np_starts, starts_at)

        # Times that occur before the start of the recording are marked as -1
        np_times[np_times < 0] = -1
        aligned_events[self.NP_TIME_FIELD] = np_times

        # Add the eegfile field to the events
        self.apply_eeg_file(aligned_events, host_times)
        return aligned_events

    def apply_eeg_file(self, events, host_times):
        """
        Adds eegfile field to events
        :param events: events to be added to
        :param host_times: Entry corresponding to each event which give the time on the host PC
        :return: None
        """
        # Get the nsx files which were used in the session
        nsx_infos = self.get_used_nsx_files()
        full_mask = np.zeros(events[self.EEG_FILE_FIELD].shape)
        # Find the events which occurred between the start and end of the recording
        for info, host_start in \
                zip(nsx_infos, self.host_time_np_starts):
            mask = np.logical_and(host_times > host_start,
                                  host_times < (host_start + info['n_samples'] * float(info['sample_rate']) / 1000))
            full_mask = np.logical_or(full_mask, mask)
            events[self.EEG_FILE_FIELD][mask] = info['name']

    def get_used_nsx_files(self):
        """
        Gets the nsx files, in order, that were useds in this session
        :return: A list of the nsx files that were used
        """

        # The difference in time between starts of recordings as seen by the host PC
        diff_np_starts = np.diff(self.host_time_np_starts)

        # Gotta try every combination of NSx files
        nsx_file_combinations = list(itertools.combinations(self.all_nsx_info, len(self.host_time_np_starts)))
        if len(nsx_file_combinations) == 0:
            raise UnAlignableEEGException('Could not assign eegfile with {} recordings and {} np resets'.format(
                len(self.all_nsx_info), len(self.host_time_np_starts)))

        # Will hold the difference between the start of the recordings as seen by NP and Host
        errors = []
        for nsx_files in nsx_file_combinations:
            nsx_lengths = np.array([nsx_file['n_samples'] * float(nsx_file['sample_rate']) / 1000
                                    for nsx_file in nsx_files])
            if not (nsx_lengths >= self.np_log_lengths).all():
                # Can't be that the recording was on for less time than was recorded in the log
                errors.append('NaN')
                continue
            nsx_start_times = [nsx_file['start_time_ms'] for nsx_file in nsx_files]
            diff_nsx_starts = [(time2 - time1) for time1, time2 \
                               in zip(nsx_start_times[:-1], nsx_start_times[1:])]
            errors.append(sum(abs(np.array(diff_nsx_starts - diff_np_starts))))

        best_index = np.argmin(errors)

        if errors[best_index] == 'NaN':
            logger.debug("nsx file lengths: {}".format([nsx_file['n_samples'] * float(nsx_file['sample_rate'] / 1000) for nsx_file in self.all_nsx_info]))
            logger.debug("host time start differences: {}".format([x for x in diff_np_starts]))
            raise UnAlignableEEGException('Could not find recording long enough to match events')

        if len(self.host_time_np_starts) > 1:
            min_errors = errors[best_index]
            if min_errors > 10000:
                raise UnAlignableEEGException('Guess at beginning of recording inaccurate by over ten seconds (%d ms)' % min_errors)
            plt.clf()
            fig, ax = plt.subplots()
            error_indices = np.arange(len(min_errors)) if isinstance(min_errors, list) else 1
            ax.bar(error_indices, min_errors)
            ax.set_ylabel('Error in estimated time difference between start of recordings')
            ax.set_title('Accuracy of multiple-nsx file match-up')
            plt.savefig(os.path.join(self.plot_save_dir, 'multi-ns2{ext}'.format(ext=self.PLOT_SAVE_FILE_EXT)))

        return nsx_file_combinations[best_index]

    def get_coefficients_from_host_log(self, coefficient_fn, *args):
        """
        Given a function to calculate coefficients based on two sets of times, will get all coefficients from the host
        :param coefficient_fn: Function that accepts the host log file + *args and returns coefficients
        :param args: to be passed through to coefficient_fn
        :return: coefficients, times at which coefficients start to apply, times at which coefficients stop applying
        """
        coefficients = []
        beginnings = []
        endings = []
        for host_log_file in self.host_log_files:
            (coefficient, host_start, host_end) = coefficient_fn(host_log_file, *args)
            if (coefficient or host_start):
                # This is arbitrary, but prevents from caring about back-to-back recordings
                if len(beginnings) < 1 or len(host_start) == 1 and host_start[0] - beginnings[-1] > 100:
                    coefficients.extend(coefficient)
                    beginnings.extend(host_start)
                    endings.extend(host_end)
                # Replace the previous values with the current values if they start too close together
                elif len(beginnings) >= 1 and len(host_start) == 1 and host_start[0] - beginnings[-1] < 100:
                    coefficients[-1] = coefficient[0]
                    beginnings[-1] = host_start[0]
                    endings[-1] = host_end[0]

        return coefficients, beginnings, endings

    def stim_event_to_mstime(self, stim_event):
        """
        Given a stim event occurring at a specified host time, determine the time in task-time at which the event
        occurred
        :param stim_event: The stim event - must contains the field 'hosttime'
        :return: the mstime at which the event occurred
        """
        earlier_resets = np.array(self.host_time_task_starts) < stim_event['hosttime'][0]
        if earlier_resets.any():
            good_coef_index = earlier_resets.nonzero()[0][-1]
        else:
            return -1
        return self.apply_coefficients_backwards(stim_event['hosttime'][0],
                                                 self.task_to_host_coefs[good_coef_index])

    @classmethod
    def align_source_to_dest(cls, time_source, coefficients, starts, okay_no_align_up_to=0):
        """
        Given an array of source times, sets of coefficients, and the times at which those coefficients start to apply,
        returns the times after the coefficients have been applied
        :param time_source: The times to be converted
        :param coefficients: (slope, intercept) for each starting time
        :param starts: starting times, in source-time, for each coefficient (must be of length len(coefficients)-1)
        :param okay_no_align_up_to: Up to n=this event, it is okay if alignment doesn't occur
        :return: times aligned to destination
        """
        time_dest = np.full(len(time_source), np.nan)
        time_dest[time_source == -1] = -1
        for (task_start, task_end, coefficient) in zip(starts[:-1], starts[1:], coefficients):
            time_mask = (time_source >= task_start) & (time_source < task_end)
            time_dest[time_mask] = cls.apply_coefficients(time_source[time_mask], coefficient)
        time_mask = time_source >= starts[-1]
        time_dest[time_mask] = cls.apply_coefficients(time_source[time_mask], coefficients[-1])
        still_nans = np.where(np.isnan(time_dest))[0]
        if len(still_nans) > 0:
            if (np.array(still_nans) <= okay_no_align_up_to).all():
                logger.warn('Warning: Could not align events %s' % still_nans)
                time_dest[np.isnan(time_dest)] = -1
            else:
                logger.error("Events {} could not be aligned! Session starts at event {}".format(still_nans, okay_no_align_up_to))
                raise Exception('Could not convert {} times past start of session'.format(len(still_nans)))
        return time_dest

    @staticmethod
    def apply_coefficients_backwards(dest, coefficients):
        """
        Applies a set of coefficients in reverse, going from destination to source
        :param dest: "destination" time to be converted to source times
        :param coefficients: coefficients to be applied to destination times
        :return: "source" times
        """
        return (dest - coefficients[1]) / coefficients[0]

    @staticmethod
    def apply_coefficients(source, coefficients):
        """
        Applies a set of coefficients (slope, intercept) to an array or list of source values
        :param source: times to be converted
        :param coefficients: (slope, intercept)
        :return: converted times
        """
        return coefficients[0] * np.array(source) + coefficients[1]

    @classmethod
    def get_host_np_coefficient(cls, host_log_file, nsx_file, plot_save_dir=None):
        """
        For a given host log file and path to an nsx file (used to get sample rate), returns a list of the
        coefficients, start, and end times for each marked recording reset
        :param host_log_file: path to a single host log file
        :param nsx_file: path to an nsx file (used only to get sample rate from extension) TODO: should use nsx_info
        :param plot_save_dir: Directory in which to save plots
        :return: list of coefficients, list of recording starts (host time), list of recording ends (host time)
        """
        # Get NEUROPORT-TIMEs from host file
        [host_times, np_tics] = System2LogParser.get_columns_by_type(host_log_file, 'NEUROPORT-TIME', [0, 2], int)
        if (not host_times and not np_tics):
            logger.debug('No NEUROPORT-TIMEs in %s' % host_log_file)
            return [], [], []
        if len(host_times) == 1:
            logger.debug('Only one NEUROPORT-TIME in {}. Skipping.'.format(host_log_file), 'WARNING')

        # "samples" from host log are actually tics of internal np counter. Convert those to actual samples
        np_times = cls.tics_to_samples(np_tics, nsx_file)

        # Split the times into lists for each recording reset
        [split_host, split_np] = cls.split_np_times(host_times, np_times)
        host_starts = []
        host_ends = []
        coefficients = []

        # Get the coefficients, and the times at which they start and stop applying
        for (host_time, np_time) in zip(split_host, split_np):
            coefficients.append(cls.get_fit(host_time, np_time))
            cls.plot_fit(host_time, np_time, coefficients[-1], plot_save_dir, 'host_np')
            host_starts.append(cls.apply_coefficients_backwards(0, coefficients[-1]))
            host_ends.append(host_time[-1])
        return coefficients, host_starts, host_ends

    @classmethod
    def split_np_times(cls, host_times, np_times):
        """
        Splits a set of neuroport times, as read from a host log, based on where the times decrease
        :param host_times: List of all host times
        :param np_times: List of all neuroport times
        :return: (host times split by neuroport resets, neuroport times split by neuroport resets)
        """
        # Get the times at which the neuroport recording restarted
        resets = np.concatenate([[0], np.where(np.diff(np_times) < 0)[0] + 1])
        split_host = []
        split_np = []

        # Add each reset
        for start, end in zip(resets[:-1], resets[1:]):
            split_host.append(host_times[start:end-1])
            split_np.append(np_times[start:end-1])

        # Then one more for the last reset till the end of the file
        split_host.append(host_times[resets[-1]:])
        split_np.append(np_times[resets[-1]:])
        return split_host, split_np

    @classmethod
    def tics_to_samples(cls, np_tics, nsx_file):
        """
        Converts neuroport 30KHz 'tic's to samples based on the sample rate corresponding to the file extension
        :param np_tics: Tics of internal NP counter
        :param nsx_file: path to nsx file (extension used to determine sample rate)
        :return: Array of samples
        """
        if nsx_file == 'N/A':
            logger.warn("Guessing at sample rate of 1000")
            sample_rate = 1000
        else:
            ext = os.path.splitext(nsx_file)[1]
            sample_rate = float(NSx_reader.SAMPLE_RATES[ext])
        return np.array(np_tics) / (float(NSx_reader.TIC_RATE) / sample_rate)

    def get_task_host_coefficient(self, host_log_file, plot_save_dir=None):
        """
        Gets the coefficient that aligns the task pc times to the host pc times for a specific host log file
        :param host_log_file: The host log file
        :param plot_save_dir: Directory in which to save the plots of the fits
        :return: [coefficients], [starts], [ends] (wrapped in lists to be consistent with host_np_coefficient function
        """
        # Get the 'OFFSET's from the host log file
        host_times, offsets = System2LogParser.get_columns_by_type(host_log_file, 'OFFSET', [0, 2], int)
        # Return just empty lists if nothing was found
        if len(host_times) <= 1:
            return [], [], []

        # Task times are just host times minus offsets
        task_times = np.array(host_times) - np.array(offsets)
        coefficients = self.get_fit(task_times, host_times)

        # Get the times at which these started and stop applying
        host_starts = host_times[0]
        host_ends = host_times[-1]

        self.plot_fit(task_times, host_times, coefficients, plot_save_dir, 'task_host')

        # Wrap in list, in case function has to return multiple of each
        return [coefficients], [host_starts], [host_ends]

    @staticmethod
    def get_fit(x, y):
        """
        Gets the fit between an arbitrary x and y
        :param x:
        :param y:
        :return: slope, intercept
        """
        coefficients = scipy.stats.linregress(x, y)
        return coefficients[:2]

    @classmethod
    def plot_fit(cls, x, y, coefficients, plot_save_dir, plot_save_label):
        """
        Plots a fit between two values (both plotting fit itself and residuals
        :param x:
        :param y:
        :param coefficients:
        :param plot_save_dir: Where to save the plot
        :param plot_save_label: What to name the saved plot
        :return: None
        """
        fit = coefficients[0] * np.array(x) + coefficients[1]
        if plot_save_dir:
            plt.clf()
            plt.plot(x, y, 'g.', x, fit, 'b-')
            plt.savefig(os.path.join(plot_save_dir, '{label}_fit{ext}'.format(label=plot_save_label,
                                                                              ext=cls.PLOT_SAVE_FILE_EXT)))
            plt.show()

        if plot_save_dir:
            plt.clf()
            plt.plot(x, y-fit, 'g-', [min(x), max(x)], [0, 0], 'k-')
            plt.savefig(os.path.join(plot_save_dir, '{label}_residuals{ext}'.format(label=plot_save_label,
                                                                              ext=cls.PLOT_SAVE_FILE_EXT)))
            plt.show()



class System2HostAligner(System2TaskAligner):
    """
    Extends functionality of the System2TaskAligner, but used to align to just the Host, instead of the task PC
    """

    def __init__(self, events, files, plot_save_dir=None):
        """
        Constructor
        :param events: Events to be aligned
        :param files: File dict, output of Transferer
        :param plot_save_dir: Where to save plots
        """
        # Host offset allows us to convert host times to epoch time. We can calculate it up front to reduce
        # alignment time
        self.host_offset = self.get_host_offset(files)
        super(System2HostAligner, self).__init__(events, files, plot_save_dir)

    def get_host_offset(self, files):
        """
        Gets the conversion from host time to epoch time
        :param files: output of transferer
        :return: a single offset that will turn host time into epoch time
        """
        host_log_files = files['host_logs']

        # Get the times listed in the host log file
        # TODO: Could use System2LogParser here?
        if isinstance(host_log_files, list):
            self.host_log_files = sorted(host_log_files)
        else:
            self.host_log_files = [host_log_files]
        contents = []
        for host_log_file in self.host_log_files:
            contents += [line.strip().split('~')
                         for line in open(host_log_file).readlines()]

        eeg_sources = json.load(open(files['eeg_sources']))

        # Get the start time of the first eeg source
        first_eeg_source = sorted(eeg_sources.values(), key=lambda source: source['start_time_ms'])[0]
        np_earliest_start = first_eeg_source['start_time_ms']
        host_offset = None

        for split_line in contents:
            # Look for the first NEUROPORT-TIME in the file.
            if split_line[1] == 'NEUROPORT-TIME':
                # Start time of the recording is the host time minus that neuroport time divided by 30 (b/c 30 KHz)
                np_start_host = int(split_line[0]) - int(split_line[2]) / 30
                host_offset = np_start_host - np_earliest_start
                break

        if not host_offset:
            raise UnAlignableEEGException('Could not align host times')

        return host_offset

    def get_task_host_coefficient(self, host_log_file, plot_save_dir=None):
        """
        "Task" time here is actually epoch time. Return coefficients to remove the host offset in this time window
        :param host_log_file:
        :param plot_save_dir:
        :return: coefficients to align host to epoch
        """
        split_lines = [line.split('~') for line in open(host_log_file)]
        return [[1, self.host_offset]], [int(split_lines[0][0])], [int(split_lines[-1][0])]

    def stim_event_to_mstime(self, stim_event):
        """
        Used for sorting stim events. Converts stim event hosttime to epoch time
        :param stim_event: used to get host time
        :return: epoch time of stim event
        """
        return stim_event[0]['hosttime'] - self.host_offset
