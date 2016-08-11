import numpy as np
import scipy.stats
import os
import glob
import matplotlib.pyplot as plt
from readers.eeg_reader import NSx_reader
from copy import deepcopy
from parsers.system2_log_parser import System2LogParser
from alignment.system1 import UnAlignableEEGException
#from nsx_utility.brpylib import NsxFile
import itertools
import collections
import json
from loggers import log

class System2Aligner:
    '''
    Aligns pre-existing events with the host PC and neuroport based on host log files
    Optionally merges stimulation events from the host log into the pre-existing events as well
    '''

    TASK_TIME_FIELD = 'mstime'
    NP_TIME_FIELD = 'eegoffset'
    EEG_FILE_FIELD = 'eegfile'

    _DEFAULT_DATA_ROOT = '../tests/test_data'



    def __init__(self, events, files, diagnostic_plots=True):
        self.files = files
        self.host_log_files = files['host_logs']

        self.diagnostic_plots = diagnostic_plots
        self.events = events
        self.merged_events = events
        self.used_nsx = {}
        nsx_info = json.load(open(files['eeg_sources']))
        self.all_nsx_info = collections.OrderedDict(sorted(nsx_info, key=lambda item:item[0]))

        self.task_to_host_coefs, self.host_time_task_starts, _ = \
            self.get_coefficients_from_host_log(self.get_task_host_coefficient,
                                                self.diagnostic_plots,
                                                self.diagnostic_plots)

        self.host_to_np_coefs, self.host_time_np_starts, self.host_time_np_ends = \
            self.get_coefficients_from_host_log(self.get_host_np_coefficient,
                                                self.get_nsx_files()[0],
                                                self.diagnostic_plots,
                                                self.diagnostic_plots)

        self.np_log_lengths = np.array(self.host_time_np_ends) - np.array(self.host_time_np_starts)

    def get_all_nsx_files(self):
        ns_files = glob.glob(os.path.join(self.raw_directory, '*', '*.ns*'))
        nsx_files = [x for x in ns_files if os.path.splitext(x)[-1] in NSx_reader.SAMPLE_RATES]
        nsx_files.sort()
        return [NSx_reader.get_nsx_info(nsx_file) for nsx_file in nsx_files]


    def add_stim_events(self, event_template, persistent_fields=lambda *_: tuple()):
        s2lp = System2LogParser(self.host_log_files)

        self.merged_events = s2lp.merge_events(self.events, event_template, self.stim_event_to_mstime,
                                               persistent_fields)

        # Have to manually correct subject and session due to events appearing before start of session
        self.merged_events['subject'] = self.events[-1].subject
        self.merged_events['session'] = self.events[-1].session

    def align(self, start_type=None):
        aligned_events = deepcopy(self.merged_events)

        host_times = self.align_source_to_dest(aligned_events[self.TASK_TIME_FIELD],
                                               self.task_to_host_coefs,
                                               self.host_time_task_starts)

        starts_at = np.where(aligned_events['type'] == start_type)[0][0] if start_type else 0
        np_times = self.align_source_to_dest(host_times,
                                             self.host_to_np_coefs,
                                             self.host_time_np_starts, starts_at)

        aligned_events[self.NP_TIME_FIELD] = np_times
        self.apply_eeg_file(aligned_events, host_times)
        return aligned_events

    def apply_eeg_file(self, events, host_times):
        nsx_infos = self.get_used_nsx_files()
        split_filenames = [info.keys()[0] for info in nsx_infos]
        full_mask = np.zeros(events[self.EEG_FILE_FIELD].shape)
        for split_file, info, host_start in \
                zip(split_filenames, nsx_infos, self.host_time_np_starts):
            mask = np.logical_and(host_times > host_start, host_times < (host_start + info['length_ms']))
            full_mask = np.logical_or(full_mask, mask)
            events[self.EEG_FILE_FIELD][mask] = split_file
            self.used_nsx[info.datafile.name] = split_file

    def get_used_nsx_files(self):
        diff_np_starts = np.diff(self.host_time_np_starts)
        nsx_file_combinations = list(itertools.combinations(self.all_nsx_info, len(self.host_time_np_starts)))
        errors = []
        for nsx_files in nsx_file_combinations:
            nsx_lengths = np.array([nsx_file['length_ms'] for nsx_file in nsx_files])
            if not (nsx_lengths >= self.np_log_lengths).all():
                # Can't be that the recording was on for less time than was recorded in the log
                errors.append('NaN')
                continue
            nsx_start_times = [nsx_file['start_time'] for nsx_file in nsx_files]
            diff_nsx_starts = [(time2 - time1).microseconds / 1000 for time1, time2 \
                               in zip(nsx_start_times[:-1], nsx_start_times[1:])]
            errors.append(sum(abs(np.array(diff_nsx_starts - diff_np_starts))))

        best_index = np.argmin(errors)

        if errors[best_index] == 'NaN':
            raise UnAlignableEEGException('Could not find recording long enough to match events')

        if len(self.host_time_np_starts) > 1:
            min_errors = errors[best_index]
            fig, ax = plt.subplot()
            error_indices = np.arange(len(min_errors))
            ax.bar(error_indices, min_errors)
            ax.set_ylabel('Error in estimated time difference between start of recordings')
            ax.set_title('Accuracy of multiple-nsx file match-up')

        return nsx_file_combinations[best_index]

    @property
    def raw_directory(self):
        return self.files['raw_eeg']

    @property
    def host_logs_directory(self):
        return self.files['host_logs']

    def get_host_log_files(self):
        raise NotImplementedError
        log_files = glob.glob(os.path.join(self.host_logs_directory, '*.log'))
        assert len(log_files) > 0, 'No .log files found in %s. Check if EEG was transferred' % self.host_logs_directory
        log_files.sort()
        return log_files

    def get_nsx_files(self):
        nsx_files = glob.glob(os.path.join(self.raw_directory, '*', '*.ns2'))
        assert len(nsx_files) > 0, 'No nsx files found in %s. Check EEG transfer' % self.raw_directory
        return nsx_files

    def get_coefficients_from_host_log(self, coefficient_fn, *args):
        coefficients = []
        beginnings = []
        endings = []
        for host_log_file in self.host_log_files:
            (coefficient, host_start, host_end) = coefficient_fn(host_log_file, *args)
            if (coefficient or host_start):
                coefficients.extend(coefficient)
                beginnings.extend(host_start)
                endings.extend(host_end)
        return coefficients, beginnings, endings

    def stim_event_to_mstime(self, stim_event):
        earlier_resets = np.array(self.host_time_task_starts) < stim_event.hostTime
        if earlier_resets.any():
            good_coef_index = earlier_resets.nonzero()[0][-1]
        else:
            return -1
        return self.apply_coefficients_backwards(stim_event.hostTime, self.task_to_host_coefs[good_coef_index])

    @classmethod
    def align_source_to_dest(cls, time_source, coefficients, starts, okay_no_align_up_to=0):
        time_dest = np.full(len(time_source), np.nan)
        time_dest[time_source == -1] = -1
        for (task_start, task_end, coefficient) in zip(starts[:-1], starts[1:], coefficients):
            time_mask = (time_source >= task_start) & (time_source < task_end)
            time_dest[time_mask] = cls.apply_coefficients(time_source[time_mask], coefficient)
        time_mask = time_source >= starts[-1]
        time_dest[time_mask] = cls.apply_coefficients(time_source[time_mask], coefficients[-1])
        still_nans = np.where(np.isnan(time_dest))[0]
        if len(still_nans) > 0:
            if (np.array(still_nans) < okay_no_align_up_to).all():
                log('Warning: Could not align events %s' % still_nans)
                time_dest[np.isnan(time_dest)] = -1
            else:
                raise Exception('Could not convert some times past start of session')
        return time_dest

    @staticmethod
    def apply_coefficients_backwards(dest, coefficients):
        return (dest - coefficients[1]) / coefficients[0]

    @staticmethod
    def apply_coefficients(source, coefficients):
        return coefficients[0] * np.array(source) + coefficients[1]

    @classmethod
    def get_host_np_coefficient(cls, host_log_file, nsx_file, plot_fit=True, plot_residuals=True):
        [host_times, np_tics] = System2LogParser.get_columns_by_type(host_log_file, 'NEUROPORT-TIME', [0, 2], int)
        if (not host_times and not np_tics):
            log('No NEUROPORT-TIMEs in %s' % host_log_file)
            return [], [], []
        np_times = cls.samples_to_times(np_tics, nsx_file)
        [split_host, split_np] = cls.split_np_times(host_times, np_times)
        host_starts = []
        host_ends = []
        coefficients = []
        for (host_time, np_time) in zip(split_host, split_np):
            coefficients.append(cls.get_fit(host_time, np_time))
            cls.plot_fit(host_time, np_time, coefficients[-1], plot_fit, plot_residuals)
            host_starts.append(cls.apply_coefficients_backwards(0, coefficients[-1]))
            host_ends.append(host_times[-1])
        return coefficients, host_starts, host_ends

    @classmethod
    def split_np_times(cls, host_times, np_times):
        resets = np.concatenate([[0], np.where(np.diff(np_times) < 0)[0] + 1])
        split_host = []
        split_np = []
        # Could yield here...
        for start, end in zip(resets[:-1], resets[1:]):
            split_host.append(host_times[start:end-1])
            split_np.append(np_times[start:end-1])
        split_host.append(host_times[resets[-1]:])
        split_np.append(np_times[resets[-1]:])
        return split_host, split_np

    @classmethod
    def samples_to_times(cls, np_samples, nsx_file):
        ext = os.path.splitext(nsx_file)[1]
        return np.array(np_samples) / (float(NSx_reader.TIC_RATE) / float(NSx_reader.SAMPLE_RATES[ext]))

    @classmethod
    def get_task_host_coefficient(cls, host_log_file, plot_fit=True, plot_residuals=True):
        host_times, offsets = System2LogParser.get_columns_by_type(host_log_file, 'OFFSET', [0, 2], int)
        if len(host_times) <= 1:
            return [], [], []
        task_times = np.array(host_times) - np.array(offsets)
        coefficients = cls.get_fit(task_times, host_times)
        host_starts = host_times[0]
        host_ends = host_times[-1]

        cls.plot_fit(task_times, host_times, coefficients, plot_fit, plot_residuals)

        return [coefficients], [host_starts], [host_ends]

    @staticmethod
    def get_fit(x, y):
        coefficients = scipy.stats.linregress(x, y)
        return coefficients[:2]

    @staticmethod
    def plot_fit(x, y, coefficients, plot_fit=True, plot_residuals=True):
        fit = coefficients[0] * np.array(x) + coefficients[1]
        if plot_fit:
            plt.plot(x, y, 'g.', x, fit, 'b-')
            plt.show()

        if plot_residuals:
            plt.plot(x, y-fit, 'g-', [min(x), max(x)], [0, 0], 'k-')
            plt.show()
