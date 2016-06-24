import numpy as np
import scipy.stats
import os
import glob
import matplotlib.pyplot as plt
from copy import deepcopy
from parsers.system2_log_parser import System2LogParser


class System2Aligner:
    '''
    Aligns pre-existing events with the host PC and neuroport based on host log files
    Optionally merges stimulation events from the host log into the pre-existing events as well
    '''

    TASK_TIME_FIELD = 'mstime'
    NP_TIME_FIELD = 'eegoffset'

    _DEFAULT_DATA_ROOT = '../tests/test_data'

    TIC_RATE = 30000

    SAMPLE_RATES = {
        '.ns2': 1000
    }

    def __init__(self, subject, experiment, session, events, diagnostic_plots=True, data_root=None):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.data_root = data_root if data_root else self._DEFAULT_DATA_ROOT
        self.host_log_files = self.get_host_log_files()
        self.diagnostic_plots = diagnostic_plots
        self.events = events
        self.merged_events = events

        self.task_to_host_coefs, self.host_time_task_starts = \
            self.get_coefficients_from_host_log(self.get_task_host_coefficient,
                                                self.diagnostic_plots,
                                                self.diagnostic_plots)

        self.host_to_np_coefs, self.host_time_np_starts = \
            self.get_coefficients_from_host_log(self.get_host_np_coefficient,
                                                self.get_nsx_files()[0],
                                                self.diagnostic_plots,
                                                self.diagnostic_plots)


    def add_stim_events(self, event_template, persistent_fields=lambda *_: tuple()):
        s2lp = System2LogParser(self.host_log_files)

        self.merged_events = s2lp.merge_events(self.events, event_template, self.stim_event_to_mstime,
                                               persistent_fields)

        # Have to manually correct subject and session due to events appearing before start of session
        self.merged_events['subject'] = self.subject
        self.merged_events['session'] = self.session

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
        return aligned_events

    @property
    def raw_directory(self):
        return os.path.join(self.data_root, self.subject, 'raw', '%s_%d' % (self.experiment, self.session))

    def get_host_log_files(self):
        log_files = glob.glob(os.path.join(self.raw_directory, '*.log'))
        assert len(log_files) > 0, 'No .log files found in %s. Check if EEG was transferred' % self.raw_directory
        log_files.sort()
        return log_files

    def get_nsx_files(self):
        nsx_files = glob.glob(os.path.join(self.raw_directory, '*', '*.ns2'))
        assert len(nsx_files) > 0, 'No nsx files found in %s. Check EEG transfer' % self.raw_directory
        return nsx_files

    def get_coefficients_from_host_log(self, coefficient_fn, *args):
        coefficients = []
        beginnings = []
        for host_log_file in self.host_log_files:
            (coefficient, host_start) = coefficient_fn(host_log_file, *args)
            if (coefficient or host_start):
                coefficients.extend(coefficient)
                beginnings.extend(host_start)
        return coefficients, beginnings

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
                print 'Warning: Could not align events %s' % still_nans
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
            print 'No NEUROPORT-TIMEs in %s' % host_log_file
            return [], []
        np_times = cls.samples_to_times(np_tics, nsx_file)
        [split_host, split_np] = cls.split_np_times(host_times, np_times)
        host_starts = []
        coefficients = []
        for (host_time, np_time) in zip(split_host, split_np):
            coefficients.append(cls.get_fit(host_time, np_time))
            cls.plot_fit(host_time, np_time, coefficients[-1], plot_fit, plot_residuals)
            host_starts.append(cls.apply_coefficients_backwards(0, coefficients[-1]))
        return coefficients, host_starts

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
        return np.array(np_samples) / (float(cls.TIC_RATE) / float(cls.SAMPLE_RATES[ext]))

    @classmethod
    def get_task_host_coefficient(cls, host_log_file, plot_fit=True, plot_residuals=True):
        host_times, offsets = System2LogParser.get_columns_by_type(host_log_file, 'OFFSET', [0, 2], int)
        if len(host_times) <= 1:
            return [], []
        task_times = np.array(host_times) - np.array(offsets)
        coefficients = cls.get_fit(task_times, host_times)
        host_starts = host_times[0]

        cls.plot_fit(task_times, host_times, coefficients, plot_fit, plot_residuals)

        return [coefficients], [host_starts]

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
