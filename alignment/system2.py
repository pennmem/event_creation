import numpy as np
import scipy.stats
import os
import matplotlib.pyplot as plt
from readers.eeg_reader import NSx_reader
from copy import deepcopy
from parsers.system2_log_parser import System2LogParser
from alignment.system1 import UnAlignableEEGException
import itertools
import json
from loggers import log


def System2Aligner(events, files, plot_save_dir=None):
    if 'session_log' in files:
        return System2TaskAligner(events, files, plot_save_dir)
    else:
        return System2HostAligner(events, files, plot_save_dir)


class System2TaskAligner(object):
    """
    Aligns pre-existing events with the host PC and neuroport based on host log files
    Optionally merges stimulation events from the host log into the pre-existing events as well
    """

    TASK_TIME_FIELD = 'mstime'
    NP_TIME_FIELD = 'eegoffset'
    EEG_FILE_FIELD = 'eegfile'

    _DEFAULT_DATA_ROOT = '../tests/test_data'

    PLOT_SAVE_FILE_EXT = '.png'

    def __init__(self, events, files, plot_save_dir=None):
        self.files = files
        if isinstance(files['host_logs'], list):
            self.host_log_files = sorted(files['host_logs'])
        else:
            self.host_log_files = [files['host_logs']]

        if 'jacksheet' in files:
            jacksheet_contents = [x.strip().split() for x in open(files['jacksheet']).readlines()]
            self.jacksheet = {int(x[0]): x[1] for x in jacksheet_contents}
        else:
            self.jacksheet = None

        self.plot_save_dir = plot_save_dir
        self.events = events
        self.merged_events = events
        nsx_info = json.load(open(files['eeg_sources']))
        for name in nsx_info:
            nsx_info[name]['name'] = name
        self.all_nsx_info = sorted([v for v in nsx_info.values()], key=lambda v: v['start_time_ms'])

        self.task_to_host_coefs, self.host_time_task_starts, _ = \
            self.get_coefficients_from_host_log(self.get_task_host_coefficient,
                                                self.plot_save_dir)

        self.host_to_np_coefs, self.host_time_np_starts, self.host_time_np_ends = \
            self.get_coefficients_from_host_log(self.get_host_np_coefficient,
                                                self.all_nsx_info[0]['source_file'],
                                                self.plot_save_dir)


        self.np_log_lengths = np.array(self.host_time_np_ends) - np.array(self.host_time_np_starts)

    def add_stim_events(self, event_template, persistent_fields=lambda *_: tuple()):
        s2lp = System2LogParser(self.host_log_files, self.jacksheet)

        self.merged_events = s2lp.merge_events(self.events, event_template, self.stim_event_to_mstime,
                                               persistent_fields)

        # Have to manually correct subject and session due to events appearing before start of session
        self.merged_events['subject'] = self.events[-1].subject
        self.merged_events['session'] = self.events[-1].session
        return self.merged_events

    def align(self, start_type=None):
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

        np_times[np_times < 0] = -1
        aligned_events[self.NP_TIME_FIELD] = np_times
        self.apply_eeg_file(aligned_events, host_times)
        return aligned_events

    def apply_eeg_file(self, events, host_times):
        nsx_infos = self.get_used_nsx_files()
        full_mask = np.zeros(events[self.EEG_FILE_FIELD].shape)
        for info, host_start in \
                zip(nsx_infos, self.host_time_np_starts):
            mask = np.logical_and(host_times > host_start,
                                  host_times < (host_start + info['n_samples'] * float(info['sample_rate']) / 1000))
            full_mask = np.logical_or(full_mask, mask)
            events[self.EEG_FILE_FIELD][mask] = info['name']

    def get_used_nsx_files(self):
        diff_np_starts = np.diff(self.host_time_np_starts)
        nsx_file_combinations = list(itertools.combinations(self.all_nsx_info, len(self.host_time_np_starts)))
        if len(nsx_file_combinations) == 0:
            raise UnAlignableEEGException('Could not assign eegfile with {} recordings and {} np resets'.format(
                len(self.all_nsx_info), len(self.host_time_np_starts)))
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
            raise UnAlignableEEGException('Could not find recording long enough to match events')

        if len(self.host_time_np_starts) > 1:
            min_errors = errors[best_index]
            if min_errors > 1000:
                raise UnAlignableEEGException('Guess at beginning of recording inaccurate by over a second')
            plt.clf()
            fig, ax = plt.subplots()
            error_indices = np.arange(len(min_errors)) if isinstance(min_errors, list) else 1
            ax.bar(error_indices, min_errors)
            ax.set_ylabel('Error in estimated time difference between start of recordings')
            ax.set_title('Accuracy of multiple-nsx file match-up')
            plt.savefig(os.path.join(self.plot_save_dir, 'multi-ns2{ext}'.format(ext=self.PLOT_SAVE_FILE_EXT)))

        return nsx_file_combinations[best_index]

    @property
    def raw_directory(self):
        return self.files['raw_eeg']

    @property
    def host_logs_directory(self):
        return self.files['host_logs']

    def get_coefficients_from_host_log(self, coefficient_fn, *args):
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
                elif len(beginnings) >= 1 and len(host_start) == 1 and host_start[0] - beginnings[-1] < 100:
                    coefficients[-1] = coefficient[0]
                    beginnings[-1] = host_start[0]
                    endings[-1] = host_end[0]

        return coefficients, beginnings, endings

    def stim_event_to_mstime(self, stim_event):
        earlier_resets = np.array(self.host_time_task_starts) < stim_event['hosttime'][0]
        if earlier_resets.any():
            good_coef_index = earlier_resets.nonzero()[0][-1]
        else:
            return -1
        return self.apply_coefficients_backwards(stim_event['hosttime'][0],
                                                 self.task_to_host_coefs[good_coef_index])

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
            if (np.array(still_nans) <= okay_no_align_up_to).all():
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
    def get_host_np_coefficient(cls, host_log_file, nsx_file, plot_save_dir=None):
        [host_times, np_tics] = System2LogParser.get_columns_by_type(host_log_file, 'NEUROPORT-TIME', [0, 2], int)
        if (not host_times and not np_tics):
            log('No NEUROPORT-TIMEs in %s' % host_log_file)
            return [], [], []
        if len(host_times) == 1:
            log('Only one NEUROPORT-TIME in {}. Skipping.'.format(host_log_file), 'WARNING')

        np_times = cls.samples_to_times(np_tics, nsx_file)
        [split_host, split_np] = cls.split_np_times(host_times, np_times)
        host_starts = []
        host_ends = []
        coefficients = []
        for (host_time, np_time) in zip(split_host, split_np):
            coefficients.append(cls.get_fit(host_time, np_time))
            cls.plot_fit(host_time, np_time, coefficients[-1], plot_save_dir, 'host_np')
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

    def get_task_host_coefficient(self, host_log_file, plot_save_dir=None):
        host_times, offsets = System2LogParser.get_columns_by_type(host_log_file, 'OFFSET', [0, 2], int)
        if len(host_times) <= 1:
            return [], [], []
        task_times = np.array(host_times) - np.array(offsets)
        coefficients = self.get_fit(task_times, host_times)
        host_starts = host_times[0]
        host_ends = host_times[-1]

        self.plot_fit(task_times, host_times, coefficients, plot_save_dir, 'task_host')

        return [coefficients], [host_starts], [host_ends]

    @staticmethod
    def get_fit(x, y):
        coefficients = scipy.stats.linregress(x, y)
        return coefficients[:2]

    @classmethod
    def plot_fit(cls, x, y, coefficients, plot_save_dir, plot_save_label):
        fit = coefficients[0] * np.array(x) + coefficients[1]
        if plot_save_dir:
            plt.clf()
            plt.plot(x, y, 'g.', x, fit, 'b-')
            plt.savefig(os.path.join(plot_save_dir, '{label}_fit{ext}'.format(label=plot_save_label,
                                                                              ext=cls.PLOT_SAVE_FILE_EXT)))

        if plot_save_dir:
            plt.clf()
            plt.plot(x, y-fit, 'g-', [min(x), max(x)], [0, 0], 'k-')
            plt.savefig(os.path.join(plot_save_dir, '{label}_residuals{ext}'.format(label=plot_save_label,
                                                                              ext=cls.PLOT_SAVE_FILE_EXT)))

class System2HostAligner(System2TaskAligner):

    def __init__(self, events, files, plot_save_dir=None):
        self.host_offset = self.get_host_offset(files)
        super(System2HostAligner, self).__init__(events, files, plot_save_dir)

    def get_host_offset(self, files):
        host_log_files = files['host_logs']
        if isinstance(host_log_files, list):
            self.host_log_files = sorted(host_log_files)
        else:
            self.host_log_files = [host_log_files]
        contents = []
        for host_log_file in self.host_log_files:
            contents += [line.strip().split('~')
                         for line in open(host_log_file).readlines()]

        eeg_sources = json.load(open(files['eeg_sources']))

        first_eeg_source = sorted(eeg_sources.values(), key=lambda source: source['start_time_ms'])[0]
        np_earliest_start = first_eeg_source['start_time_ms']
        host_offset = None

        for split_line in contents:
            if split_line[1] == 'NEUROPORT-TIME':
                np_start_host = int(split_line[0]) - int(split_line[2])/30
                host_offset = np_start_host - np_earliest_start
                break

        if not host_offset:
            raise UnAlignableEEGException('Could not align host times')

        return host_offset

    def get_task_host_coefficient(self, host_log_file, plot_save_dir=None):
        split_lines = [line.split('~') for line in open(host_log_file)]
        return [[1, self.host_offset]], [int(split_lines[0][0])], [int(split_lines[-1][0])]

    def stim_event_to_mstime(self, stim_event):
        return stim_event[0]['hosttime'] - self.host_offset