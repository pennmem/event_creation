import numpy as np
import scipy.stats
import os

from nose.tools import raises
import json
from loggers import log


class UnAlignableEEGException(Exception):
    pass


class System1Aligner:

    TASK_TIME_FIELD = 'mstime'
    EEG_TIME_FIELD = 'eegoffset'
    EEG_FILE_FIELD = 'eegfile'

    _DEFAULT_DATA_ROOT = '../tests/test_data'

    TIC_RATE = 30000

    STARTING_ALIGNMENT_WINDOW = 100
    ALIGNMENT_WINDOW_STEP = 10
    MIN_ALIGNMENT_WINDOW = 5
    ALIGNMENT_THRESHOLD = 10

    def __init__(self, events, files):
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
        task_times, eeg_times = self.get_task_pulses(), self.get_eeg_pulses()
        slope, intercept = self.get_coefficient(task_times, eeg_times)
        self.events[self.EEG_TIME_FIELD] = (self.events[self.TASK_TIME_FIELD] * slope + intercept) * self.samplerate/1000.
        self.events[self.EEG_FILE_FIELD] = self.eeg_file_stem

        out_of_range = np.logical_or(self.events[self.EEG_TIME_FIELD] < 0,
                                     self.events[self.EEG_TIME_FIELD] > self.n_samples)

        n_out_of_range = np.count_nonzero(out_of_range)
        if n_out_of_range == self.events.size:
            raise UnAlignableEEGException('Could not align any events.')
        elif n_out_of_range:
            log('{} events out of range of eeg'.format(n_out_of_range), 'WARNING')

        self.events[self.EEG_FILE_FIELD][out_of_range] = ''
        self.events[self.EEG_TIME_FIELD][out_of_range] = 0
        return self.events

    def get_task_pulses(self):
        split_lines = [line.split() for line in open(self.task_pulse_file).readlines()]
        times = [float(line[0]) for line in split_lines if line[2]=='CHANNEL_0_UP']
        return times

    def get_eeg_pulses(self):
        times = np.array([float(line) for line in open(self.eeg_pulse_file).readlines()])
        dp = np.diff(times)
        times = times[:-1][dp > 100]
        return times * 1000. / self.samplerate

    @classmethod
    def get_coefficient(cls, task_pulse_ms, eeg_pulse_ms):
        task_pulse_ms = np.array(task_pulse_ms)
        eeg_pulse_ms = np.array(eeg_pulse_ms)
        matching_task_times, matching_eeg_times, max_residual= \
            cls.get_matching_pulses(task_pulse_ms, eeg_pulse_ms)
        slope, intercept, r, p, err = scipy.stats.linregress(matching_task_times, matching_eeg_times)
        prediction = matching_task_times * slope + intercept
        residual = matching_eeg_times - prediction
        #if max(abs(residual)) > max_residual + 1: #TODO: This is totally arbitrary
        #    raise UnAlignableEEGException('Start and end window do not match -- max residual: {}'.format(np.max(abs(residual))))
        log('Slope: {0}\nIntercept: {1}\nStd. Err: {2}\nMax residual: {3}\nMin residual: {4}'\
            .format(slope, intercept, err, max(residual), min(residual)))
        return slope, intercept


    @classmethod
    def get_matching_pulses(cls, task_pulse_ms, eeg_pulse_ms):
        task_diff = np.diff(task_pulse_ms)
        eeg_diff = np.diff(eeg_pulse_ms)

        log('Scanning for start window')
        task_start_range, eeg_start_range = cls.find_matching_window(eeg_diff, task_diff, True)

        log('Scanning for end window')
        task_end_range, eeg_end_range = cls.find_matching_window(eeg_diff, task_diff, False)

        [slope_start, intercept_start, _, _, _] = cls.get_fit(task_pulse_ms[task_start_range[0]: task_start_range[1]],
                                                              eeg_pulse_ms[eeg_start_range[0]: eeg_start_range[1]])
        [slope_end, intercept_end, _, _, _] = cls.get_fit(task_pulse_ms[task_end_range[0]: task_end_range[1]],
                                                          eeg_pulse_ms[eeg_end_range[0]: eeg_end_range[1]])

        prediction_start = slope_start * task_pulse_ms[task_start_range[0]: task_start_range[1]] + intercept_start
        residuals_start = eeg_pulse_ms[eeg_start_range[0]: eeg_start_range[1]] - prediction_start
        max_residual_start = max(abs(residuals_start))
        log('Max residual start %.1f' % max_residual_start)

        prediction_end = slope_end * task_pulse_ms[task_end_range[0]: task_end_range[1]] + intercept_end
        residuals_end = eeg_pulse_ms[eeg_end_range[0]: eeg_end_range[1]] - prediction_end
        max_residual_end = max(abs(residuals_end))
        log('Max residual end %.1f;' % max_residual_end)

        max_residual = max(max_residual_start, max_residual_end)

        task_range = np.union1d(range(task_start_range[0], task_start_range[1]),
                               range(task_end_range[0], task_end_range[1]))
        eeg_range = np.union1d(range(eeg_start_range[0], eeg_start_range[1]),
                                 range(eeg_end_range[0], eeg_end_range[1]))

        task_pulse_out = task_pulse_ms[task_range]
        eeg_pulse_out = eeg_pulse_ms[eeg_range]

        return task_pulse_out, eeg_pulse_out, max_residual

    @staticmethod
    def get_fit(x,y):
        return scipy.stats.linregress(x, y)

    @classmethod
    def find_matching_window(cls, eeg_diff, task_diff, from_front=True, alignment_window=None):
        if alignment_window is None:
            alignment_window = cls.STARTING_ALIGNMENT_WINDOW

        task_start_ind = None
        eeg_start_ind = None

        start_i = 0 if from_front else len(eeg_diff) - alignment_window
        end_i = len(eeg_diff) - alignment_window if from_front else 0
        step_i = 1 if from_front else -1

        for i in range(start_i, end_i, step_i):
            task_start_ind = cls.get_best_offset(eeg_diff[i:i + alignment_window],
                                            task_diff,
                                            cls.ALIGNMENT_THRESHOLD)
            if task_start_ind:
                eeg_start_ind = i
                break

        if not task_start_ind:
            if alignment_window - cls.ALIGNMENT_WINDOW_STEP < cls.MIN_ALIGNMENT_WINDOW:
                raise UnAlignableEEGException("Could not align window")
            else:
                log('Reducing align window to {}'.format(alignment_window - cls.ALIGNMENT_WINDOW_STEP), 'WARNING')
                return cls.find_matching_window(eeg_diff, task_diff, from_front,
                                                alignment_window - cls.ALIGNMENT_WINDOW_STEP)

        return (task_start_ind, task_start_ind + alignment_window), \
               (eeg_start_ind, eeg_start_ind + alignment_window)

    @staticmethod
    def get_best_offset(eeg_diff, task_diff, delta):
        ind = np.where(abs(task_diff - eeg_diff[0]) < delta)
        for i, this_eeg_diff in enumerate(eeg_diff):
            ind = np.intersect1d(ind, np.where(abs(task_diff-this_eeg_diff) < delta)[0] - i)
            if len(ind) == 0:
                return None
        if len(ind) > 1:
            raise UnAlignableEEGException("Multiple matching windows. Lower threshold or increase window.")
        return ind[0]

def xtest_align_sys1_pass():
    eeg_times = [int(x.strip())*2 for x in \
                 open('/Volumes/RHINO2/data/eeg/R1180C/eeg.noreref/R1180C_catFR1_0_04Jun16_1046.000.catFR1_0.sync.txt').readlines()]
    task_times = [int(x.strip().split()[0]) for x in \
                  open('/Volumes/RHINO2/data/eeg/R1180C/behavioral/catFR1/session_0/eeg.eeglog.up')]
    System1Aligner.get_coefficient(task_times, eeg_times)

@raises(UnAlignableEEGException)
def xtest_align_sys1_fail():
    eeg_times = [int(x.strip())*2 for x in \
                 open('/Volumes/RHINO2/data/eeg/R1180C/eeg.noreref/R1180C_catFR1_1_05Jun16_1042.000.catFR1_1.sync.txt').readlines()]
    task_times = [int(x.strip().split()[0]) for x in \
                  open('/Volumes/RHINO2/data/eeg/R1180C/behavioral/catFR1/session_0/eeg.eeglog.up')]
    System1Aligner.get_coefficient(task_times, eeg_times)

@raises(UnAlignableEEGException)
def test_align_sys2_fail2():
    eeg_times = [int(x.strip())*2 for x in \
                 open('/Volumes/RHINO2/data/eeg/R1180C/eeg.noreref/R1180C_catFR1_0_04Jun16_1046.000.catFR1_0.sync.txt').readlines()]
    task_times = [int(x.strip().split()[0]) for x in \
                  open('/Volumes/RHINO2/data/eeg/R1180C/behavioral/catFR1/session_0/eeg.eeglog.up')]
    eeg_times = np.array(eeg_times)
    print '*****BAD'
    eeg_times[500:] += 100

    System1Aligner.get_coefficient(task_times, eeg_times)

def xtest_align_sys1_events():
    subject = 'R1001P'
    session = 1
    exp = 'FR1'

    task_file = '/Volumes/RHINO2/data/eeg/R1001P/behavioral/FR1/session_1/eeg.eeglog'
    eeg_file = '/Volumes/RHINO2/data/eeg/R1001P/eeg.noreref/R1001P_13Oct14_1457.062.063.FR1_1.sync.txt'
    FR_parser = fr_log_parser_wrapper(subject, session, exp, base_dir='/Volumes/RHINO2/data/eeg')
    aligner = System1Aligner(FR_parser.parse(), task_file, eeg_file)
    py_events = aligner.align()
    from viewers.view_recarray import pprint_rec as ppr
    for py_event in py_events[-20:]:
        ppr(py_event)