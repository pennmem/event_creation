
import json
import scipy.stats
import numpy as np

from copy import deepcopy
from loggers import logger

from parsers.system3_log_parser import System3LogParser

class System3Aligner(object):

    TASK_TIME_FIELD = 'mstime'
    ENS_TIME_FIELD = 'eegoffset'
    EEG_FILE_FIELD = 'eegfile'


    def __init__(self, events, files, plot_save_dir=None):

        self.files = files

        self.events_logs = files['event_log']

        self.electrode_config = files['electrode_config']

        self.plot_save_dir = plot_save_dir

        self.events = events
        self.merged_events = events

        self.task_to_ens_coefs, self.task_ends = \
            self.get_coefficients_from_event_log('t_event', 'offset')

        self.eeg_info = json.load(open(files['eeg_sources']))

    def add_stim_events(self, event_template, persistent_fields=lambda *_: tuple()):
        # Merge in the stim events

        s3lp = System3LogParser(self.events_logs, self.electrode_config)
        self.merged_events = s3lp.merge_events(self.events, event_template, persistent_fields)

        # Have to manually correct subject and session due to events appearing before start of session
        self.merged_events['subject'] = self.events[-1].subject
        self.merged_events['session'] = self.events[-1].session

        return self.merged_events


    def get_coefficients_from_event_log(self, from_label, to_label):

        ends = []
        coefs = []

        for event_log in self.events_logs:

            event_dict = json.load(open(event_log))['events']

            froms = [event[from_label] * 1000 for event in event_dict]
            tos = [event[to_label] for event in event_dict]

            coefs.append(scipy.stats.linregress(froms, tos)[:2])
            ends.append(froms[-1])

        return coefs, ends

    def align(self, start_type=None):

        aligned_events = deepcopy(self.merged_events)

        if start_type:
            starting_entries= np.where(aligned_events['type'] == start_type)[0]
            # Don't have to align until one after the starting type (which is normally SESS_START)
            starts_at = starting_entries[0] + 1 if len(starting_entries) > 0 else 0
        else:
            starts_at = 0

        ens_times = self.align_source_to_dest(aligned_events[self.TASK_TIME_FIELD],
                                              self.task_to_ens_coefs,
                                              self.task_ends, starts_at)

        ens_times[ens_times < 0] = -1
        aligned_events[self.ENS_TIME_FIELD] = ens_times

        self.apply_eeg_file(aligned_events)

        return aligned_events

    def apply_eeg_file(self, events):

        eeg_info = sorted(self.eeg_info.items(), key= lambda info:info[1]['start_time_ms'])

        for eegfile, info in eeg_info:
            mask = events[self.TASK_TIME_FIELD] >= info['start_time_ms']
            events[self.EEG_FILE_FIELD][mask] = eegfile

    @classmethod
    def align_source_to_dest(cls, source, coefs, ends, align_start_index=0):

        dest = np.full(len(source), np.nan)
        dest[source == -1] = -1

        sorted_ends, sorted_coefs = zip(*sorted(zip(ends, coefs)))

        for (start, coef) in zip((0,)+sorted_ends, coefs):
            time_mask = (source >= start)
            dest[time_mask] = cls.apply_coefficients(source[time_mask], coef)

        still_nans = np.where(np.isnan(dest))[0]
        if len(still_nans) > 0:
            if (np.array(still_nans) <= align_start_index).all():
                logger.warn('Warning: Could not align events %s' % still_nans)
                dest[np.isnan(dest)] = -1
            else:
                logger.error("Events {} could not be aligned! Session starts at event {}".format(still_nans, align_start_index))
                raise Exception('Could not convert {} times past start of session'.format(len(still_nans)))
        return dest



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
