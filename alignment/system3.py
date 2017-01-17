
import json
import scipy.stats
import numpy as np
import os

from copy import deepcopy
from loggers import logger
import matplotlib.pyplot as plt

from system1 import UnAlignableEEGException
from parsers.system3_log_parser import System3LogParser

from configuration import config

class System3Aligner(object):

    TASK_TIME_FIELD = 'mstime'
    ENS_TIME_FIELD = 'eegoffset'
    EEG_FILE_FIELD = 'eegfile'

    MAXIMUM_ALLOWED_RESIDUAL = 500

    FROM_LABELS = (('orig_timestamp', 1000),
                   ('t_event', 1))

    def __init__(self, events, files, plot_save_dir=None):

        self.files = files

        self.events_logs = files['event_log']

        self.electrode_config = files['electrode_config']

        self.plot_save_dir = plot_save_dir

        self.events = events
        self.merged_events = events


        for label, rate in self.FROM_LABELS:
            try:
                self.task_to_ens_coefs, self.task_ends = \
                    self.get_coefficients_from_event_log(label, 'offset', rate)
            except KeyError as key_error:
                if key_error.message != label:
                    raise
                continue
            self.from_label = label
            self.from_multiplier = rate
            break
        else:
            raise UnAlignableEEGException("Could not find sortable label in events")


        self.eeg_info = json.load(open(files['eeg_sources']))

    def add_stim_events(self, event_template, persistent_fields=lambda *_: tuple()):
        # Merge in the stim events

        s3lp = System3LogParser(self.events_logs, self.electrode_config, self.from_label, 1000 / self.from_multiplier)
        self.merged_events = s3lp.merge_events(self.events, event_template, persistent_fields)

        # Have to manually correct subject and session due to events appearing before start of session
        self.merged_events['subject'] = self.events[-1].subject
        self.merged_events['session'] = self.events[-1].session

        return self.merged_events


    def get_coefficients_from_event_log(self, from_label, to_label, rate):

        ends = []
        coefs = []

        for i, event_log in enumerate(self.events_logs):

            event_dict = json.load(open(event_log))['events']

            froms = [float(event[from_label]) * 1000. / rate for event in event_dict]
            tos = [float(event[to_label]) for event in event_dict]

            froms = np.array(froms)
            tos = np.array(tos)

            froms = froms[tos > 0]
            tos = tos[tos > 0]

            if len(froms) <= 1:
                continue

            coefs.append(scipy.stats.linregress(froms, tos)[:2])
            ends.append(froms[-1])

            self.plot_fit(froms, tos, coefs[-1], '.', 'fit{}'.format(i))
            self.check_fit(froms, tos, coefs[-1])

        if len(coefs) == 0:
            raise UnAlignableEEGException("Could not find enough events to determine coefficients!")

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
                raise UnAlignableEEGException('Could not convert {} times past start of session'.format(len(still_nans)))
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

    @classmethod
    def check_fit(cls, x, y, coefficients):
        fit = coefficients[0] * np.array(x) + coefficients[1]
        residuals = np.array(y) - fit
        if abs(1 - coefficients[0]) > .05:
            raise UnAlignableEEGException(
                "Maximum deviation from slope is .1, current slope is {}".format(coefficients[0])
            )

        if max(residuals) > cls.MAXIMUM_ALLOWED_RESIDUAL:
            logger.error("Maximum residual of fit ({}) "
                         "is higher than allowed maximum ({})".format(max(residuals), cls.MAXIMUM_ALLOWED_RESIDUAL))

            max_index = np.where(residuals == max(residuals))[0]

            logger.info("Maximum residual occurs at time={time}, sample={sample}, index={index}/{len}".format(
                time=int(x[max_index]), sample=y[max_index], index=max_index, len=len(x)
            ))
            raise UnAlignableEEGException(
                "Maximum residual of fit ({}) "
                "is higher than allowed maximum ({})".format(max(residuals), cls.MAXIMUM_ALLOWED_RESIDUAL))
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
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.plot(x, y, 'g.', x, fit, 'b-')
        plt.title("EEG Samples vs Timestamps")
        plt.xlabel("Timestamp (ms)")
        plt.ylabel("EEG Samples")
        plt.xlim(min(x), max(x))

        plt.subplot(122)
        plt.plot(x, y - fit, 'g.-', [min(x), max(x)], [0, 0], 'k-')
        plt.title("Fit residuals")
        plt.xlabel("Timestamp (ms)")
        plt.ylabel("Best-fit residuals")
        plt.xlim(min(x), max(x))
        plt.savefig(os.path.join(plot_save_dir, '{label}_fit{ext}'.format(label=plot_save_label,
                                                                                ext='.png')))
        plt.show()