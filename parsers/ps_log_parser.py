from base_log_parser import BaseSessionLogParser, UnparsableLineException
from system2_log_parser import System2LogParser
import numpy as np
import re
import json


def PSLogParser(protocol, subject, montage, experiment, session,  files):
    """
    Decides which of the PS parsers to use
    :param protocol:
    :param subject:
    :param montage:
    :param experiment:
    :param files:
    :return:
    """
    if 'session_log' in files:
        return PSSessionLogParser(protocol, subject, montage, experiment, session, files)
    else:
        return PSHostLogParser(protocol, subject, montage, experiment, session, files)


class PSSessionLogParser(BaseSessionLogParser):

    _STIM_PARAM_FIELDS = System2LogParser.sys2_fields()

    PULSE_WIDTH = 300

    @classmethod
    def empty_stim_params(cls):
        """
        Makes a recarray for empty stim params (no stimulation)
        :return:
        """
        return cls.event_from_template(cls._STIM_PARAM_FIELDS)

    @classmethod
    def _ps_fields(cls):
        """
        Returns the template for a new FR field
        Has to be a method because of call to empty_stim_params, unfortunately
        :return:
        """
        return (
            ('exp_version', '', 'S16'),
            ('ad_observed', 0, 'b1'),
            ('is_stim', 0, 'b1')
        )


    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(PSSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
                                                 include_stim_params=True)
        self._exp_version = '1.0'
        self._stim_anode = None
        self._stim_cathode = None
        self._stim_anode_label = None
        self._stim_cathode_label = None
        self._stim_amplitude = None
        self._previous_stim_duration = None
        self._saw_ad = False
        self._add_fields(*self._ps_fields())
        self._add_type_to_new_event(
            STIM_LOC=self.stim_loc,
            AMPLITUDE_CONFIRMED=self.amplitude_confirmed,
            STIMULATING=self.event_stimulating,
            BEGIN_BURST=self.begin_burst,
            AFTER_DISCHARGE=self.event_ad_check,
            ADs_CHECKED=self.event_ads_checked,
            BEGIN_PS1=self.event_default,
            BEGIN_PS2=self.event_default,
            BEGIN_PS3=self.event_default,
            BURST=self._event_skip,
            END_EXP=self.event_default,
            RAM_PS=self.ram_ps,
            STIM_SINGLE_PULSE=self.event_stim_single_pulse,
            PAUSED=self.event_default,
            UNPAUSED=self.event_default
        )
        self._add_type_to_modify_events(
            BEGIN_PS1=self.begin_ps1,
            BEGIN_PS2=self.begin_ps2,
            BEGIN_PS3=self.begin_ps3,
            AFTER_DISCHARGE=self.modify_ad_check,
            STIMULATING=self.make_stim_off,
            BEGIN_BURST=self.make_stim_off,
        )

    def event_default(self, split_line):
        event = super(PSSessionLogParser, self).event_default(split_line)
        event.exp_version = self._exp_version
        return event

    def stim_loc(self, split_line):
        self._stim_anode_label = split_line[4]
        self._stim_cathode_label = split_line[6]
        #reverse_jacksheet = {v: k for k, v in self._jacksheet_dict.items()}
        #self._stim_anode = reverse_jacksheet[self._stim_anode_label]
        #self._stim_cathode = reverse_jacksheet[self._stim_cathode_label]
        return False

    def ram_ps(self, split_line):
        self._exp_version = re.sub(r'[^\d.]', '', split_line[3])
        return False

    def amplitude_confirmed(self, split_line):
        self._stim_amplitude = float(split_line[3])
        return False

    def begin_ps1(self, events):
        events.experiment = 'PS1'
        self._experiment = 'PS1'
        return events

    def begin_ps2(self, events):
        events.experiment = 'PS2'
        self._experiment = 'PS2'
        return events

    def begin_ps3(self, events):
        events.experiment = 'PS3'
        self._experiment = 'PS3'
        return events

    def begin_burst(self, split_line):
        event = self.event_default(split_line)
        event.type = 'STIM_ON'
        event.is_stim = True
        params = {}
        params['anode_label'] = self._stim_anode_label
        params['cathode_label'] = self._stim_cathode_label
        params['amplitude'] = float(split_line[7])
        params['pulse_freq'] = int(split_line[3])
        params['burst_freq'] = int(split_line[4])
        params['n_pulses'] = int(float(split_line[5]) / 1000 * params['pulse_freq'])
        params['n_bursts'] = int(split_line[6])
        params['pulse_width'] = self.PULSE_WIDTH
        params['stim_duration'] = 1000. * params['n_bursts'] / params['burst_freq']
        params['stim_on'] = True
        self._previous_stim_duration = params['stim_duration'] + params['n_pulses'] * params['pulse_freq']
        self.set_event_stim_params(event, self._jacksheet, **params)
        return event

    def event_stimulating(self, split_line):
        event = self.event_default(split_line)
        event.type = 'STIM_ON'
        event.is_stim = True
        params = {}
        params['stim_duration'] = int(split_line[5])

        if not self._stim_anode_label or not self._stim_cathode_label:
            raise UnparsableLineException('Stim occurred prior to defining stim pairs!')

        #params['anode_number'] = self._stim_anode
        #params['cathode_number'] = self._stim_cathode
        params['anode_label'] = self._stim_anode_label
        params['cathode_label'] = self._stim_cathode_label
        params['amplitude'] = float(split_line[4])
        params['pulse_freq'] = int(split_line[3])
        params['n_pulses'] = params['pulse_freq'] *  params['stim_duration'] / 1000
        params['burst_freq'] = 1
        params['n_bursts'] = 1
        params['pulse_width'] = self.PULSE_WIDTH
        params['stim_on'] = True
        self._previous_stim_duration = params['stim_duration']
        self.set_event_stim_params(event, self._jacksheet, **params)
        return event

    def event_stim_single_pulse(self, split_line):
        event = self.event_default(split_line)
        event.is_stim = True
        if not self._stim_anode_label or not self._stim_cathode_label:
            raise UnparsableLineException('Stim occurred prior to defining stim pairs!')

        self.set_event_stim_params(event, self._jacksheet,
                                   anode_label=self._stim_anode_label,
                                   cathode_label=self._stim_cathode_label,
                                   amplitude=float(split_line[3]),
                                   pulse_freq=-1,
                                   n_pulses=1,
                                   burst_freq=1,
                                   n_bursts=1,
                                   pulse_width=self.PULSE_WIDTH,
                                   stim_duration=1,
                                   stim_on=True)
        return event

    def make_stim_off(self, events):
        off_event = events[-1].copy()
        off_event.type = 'STIM_OFF'
        off_event.is_stim = False
        off_event.mstime += self._previous_stim_duration
        self.set_event_stim_params(off_event, self._jacksheet, stim_on=False)

        return np.append(events, off_event).view(np.recarray)

    def event_ads_checked(self, split_line):
        event = self.event_default(split_line)
        event.type = 'AD_CHECK'
        return event

    def event_ad_check(self, split_line):
        event = self.event_default(split_line)
        event.type = 'AD_CHECK'
        self._saw_ad = split_line[3] == 'YES'
        event.ad_observed = self._saw_ad
        return event

    def modify_ad_check(self, events):
        if not self._saw_ad:
            return events
        last_ad_check = np.where(events.type == 'AD_CHECK')[0]
        if len(last_ad_check) > 1:
            start_index = last_ad_check[-2]
        else:
            start_index = 0
        events[start_index:].ad_observed = True
        return events


class PSHostLogParser(BaseSessionLogParser):

    _TYPE_INDEX = 1
    _SPLIT_DELIMITER = '~'

    NP_TIC_RATE = 1000

    _PS_FIELDS =  (
        ('exp_version', '', 'S16'),
        ('ad_observed', 0, 'b1'),
        ('is_stim', 0, 'b1')
    )

    def __init__(self, protocol, subject, montage, experiment, session, files,
                 primary_log='host_logs', allow_unparsed_events=True, include_stim_params=True):
        super(PSHostLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
                                                   primary_log, allow_unparsed_events, include_stim_params)
        self._experiment = 'PS2.1'
        self._exp_version = '2.0'
        self._saw_ad = False
        self.beginning_marked = False

        eeg_sources = json.load(open(files['eeg_sources']))
        first_eeg_source = sorted(eeg_sources.values(), key=lambda source: source['start_time_ms'])[0]
        np_earliest_start = first_eeg_source['start_time_ms']
        self.host_offset = None

        for line in self._contents:
            if line[1] == 'NEUROPORT-TIME':
                #  np_start_host = host_time - (np_time / 30)
                #  np_start_host + host_offset = np_start_ms
                #  host_time + host_offset = ms_time
                np_start_host = int(line[0]) - int(line[2])/30
                self.host_offset = np_start_host - np_earliest_start
                break
        if not self.host_offset:
            raise UnparsableLineException("Cannot determine host offset")

        self._add_fields(*self._PS_FIELDS)

        self._add_type_to_new_event(
            SHAM=self.event_default,
            AD_CHECK=self.event_ad_check,
            PS2=self.event_version

        )
        self._add_type_to_new_event(**{
            'NEUROPORT-TIME': self.mark_beginning
        })

        self._add_type_to_modify_events(
            PS2=self.remove_previous_starts,
            AD_CHECK=self.modify_ad_check,
        )

    def clean_events(self, events):
        events = events.view(np.recarray)
        events.protocol = self._protocol
        events.montage = self._montage
        events.experiment = self._experiment
        events.exp_version = self._exp_version

        stim_events = np.logical_or(events['type'] == 'STIM', events['type'] == 'STIM_OFF')
        stim_events = np.logical_or(stim_events, events['type'] == 'SHAM')
        stim_event_indices = np.where(stim_events)[0]

        poll_events = np.where(events['type'] == 'NP_POLL')[0]
        first_poll_event = poll_events[0]
        last_poll_event = poll_events[-1]

        # Need the last two events (on/off) before the first np poll and the two events after the last np poll
        stim_before = np.array([index for index in stim_event_indices if index < first_poll_event - 2])
        stim_after = np.array([index for index in stim_event_indices if index > last_poll_event + 2])

        good_range = np.array([index for index in range(len(events)) \
                               if index not in stim_before and index not in stim_after])

        cleaned_events = events[good_range]
        # Remove NP_POLL
        cleaned_events = cleaned_events[cleaned_events['type'] != 'NP_POLL']
        cleaned_events.sort(order='mstime')
        return cleaned_events

    def mark_beginning(self, split_line):
        event = self.event_default(split_line)
        event['type'] = 'NP_POLL'
        return event

    def read_primary_log(self):
        if isinstance(self._primary_log, list):
            self.host_log_files = sorted(self._primary_log)
        else:
            self.host_log_files = [self._primary_log]
        contents = []
        for host_log_file in self.host_log_files:
            contents += [line.strip().split(self._SPLIT_DELIMITER)
                         for line in open(host_log_file).readlines()]
        return contents

    @staticmethod
    def persist_fields_during_stim(event):
        return ('protocol', 'subject', 'montage', 'experiment', 'session', 'eegfile', 'exp_version')

    def event_default(self, split_line):
        event = self._empty_event
        event.mstime = int(split_line[0]) - self.host_offset
        event.type = split_line[1]
        event.exp_version = self._exp_version
        return event.view(np.recarray)

    def event_version(self, split_line):
        self._exp_version = re.sub(r'[^\d.]', '', split_line[2])
        event = self.event_default(split_line)
        event.type = 'SESS_START'
        return event

    def remove_previous_starts(self, events):
        starts = events.type == 'SESS_START'
        if np.count_nonzero(starts) > 1:
            starts[-1] = False
            events = events[np.logical_not(starts)]
        return events


    def event_ad_check(self, split_line):
        event = self.event_default(split_line)
        self._saw_ad =  not split_line[2] == 'NOT_OBSERVED'
        event.ad_observed = self._saw_ad
        return event

    def modify_ad_check(self, events):
        if not self._saw_ad:
            return events
        last_ad_check = np.where(events.type == 'AD_CHECK')[0]
        if len(last_ad_check) > 1:
            start_index = last_ad_check[-2]
        else:
            start_index = 0
        events[start_index:].ad_observed = True
        return events