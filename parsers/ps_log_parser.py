from base_log_parser import BaseSessionLogParser, UnparsableLineException
from system2_log_parser import System2LogParser
from viewers.view_recarray import strip_accents
import numpy as np
import os
import re


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


    def __init__(self, protocol, subject, montage, experiment, files):
        super(PSSessionLogParser, self).__init__(protocol, subject, montage, experiment, files,
                                                 include_stim_params=True)
        if 'jacksheet' in files:
            jacksheet_contents = [x.strip().split() for x in open(files['jacksheet']).readlines()]
            self._jacksheet = {int(x[0]): x[1] for x in jacksheet_contents}
        else:
            self._jacksheet = None
        self._exp_version = '1.0'
        self._session = -999
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
        event.type = 'STIM'
        params = {}
        params['anode_label'] = self._stim_anode_label
        params['cathode_label'] = self._stim_cathode_label
        params['amplitude'] = float(split_line[7])
        params['pulse_freq'] = int(split_line[3])
        params['burst_freq'] = int(split_line[4])
        params['n_pulses'] = int(split_line[5])
        params['n_bursts'] = int(split_line[6])
        params['pulse_width'] = self.PULSE_WIDTH
        params['stim_duration'] = 1000. * params['n_bursts'] / params['burst_freq']
        params['stim_on'] = True
        self._previous_stim_duration = params['stim_duration'] + params['n_pulses'] * params['pulse_freq']
        self.set_event_stim_params(event, self._jacksheet, **params)
        return event

    def event_stimulating(self, split_line):
        event = self.event_default(split_line)
        event.type = 'STIM'
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
        params['n_pulses'] = params['pulse_freq'] / params['stim_duration']
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
