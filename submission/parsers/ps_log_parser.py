from .base_log_parser import BaseSessionLogParser, LogParseError, BaseSys3LogParser
from .system2_log_parser import System2LogParser
import numpy as np
import re
import json
from .electrode_config_parser import ElectrodeConfig
from ..alignment.system3 import System3Aligner
import codecs

def PSLogParser(protocol, subject, montage, experiment, session, files):
    """
    Decides which of the PS parsers to use
    :param protocol:
    :param subject:
    :param montage:
    :param experiment:
    :param files:
    :return:
    """
    if 'event_log' in files:
        if 'PS4' in experiment:
            return PS4Sys3LogParser(protocol,subject,montage,experiment,session,files)
        else:
            return PSSys3LogParser(protocol, subject, montage, experiment, session, files)
    elif 'session_log' in files:
        return PSSessionLogParser(protocol, subject, montage, experiment, session, files)
    elif 'host_logs' in files:
        return PSHostLogParser(protocol, subject, montage, experiment, session, files)
    else:
        raise Exception("Could not determine system 1, 2, or 3 from inputs")


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
        if params['amplitude'] < 10:
            params['amplitude'] *= 1000
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
            raise LogParseError('Stim occurred prior to defining stim pairs!')

        #params['anode_number'] = self._stim_anode
        #params['cathode_number'] = self._stim_cathode
        params['anode_label'] = self._stim_anode_label
        params['cathode_label'] = self._stim_cathode_label
        params['amplitude'] = float(split_line[4])
        if params['amplitude'] < 10:
            params['amplitude'] *= 1000
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
            raise LogParseError('Stim occurred prior to defining stim pairs!')

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
            raise LogParseError("Cannot determine host offset")

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

    def _read_primary_log(self):
        if isinstance(self._primary_log, list):
            self.host_log_files = sorted(self._primary_log)
        else:
            self.host_log_files = [self._primary_log]
        contents = []
        for host_log_file in self.host_log_files:
            contents += [line.strip().split(self._SPLIT_DELIMITER)
                         for line in codecs.open(host_log_file,encoding='utf8').readlines()]
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

class PSSys3LogParser(BaseSys3LogParser):

    _PS_FIELDS =  (
        ('exp_version', '', 'S16'),
        ('ad_observed', -1, 'int'),
        ('is_stim', 0, 'b1')
    )

    def __init__(self, protocol, subject, montage, experiment, session, files,
                 primary_log='event_log', allow_unparsed_events=True, include_stim_params=True):
        super(PSSys3LogParser, self).__init__(protocol, subject, montage, experiment, session,
                                              files, primary_log, allow_unparsed_events, include_stim_params)

        self._experiment = 'PS2.1'
        self._exp_version = '3.0'
        self._saw_ad = False

        self._add_type_to_new_event(
            STIM=self._event_skip, # Skip because it is added later with alignment
            SHAM=self.event_default,
            PAUSED=self.event_paused,
        )

    def event_paused(self, event_json):
        if event_json['event_value'] == 'OFF':
            return False
        event = self.event_default(event_json)
        event.type = 'AD_CHECK'
        return event


class PS4Sys3LogParser(BaseSys3LogParser):

    _frequency = 200
    _ID_FIELD = 'hashtag'
    _STIM_PARAMS_FIELD = 'msg_stub'
    _DELTA_CLASSIFIER_FIELD = 'stim_delta_classifier'
    ADD_STIM_EVENTS = False

    _LOC_FIELDS = (
        ('loc_name','','S16'),
        ('amplitude',-999,'float64'),
        ('delta_classifier',-999,'float64'),
        ('sem',-999,'float64'),
        ('snr',-999,'float64')
    )

    _SHAM_FIELDS = (
        ('delta_classifier',-999,'float64'),
        ('sem',-999,'float64'),
        ('p_val',-999,'float64',),
        ('t_stat',-999,'float64',),
    )

    _DECISION_FIELDS = (
        ('p_val',-999.0, 'float64'),
        ('t_stat',-999.0,'float64'),
        ('best_location','','S16'),
        ('tie',-1,'int16'),

    )

    _BASE_FIELDS = BaseSys3LogParser._BASE_FIELDS + (
        ('id','XXX','S64'),
        ('list',-999,'int16'),
        ('biomarker_value',-999,'float64'),
        ('position','None','S10'),
        ('delta_classifier',-999,'float64'),
        ('anode_label','','S20'),
        ('cathode_label','','S20'),
        ('anode_num',-999,'int16'),
        ('cathode_num',-999,'int16'),
        ('amplitude',-999,'int16'),
        ('list_phase','','S16'),
        ('loc1', BaseSessionLogParser.event_from_template(_LOC_FIELDS),
         BaseSessionLogParser.dtype_from_template(_LOC_FIELDS)),
        ('loc2', BaseSessionLogParser.event_from_template(_LOC_FIELDS),
         BaseSessionLogParser.dtype_from_template(_LOC_FIELDS)),
        ('sham', BaseSessionLogParser.event_from_template(_SHAM_FIELDS),
         BaseSessionLogParser.dtype_from_template(_SHAM_FIELDS)),
        ('decision',BaseSessionLogParser.event_from_template(_DECISION_FIELDS),BaseSessionLogParser.dtype_from_template(_DECISION_FIELDS))

    )


    def __init__(self,protocol,subject,montage,experiment,session,files,
                 primary_log = 'event_log',allow_unparsed_events=True,include_stim_params=True):
        super(PS4Sys3LogParser,self).__init__(
            protocol,subject,montage,experiment,session,files,primary_log,allow_unparsed_events,include_stim_params)
        self._files = files
        electrode_config_files = files['electrode_config']
        if not isinstance(electrode_config_files,list):
            electrode_config_files = [electrode_config_files]
        self._electrode_config = ElectrodeConfig(electrode_config_files[0])
        self._add_type_to_new_event(
            FEATURES = self.event_features,
            BIOMARKER= self.event_biomarker,
            OPTIMIZATION = self.event_optimization,
            OPTIMIZATION_DECISION = self.event_decision,
            PS_OPTIMIZATION_DECISION_NOT_POSSIBLE =  self.event_default,
            ENCODING = self.encoding,
            DISTRACT = self.distract,
            RETRIEVAL = self.retrieval,
            STIM=self.event_stim,
            TRIAL  = self.event_trial,
        )
        self._add_type_to_modify_events(
            STIM = self.modify_with_stim_params,
            BIOMARKER = self.modify_with_stim_params,
            OPTIMIZATION = self.modify_with_stim_params
        )

        self._list = -999
        self._list_phase = 'INSTRUCT'
        self._anode = ''
        self._cathode = ''
        self._anode_num = -999
        self._cathode_num = -999
        self._amplitude = -999
        self._id = -999


    def modify_with_stim_params(self,events):
        if events.shape:
            event_id = events[-1]['id']
            matches = [events['id']==event_id]
            new_events = self.apply_stim_params(events[matches])
            events[matches] = new_events
        return events



    def apply_stim_params(self,event):
        event.anode_label  = self._anode
        event.cathode_label = self._cathode
        event.anode_num = self._anode_num
        event.cathode_num = self._cathode_num
        event.amplitude = self._amplitude
        event.frequency = self._frequency
        return event

    def event_default(self, event_json):
        event  = super(PS4Sys3LogParser,self).event_default(event_json)
        event.list = self._list
        event.list_phase = self._list_phase
        return event

    def encoding(self,event_json):
        self._list_phase = 'ENCODING'
        return False

    def distract(self,event_json):
        self._list_phase = 'DISTRACT'
        return False

    def retrieval(self,event_json):
        self._list_phase = 'RETRIEVAL'
        return False

    def event_features(self,event_json):
        event = self.event_default(event_json)
        event.id = event_json[self._STIM_PARAMS_FIELD][self._ID_FIELD]
        params_dict = event_json[self._STIM_PARAMS_FIELD]
        event.position = 'POST' if 'post' in params_dict['buffer_name'] else 'PRE'
        event.eegoffset = event_json[self._STIM_PARAMS_FIELD]['start_offset']
        event.list_phase = event_json[self._STIM_PARAMS_FIELD]['buffer_name'].partition('_')[0].upper()
        return event


    def event_biomarker(self,event_json):

        params_dict = event_json[self._STIM_PARAMS_FIELD]
        biomarker_dict_to_field = {
            'biomarker_value':'biomarker_value',
            self._ID_FIELD:'id'
        }
        event=self.event_default(event_json)
        for k,v in biomarker_dict_to_field.items():
            event[v] = params_dict[k]
        event.eegoffset = event_json[self._STIM_PARAMS_FIELD]['start_offset']
        event['position'] = 'POST' if 'post' in params_dict['buffer_name'] else 'PRE'
        self._list_phase = event_json[self._STIM_PARAMS_FIELD]['buffer_name'].partition('_')[0].upper()
        self._id = event_json[self._STIM_PARAMS_FIELD][self._ID_FIELD]
        event.list_phase = self._list_phase
        event.id = self._id
        return event.view(np.recarray)


    def event_optimization(self,event_json):
        event=self.event_default(event_json)
        event.id = event_json[self._STIM_PARAMS_FIELD][self._ID_FIELD]
        event.delta_classifier = event_json[self._STIM_PARAMS_FIELD][self._DELTA_CLASSIFIER_FIELD]
        return event

    def event_stim(self,event_json):
        if event_json['event_value']:
            try:
                stim_params_dict = event_json[self._STIM_PARAMS_FIELD]
                self._amplitude = stim_params_dict['amplitude']
                self._frequency = stim_params_dict['pulse_freq'] /1000
                stim_pair = stim_params_dict['stim_pair']
                self._anode, self._cathode = stim_pair.split('_')

            except KeyError:
                for stim_pair in stim_params_dict['stim_channels']:
                    self._anode,self._cathode = stim_pair.split('_')
                    self._amplitude = stim_params_dict['stim_channels'][stim_pair]['amplitude']
                    self._frequency = stim_params_dict['stim_channels'][stim_pair]['pulse_freq'] / 1000
            self._anode_num,self._cathode_num = [self._electrode_config.contacts[c].jack_num for c in (self._anode,self._cathode)]
        event = self.event_default(event_json)
        event.list_phase = self._list_phase
        event.id = self._id
        event.type = 'STIM_ON' if event_json['event_value'] else 'STIM_OFF'
        return event

    def event_decision(self,event_json):
        event = self.event_default(event_json)
        loc_dict_to_field={
            'sem':'sem',
            'SNR':'snr',
            'best_amplitude':'amplitude',
            'best_delta_classifier':'delta_classifier'
        }
        for i,k in enumerate(event_json[self._STIM_PARAMS_FIELD]['loc_info_list']):
            v = 'loc1' if i==1 else 'loc2'
            for dk in loc_dict_to_field:
                event[v][loc_dict_to_field[dk]] = event_json[self._STIM_PARAMS_FIELD]['loc_info_list'][k][dk]
            event[v]['loc_name']=k

        sham_dict_to_field = {
            'sham_delta_classifier':'delta_classifier',
            'sham_sem':'sem',
            'p_val_champ_sham':'p_val',
            't_stat_champ_sham':'t_stat',
        }
        for d in sham_dict_to_field:
            event.sham[sham_dict_to_field[d]]=event_json[self._STIM_PARAMS_FIELD]['decision'][d]

        decision_dict_from_fields = {
            'p_val':'p_val','t_stat':'t_stat','best_location':'best_location_name','tie':'Tie'
        }
        for k,v in decision_dict_from_fields.items():
            event.decision[k] = event_json[self._STIM_PARAMS_FIELD]['decision'][v]
        return event

    def event_trial(self,event_json):
        if event_json['event_value']:
            list = event_json[self._STIM_PARAMS_FIELD]['data']['listno']
            if list==0:
                self._list = -1
            else:
                self._list = list
        return False

    def clean_events(self, events):
        aligner = System3Aligner(events,self._files)
        events['mstime'] =aligner.apply_coefficients_backwards(events['eegoffset'],aligner.task_to_ens_coefs[0])
        aligner.apply_eeg_file(events)
        return events
