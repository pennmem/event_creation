from copy import deepcopy
import json
import numpy as np

from .base_log_parser import BaseLogParser,BaseSys3LogParser
from .fr_log_parser import FRSessionLogParser
from .electrode_config_parser import ElectrodeConfig
from ..loggers import logger


class System3LogParser(object):

    _LABEL_FIELD = 'event_label'
    _VALUE_FIELD = 'event_value'
    _DEST_SORT_FIELD = 'mstime'
    _STIM_CHANNEL_FIELD = 'stim_pair'
    _ID_FIELD = 'hashtag'

    _STIM_LABEL = 'STIM'
    _STIM_ON_VALUE = 'ON'

    _STIM_PARAMS_FIELD = 'msg_stub'

    _SYS3_FIELDS = BaseLogParser._STIM_FIELDS + (
        ('host_time', -1, 'int64'),
        ('id','','S40')
    )

    _DICT_TO_FIELD = {
        'amplitude': 'amplitude',
        'pulse_freq': 'pulse_freq',
        'duration': 'stim_duration'
    }

    _STIM_ON_FIELD = 'stim_on'

    _DEFAULT_PULSE_WIDTH = 200

    SOURCE_TIME_FIELD = 't_event'
    SOURCE_TIME_MULTIPLIER = 1000


    def __init__(self, event_logs, electrode_config_files):
        self._allow_unparsed_events = True
        self.source_time_field = self.SOURCE_TIME_FIELD
        self.source_time_multiplier = self.SOURCE_TIME_MULTIPLIER

        stim_events = self._empty_event()
        for i, (log, electrode_config_file) in enumerate(zip(event_logs, electrode_config_files)):
            electrode_config = ElectrodeConfig(electrode_config_file)
            event_dict = json.load(open(log))['events']

            stim_dicts = [event for event in event_dict if event[self._LABEL_FIELD]==self._STIM_LABEL]

            for event_json in stim_dicts:

                new_event = self.make_stim_event(event_json, electrode_config)

                stim_events = np.append(stim_events, new_event)

        self._stim_events = stim_events[1:] if stim_events.shape else stim_events

        if stim_events.shape:
            logger.info("Found {} stim events".format(stim_events.shape))
        else:
            logger.warn("Found no stim events")

    @property
    def stim_events(self):
        return self._stim_events

    @classmethod
    def empty_stim_params(cls):
        return BaseLogParser.event_from_template(cls._SYS3_FIELDS).view(np.recarray)

    @staticmethod
    def event_skip(*args):
        return False

    def make_biomarker_event(self,event_json,electrode_config):
        params_dict = event_json[self._STIM_PARAMS_FIELD]
        biomarker_dict_to_field = {
            'biomarker_value':'biomarker_value',
            self._ID_FIELD:'id'
        }
        event=self.event_default(event_json)
        for k,v in biomarker_dict_to_field.items():
            event.stim_params[v] = params_dict[k]
        event.stim_params['position'] = 'POST' if 'post' in params_dict['buffer_name'] else 'NONE' # Need stim event following it to
                                                                                          # tell if biomarker creation is pre-stim
        event.stim_params = event.stim_params.view(np.recarray)
        return event.view(np.recarray)





    @classmethod
    def stim_params_template(cls):
        return (('stim_params',
                cls.empty_stim_params(),
                 BaseLogParser.dtype_from_template(cls._SYS3_FIELDS),
                 BaseLogParser.MAX_STIM_PARAMS),)

    @classmethod
    def _empty_event(cls):
        return BaseLogParser.event_from_template(cls.stim_params_template())

    def merge_events(self, events, event_template, event_to_sort_value, persistent_field_fn):

        merged_events = events[:]

        # If no stim events available
        if len(self.stim_events.shape) == 0:
            return merged_events

        for i, stim_event in enumerate(self.stim_events):
            # Get the mstime for this host event
            sort_value = event_to_sort_value(stim_event.stim_params[0])

            # Determine where to insert it in the full events structure
            after_events = (merged_events[self._DEST_SORT_FIELD] > sort_value)
            if after_events.any():
                insert_index = after_events.nonzero()[0][0]
            else:
                insert_index = merged_events.shape[0]

            # Copy the persistent fields from the previous event, modify the remaining fields
            if insert_index > 0:
                event_to_copy = merged_events[insert_index-1]
            else:
                event_to_copy = BaseLogParser.event_from_template(event_template)
            new_event = self.partial_copy(event_to_copy, event_template, persistent_field_fn)

            if 'type' in stim_event.dtype.names:
                new_event.type=stim_event['type']
            else:
                new_event.type = 'STIM_ON' if stim_event.stim_params['n_pulses'][0] > 1 else 'STIM_SINGLE_PULSE'
            new_event.stim_params = stim_event.stim_params

            new_event[self._DEST_SORT_FIELD] = sort_value

            # Insert into the events structure
            merged_events = np.append(merged_events[:insert_index],
                                      np.append(new_event, merged_events[insert_index:]))

            if new_event.type=='STIM_ON':
                # Do the same to create the stim_off_event
                stim_off_sub_event = deepcopy(stim_event.stim_params).view(np.recarray)
                stim_off_sub_event.stim_on = False
                stim_off_sub_event['host_time'] += stim_off_sub_event.stim_duration
                stim_off_time = event_to_sort_value(stim_off_sub_event)

                modify_indices = np.where(np.logical_and(merged_events[self._DEST_SORT_FIELD] > sort_value,
                                                         merged_events[self._DEST_SORT_FIELD] < stim_off_time))[0]

                # Modify the events between STIM and STIM_OFF to show that stim was applied
                for modify_index in modify_indices:
                    merged_events[modify_index].stim_params = stim_off_sub_event

                # Insert the STIM_OFF event after the modified events if any, otherwise directly after the STIM event
                insert_index = modify_indices[-1] + 1 if len(modify_indices) > 0 else insert_index + 1
                stim_off_event = self.partial_copy(merged_events[insert_index-1], event_template, persistent_field_fn)
                stim_off_event.type = 'STIM_OFF'
                stim_off_event.stim_params = stim_off_sub_event
                stim_off_event[self._DEST_SORT_FIELD] = stim_off_time

                # Merge the stim off event
                merged_events = np.append(merged_events[:insert_index],
                                          np.append(stim_off_event, merged_events[insert_index:]))
        return merged_events

    @staticmethod
    def partial_copy(event_to_copy, event_template, persistent_field_fn):
        new_event = BaseLogParser.event_from_template(event_template)
        for field in persistent_field_fn(event_to_copy):
            new_event[field] = deepcopy(event_to_copy[field])
        return new_event

    @staticmethod
    def get_n_pulses(params):
        return params['pulse_freq'] * params['stim_duration'] / (1000 * 1000)


    def make_stim_event(self, stim_dict, electrode_config):
        stim_event = self._empty_event()
        stim_params = {}
        stim_params['host_time'] = stim_dict[self.source_time_field] * self.source_time_multiplier

        for input, param in self._DICT_TO_FIELD.items():
            try:
                stim_params[param] = stim_dict[self._STIM_PARAMS_FIELD][input]
            except KeyError: #
                logger.debug('Field %s is missing'%input)

        stim_params['pulse_width'] = stim_dict['pulse_width'] if 'pulse_width' in stim_dict else self._DEFAULT_PULSE_WIDTH
        stim_params['n_pulses'] = self.get_n_pulses(stim_params)
        stim_params['stim_on'] = True

        stim_channels = electrode_config.stim_channels[stim_dict[self._STIM_PARAMS_FIELD][self._STIM_CHANNEL_FIELD]]
        anode_numbers = stim_channels.anodes
        cathode_numbers = stim_channels.cathodes

        for i, (anode_num, cathode_num) in enumerate(zip(anode_numbers, cathode_numbers)):
            BaseLogParser.set_event_stim_params(stim_event, electrode_config.as_jacksheet(), i,
                                                anode_number=anode_num,
                                                cathode_number=cathode_num,
                                                **stim_params)
        return stim_event



class Sys3EventsParser(BaseSys3LogParser):
    _STIM_FIELDS =System3LogParser._SYS3_FIELDS + (
        ('biomarker_value',-1,'float64'),
        ('position','','S10')

    )

    _ID_FIELD = 'hashtag'

    def __init__(self,protocol,subject,montage,experiment,session,files):
        super(Sys3EventsParser,self).__init__(protocol,subject,montage,experiment,session,files,
                                              primary_log='event_log',allow_unparsed_events=True,include_stim_params=True)
        self.electrode_config = ElectrodeConfig(files['electrode_config'][0])
        self.stim_parser = System3LogParser(event_logs=files['event_log'],electrode_config_files=files['electrode_config'])
        self.stim_parser._DICT_TO_FIELD.update({
            self._ID_FIELD:'id',
        })
        self._type_to_new_event={
            'STIM':self.stim_event,
            'BIOMARKER':self.biomarker_event,
            'VOCALIZATION': self.event_default
        }

        self._type_to_modify_events = {
            'STIM':self.modify_pre_stim
        }

    def event_default(self, event_json):
        event = super(Sys3EventsParser,self).event_default(event_json=event_json)
        event.host_time=event.mstime
        event.stim_params['host_time']=event.host_time
        return event

    def biomarker_event(self,event_json):
        params_dict = event_json[self.stim_parser._STIM_PARAMS_FIELD]
        biomarker_dict_to_field = {
            'biomarker_value':'biomarker_value',
            self._ID_FIELD:'id'
        }
        event=self.event_default(event_json)
        for k,v in biomarker_dict_to_field.items():
            event.stim_params[v] = params_dict[k]
        event.stim_params['position'] = 'POST' if 'post' in params_dict['buffer_name'] else 'NONE' # Need stim event following it to
                                                                                          # tell if biomarker creation is pre-stim
        event.stim_params = event.stim_params.view(np.recarray)
        return event

    def stim_event(self,event_json):
        stim_on_event = self.event_default(event_json)
        stim_params_subevent = self.stim_parser.make_stim_event(event_json,self.electrode_config)
        stim_on_event.stim_params=stim_params_subevent.stim_params.view(np.recarray)
        stim_on_event.type='STIM_ON'
        # stim_off_event= deepcopy(stim_on_event)
        # stim_off_event.stim_params['host_time']+=stim_off_event.stim_params.stim_duration
        # stim_off_event.stim_params['stim_on']=False
        # stim_off_event[self.stim_parser._DEST_SORT_FIELD]=stim_off_event.stim_params['host_time'][0]
        # stim_off_event['host_time'] = stim_off_event.stim_params['host_time'][0]
        # stim_off_event.type='STIM_OFF'
        # return np.append(stim_on_event,stim_off_event).view(np.recarray)
        return stim_on_event

    def modify_pre_stim(self,events):
        if events.shape:
            stim_event = events[-1]
            pre_stim_biomarker_events = events[((events['stim_params']['id']==stim_event['stim_params']['id'])[:,0]) & (events.type=='BIOMARKER')]
            if len(pre_stim_biomarker_events):
                pre_stim_biomarker_events.stim_params.position='PRE'
                events[((events['stim_params']['id']==stim_event['stim_params']['id'])[:,0]) & (events.type=='BIOMARKER')]= pre_stim_biomarker_events
        return events

    def parse(self):
        events = BaseSys3LogParser.parse(self).view(np.recarray)
        events.sort(order='mstime')
        for stim_event in events[events.type=='STIM_ON']:
            matched_events = events[((events.stim_params.id == stim_event.stim_params['id'])[:,0]) & (events.type=='BIOMARKER')]
            stim_fields = list(self.stim_parser.empty_stim_params().dtype.names)
            for field in ['host_time','stim_on','stim_duration']:
                stim_fields.remove(field)
            for field in stim_fields:
                matched_events.stim_params[field]=stim_event.stim_params[field]
            events[((events.stim_params.id == stim_event.stim_params['id'])[:,0]) & (events.type=='BIOMARKER')] = matched_events
        return events


#
# class VocalizationParser(BaseSys3LogParser,FRSessionLogParser):
#     def __init__(self,protocol, subject, montage, experiment, session, files):
#         super(VocalizationParser,self).__init__(protocol, subject, montage, experiment, session, files,
#                                                 primary_log='event_log',allow_unparsed_events=True)
#
#         self._type_to_new_event = {
#             'VOCALIZATION':self.event_default
#         }
#
# class BiomarkerParser(BaseSys3LogParser,FRSessionLogParser):
#     STIM_PARAMS_FIELD= 'msg_stub'
#     BIOMARKER_FIELD = 'biomarker_value'
#     ID_FIELD = 'hashtag'
#     POSITION_FIELD ='buffer_name'
#
#     _STIM_FIELDS = BaseLogParser._STIM_FIELDS + (
#             ('biomarker_value',-1.,'float64'),
#             ('id','','S40'),
#             ('pre_stim',True,'bool')
#         )
#
#     def __init__(self,*args):
#         super(BiomarkerParser,self).__init__(*args,primary_log='event_log',allow_unparsed_events=True)
#
#         self._type_to_new_event = {
#             'BIOMARKER':self.event_biomarker
#         }
#
#     def event_biomarker(self,event_json):
#         event= self.event_default(event_json)
#         json_stim_params=event_json[self.STIM_PARAMS_FIELD]
#         event.stim_params.biomarker_value = json_stim_params[self.BIOMARKER_FIELD]
#         event.stim_params.id = json_stim_params[self.ID_FIELD]
#         event.stim_params.pre_stim = 'post' not in json_stim_params[self.POSITION_FIELD]
#         return event
#
#






if __name__ == '__main__':
    import glob
    import os
    from viewers.view_recarray import pprint_rec as ppr
    # d = '/Users/iped/event_creation/tests/test_input/R9999X/behavioral/PS2/session_37/host_pc/123_45_6788'
    # event_log = os.path.join(d, 'event_log.json')
    # conf = os.path.join(d, 'config_files', 'SubjID_TwoStimChannels.csv')
    #
    # s3lp = System3LogParser([event_log], [conf])
    # ppr(s3lp.stim_events.view(np.recarray).stim_params[:,0])
    event_log = ['/Users/leond/Downloads/event_log.json.txt']
    # ann_files = glob.glob('/Volumes/rhino_root/data/eeg/R1286J/behavioral/catFR3/session_0/*.ann')
    # VP = VocalizationParser('r1','R1999X','0.0','FR5','0',{'event_log':event_log, 'annotations':ann_files})
    # events = VP.parse()
    conf = '/Volumes/rhino_root/data/eeg/R1286J/behavioral/catFR3/session_0/host_pc/20170314_165146/config_files/R1286J_14MAR2017L1M1STIM.csv'
    parser = Sys3EventsParser('r1','R1999X','0.0','FR5','0',{'event_log':event_log,'electrode_config':[conf]}) #'annotations':ann_files})
    events = parser.parse()
    pass
