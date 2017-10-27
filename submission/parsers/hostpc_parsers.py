from .base_log_parser import BaseLogParser,BaseSys3_1LogParser
from fr_sys3_log_parser import FRSys3LogParser
import pandas as pd
import json
from functools import wraps


def with_offset(event_handler):
    """
    Decorator for event handlers.
    @with_offset has the log parser set the eegoffset field of the event after the event handler is called.
    The behavior of this decorator is determined by the parser's implementation of the `set_offset` method.
    """

    @wraps(event_handler)
    def handle_and_add_offset(self,event_json):
        event = event_handler(self,event_json)
        return self.set_offset(event,event_json)
    return handle_and_add_offset

class BaseHostPCLogParser(BaseSys3_1LogParser):
    """
    This class implements the basic logic for producing event structures from event_log.json files written by the
    host PC in system 3.1+

    In order to account for possible changes to the structure of RAMulator messages, particularly between experiments
    that use pyEPL and those using UnityEPL, this parser and parsers inheriting from it should depend as *little* as
    possible on the existence of particular fields.
    """

    _TYPE_FIELD = 'event_label'
    _MSTIME_FIELD = 't_event'

    DO_ALIGNMENT = False
    ADD_STIM_EVENTS = False

    _STIM_FIELDS = (
        ('anode_label','','S64'),
        ('andode_number',-1,'int16'),
        ('cathode_label','','S64'),
        ('cathode_number',-1,'int16'),
        ('amplitude',    -1,'float16'),
        ('pulse_freq',   -1,'float16'),
        ('stim_duration',-1,'int16'),
        ('biomarker_value',-1.0,'float'),
        ('remove',True,'bool'),
        )


    def __init__(self, protocol, subject, montage, experiment, session, files,
                 primary_log='event_log', allow_unparsed_events=False, include_stim_params=False):
        super(BaseHostPCLogParser, self).__init__(protocol,subject,montage,experiment,session,files,
                                                  primary_log,allow_unparsed_events,include_stim_params)

        self._biomarker_value = -1.0
        self._stim_params = {}

    def _read_primary_log(self):
        """
        Overrides BaseSys3_1LogParser._read_primary_log
        :return: List of dicts, 1 per entry in the log
        """
        with open(self._primary_log,'r') as primary_log:
            contents = pd.DataFrame.from_records(json.load(primary_log)['events'])
            contents = pd.concat([contents,pd.DataFrame.from_records([msg.get('data',{}) for msg in contents.msg_stub])])
            return [e.to_dict() for _, e in contents.iterrows()]



    def parse(self):
        """
        BaseSys3_1LogParser has some extra logic to try different logs if the first fails; we don't want any of that.
        :return:
        """
        return BaseLogParser.parse(self)

    def set_offset(self, event,event_json):
        """
        Adds the recorded EEG offset to the event in question. `event` should be the event parsed from `event_json`
        :param event: Parsed event
        :param event_json: Unparsed event record
        :return:
        """
        event = super(BaseHostPCLogParser, self).event_default(event_json)
        event.eegoffset = event_json['offset']
        return event

    # I'm defining and adding a couple of event_type handlers here since their format is defined on the hostPC side, rather than
    # the task laptop side, and therefore their structure is independent of the task being run.

    def event_stim(self,event_json):
        event = self.event_default(event_json)
        event.type = 'STIM_ON'
        stim_params = event_json['msg_stub']['stim_channels']
        for i,stim_pair in enumerate(stim_params):
            stim_params['anode_name'] = stim_pair.split('_')[0]
            stim_params['cathode_name'] = stim_pair.split('_')[1]
            stim_params['stim_duration'] = stim_params['duration']
            self.set_event_stim_params(event,self._jacksheet,index=i,**stim_params[stim_pair])
        self._stim_params = stim_params
        return event

    def event_biomarker(self,event_json):
        event=self.event_default(event_json)
        event = self.set_event_stim_params(event,self._jacksheet,0,**event_json['msg_stub'])
        event.eegoffset = event_json['msg_stub']['start_offset']



class FRHostPCLogParser(BaseHostPCLogParser,FRSys3LogParser):

    _VALUE_FIELD = 'event_value'

    def __init__(self,protocol,subject,montage,experiment,session,files):
        BaseHostPCLogParser.__init__(self,protocol,subject,montage,experiment,session,files,allow_unparsed_events=True,
                                     include_stim_params=True)

        self._add_type_to_new_event(
            INSTRUCT = self.event_default,
            COUNTDOWN = self.event_default,
            ENCODING = self.event_trial,
            RETRIEVAL = self.event_recall,
            WORD = self.event_word,
            DISTRACT = self.event_default,
            FEATURES = self.event_default,
            WAITING = self._event_skip,
            STIM = self.event_stim,
            BIOMARKER = self.event_biomarker
        )

        self._add_type_to_modify_events(
            RETRIEVAL = self.modify_recalls,
            BIOMARKER = self.modify_biomarker,
        )
        self._biomarker_value =-1
        self._stim_on = False

    def event_stim(self,event_json):
        event = super(FRHostPCLogParser, self).event_stim(event_json)
        if self._stim_on:
            event.stim_params['biomarker_value']=self._biomarker_value
        self._stim_on = True

    def event_biomarker(self,event_json):
        super(FRHostPCLogParser, self).event_biomarker(event_json)
        self._biomarker_value = event_json['msg_stub']['biomarker_value']
        self._stim_on = True


    @with_offset
    def event_default(self, event_json):
        event = FRSys3LogParser.event_default(self,event_json)
        event.phase = self._phase
        event['mstime'] = int(event_json[self._MSTIME_FIELD]*1000)
        if event_json[self._VALUE_FIELD]:
            event['type'] = event['type']+'_START'
        else:
            event['type'] = event['type']+ '_END'

    @with_offset
    def event_word(self,event_json):
        event = self.event_default(event_json)
        self._word = event_json[self._ITEM_FIELD]
        if event_json[self._VALUE_FIELD]:
            self._stim_on = False
            event.type='WORD'
        else:
            event.type='WORD_OFF'
        self.apply_word(event)
        return event

    @with_offset
    def event_trial(self, event_json):
        self._phase = event_json[self._PHASE_TYPE_FIELD]
        event = self.event_default(event_json)
        if self._list == -1:
            self._list = 1
        else:
            self._list+=1

        return event

    @with_offset
    def event_recall(self,event_json):
        event = self.event_default(event_json)
        if event_json[self._VALUE_FIELD]:
            event.type='REC_START'
        else:
            event.type='REC_END'
        return event


    def modify_recalls(self, events):
        if events[-1].type=='REC_END':
            events= FRSys3LogParser.modify_recalls(self,events)
            if self._phase == 'STIM':
                events = self.modify_stim(events)
        return events


    # def modify_stim(self,events):
    #     in_list = events.list==self._list
    #     for event in events[in_list]:

    def modify_biomarker(self,events):
        """
        Adds biomarker value to stim events

        Note that this method assumes that stim events are immediately followed by a biomarker event.
        :param events:
        :return:
        """
        last_word = events[events.type=='WORD'][-1]
        events_to_modify = events[events.mstime>=last_word.mstime]
        events_to_modify.stim_params['biomarker_value'] = self._biomarker_value
        self._biomarker_value = -1
        events[events.mstime>=last_word.mstime]= events_to_modify
        return events




