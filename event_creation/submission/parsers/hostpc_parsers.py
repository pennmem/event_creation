from .base_log_parser import BaseLogParser,BaseSys3_1LogParser,BaseSessionLogParser
from ..readers.eeg_reader import read_jacksheet
from .fr_sys3_log_parser import FRSys3LogParser
import pandas as pd
import json
from functools import wraps
from copy import deepcopy
import numpy as np
from ..quality import fr_tests
import os


def with_offset(event_handler):
    """
    Decorator for event handlers.
    @with_offset has the log parser set the eegoffset field of the event after the event handler is called.
    The behavior of this decorator is determined by the parser's implementation of the `set_offset` method.
    """

    @wraps(event_handler)
    def handle_and_add_offset(self,event_json):
        event = event_handler(self,event_json)
        if event is None:
            pass
        return self.set_offset(event,event_json)
    return handle_and_add_offset


class BaseHostPCLogParser(BaseSessionLogParser):
    """
    This class implements the basic logic for producing event structures from event_log.json files written by the
    host PC in system 3.1+

    In order to account for possible changes to the structure of RAMulator messages, particularly between experiments
    that use pyEPL and those using UnityEPL, this parser and parsers inheriting from it should depend as *little* as
    possible on the existence of particular fields.
    """

    ADD_STIM_EVENTS = False

    _MESSAGE_CLASS = 'StoreEventMessage'
    _TYPE_FIELD = 'event_label'
    _MSTIME_FIELD = 'orig_timestamp'

    def __init__(self, protocol, subject, montage, experiment, session, files,
                 primary_log='event_log', allow_unparsed_events=False, include_stim_params=False):
        BaseSessionLogParser.__init__(self,protocol,subject,montage,experiment,session,files,
                                                  primary_log=primary_log,
                                                  allow_unparsed_events=allow_unparsed_events,
                                                  include_stim_params=include_stim_params)

        self.files = files
        self._phase = ''

        self._biomarker_value = -1.0
        self._stim_params = {}
        self._set_experiment_config()
        self._jacksheet = read_jacksheet(files['electrode_config'][0])

    def _read_primary_log(self):
        """
        Overrides BaseSys3_1LogParser._read_primary_log
        :return: List of dicts, 1 per entry in the log
        """
        if isinstance(self._primary_log,(str,unicode)):

            with open(self._primary_log,'r') as primary_log:
                contents = pd.DataFrame.from_records(json.load(primary_log)['events']).dropna(subset=['msg_stub']).reset_index()
                stubs =  pd.DataFrame.from_records([msg for msg in contents.msg_stub])
                messages = pd.DataFrame.from_records([msg.get('data',{}) for msg in contents.msg_stub])

                contents = pd.concat([contents,stubs,messages],
                                     axis=1)
                contents[self._MSTIME_FIELD].fillna(-1,inplace=True)
                return [e.to_dict() for _, e in contents.iterrows()]
        elif isinstance(self._primary_log,list):
            all_contents = []
            for log in self._primary_log:
                with open(log,'r') as primary_log:
                    contents = pd.DataFrame.from_records(json.load(primary_log)['events']).dropna(
                        subset=['msg_stub']).reset_index()
                messages = pd.DataFrame.from_records([msg.get('data', {}) for msg in contents.msg_stub])
                contents = pd.concat([contents, messages],
                                     axis=1)
                contents[self._MSTIME_FIELD].fillna(-1, inplace=True)
                all_contents.extend([e.to_dict() for _, e in contents.iterrows()])
            return all_contents

    def _set_experiment_config(self):
        config_file = (self.files['experiment_config'][0] if isinstance(self.files['experiment_config'],list)
                       else self.files['experiment_config'])
        with open(config_file,'r') as ecf:
            self._experiment_config = json.load(ecf)

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

        event.eegoffset = event_json['offset']
        return event

    # I'm defining and adding a couple of event_type handlers here since their format is defined on the hostPC side, rather than
    # the task laptop side, and therefore their structure is independent of the task being run.

    def event_stim(self,event_json):
        event_json[self._MSTIME_FIELD] = -1
        event = self.event_default(event_json)
        event.type = 'STIM_ON'
        stim_params = event_json['msg_stub']['stim_channels']
        new_params = {}
        for i,stim_pair in enumerate(stim_params):
            new_pair_params = {}
            new_pair_params['anode_label'] = stim_pair.split('_')[0]
            new_pair_params['cathode_label'] = stim_pair.split('_')[1]
            new_pair_params['stim_duration'] = int(stim_params[stim_pair]['duration'])
            new_pair_params['amplitude'] = int(stim_params[stim_pair]['amplitude'])
            new_pair_params['pulse_freq'] = int(stim_params[stim_pair]['pulse_freq'])
            self.set_event_stim_params(event,self._jacksheet,index=i,**new_pair_params)
            new_params[stim_pair] = new_pair_params
        self._stim_params = new_params
        return event

    @with_offset
    def event_biomarker(self,event_json):
        event=self.event_default(event_json)
        self.set_event_stim_params(event,self._jacksheet,0,**event_json['msg_stub'])
        return event

    def clean_events(self, events):
        events['protocol'] = self._protocol
        events['experiment'] = self._experiment
        events['subject'] = self._subject
        events['session'] = self._session
        events['montage'] = self._montage
        return events


class FRHostPCLogParser(BaseHostPCLogParser,FRSys3LogParser):

    _VALUE_FIELD = 'event_value'

    def __init__(self,protocol,subject,montage,experiment,session,files):
        FRSys3LogParser.__init__(self,protocol,subject,montage,experiment,session,files,primary_log='event_log')
        fields = self._fields
        BaseHostPCLogParser.__init__(self,protocol,subject,montage,experiment,session,files,allow_unparsed_events=True,
                                     include_stim_params=True)


        self._fields = fields
        self._add_type_to_new_event(
            STIM_ARTIFACT_DETECTION = self.event_default,
            INSTRUCT = self.event_default,
            COUNTDOWN = self.event_default,
            ENCODING = self.event_trial,
            RETRIEVAL = self.event_recall,
            WORD = self.event_word,
            DISTRACT = self.event_default,
            FEATURES = self._event_skip,
            WAITING = self._event_skip,
            STIM = self.event_stim,
            BIOMARKER = self._event_skip,
        )

        self._add_type_to_modify_events(
            RETRIEVAL = self.modify_recalls,
            BIOMARKER = self.modify_biomarker,
            ENCODING = self.modify_stim,
        )
        self._biomarker_value =-1
        self._stim_on = False
        self._list = -999
        self._phase = ''
        self._serialpos = -999
        self._wordpool = np.loadtxt(files['wordpool'],dtype=str)


    @property
    def stim_list(self):
        return self._phase in ('STIM','PS4')

    @with_offset
    def event_stim(self,event_json):
        event = super(FRHostPCLogParser, self).event_stim(event_json)
        self.apply_word(event)
        event.serialpos = self._serialpos
        if self._stim_on:
            event.stim_params['biomarker_value']=self._biomarker_value
        self._stim_on = True
        return event

    def event_biomarker(self,event_json):
        event = super(FRHostPCLogParser, self).event_biomarker(event_json)
        self._biomarker_value = event_json['msg_stub']['biomarker_value']
        self._stim_on = True
        return event


    @with_offset
    def event_default(self, event_json):
        event = FRSys3LogParser.event_default(self,event_json)
        event.list  = self._list
        event.phase = self._phase
        event.stim_list = self.stim_list
        if event_json[self._VALUE_FIELD]:
            event['type'] = '%s_START'%event['type']
        else:
            event['type'] = '%s_END'%event['type']
        return event

    @with_offset
    def event_word(self,event_json):
        event = self.event_default(event_json)
        event.serialpos = self._serialpos
        self._word = event_json[self._ITEM_FIELD]

        if event_json[self._VALUE_FIELD]:
            self._stim_on = False
            event.type='WORD'
        else:
            event.type='WORD_OFF'
            self._serialpos += 1
        self.apply_word(event)
        return event

    @with_offset
    def event_trial(self, event_json):
        self._phase = event_json[self._PHASE_TYPE_FIELD]
        if event_json[self._VALUE_FIELD]:

            if self._list == -999:
                self._list = -1
            else:
                self._list = int(event_json.get('current_trial',self._list))
            self._serialpos = 1
        event = self.event_default(event_json)
        return event

    @with_offset
    def event_recall(self,event_json):
        event = self.event_default(event_json)
        if event_json[self._VALUE_FIELD]:
            event.type='REC_START'
        else:
            if not event_json[self._VALUE_FIELD] and not event_json.get('current_trial'):
                if self._list == -1:
                    self._list = 1
                else:
                    self._list += 1
            event.type='REC_END'
        return event


    def modify_recalls(self, events):
        if events[-1].type=='REC_START':
            events= FRSys3LogParser.modify_recalls(self,events)
            if self._phase == 'STIM':
                events = self.modify_stim(events)
        return events


    def modify_stim(self,events):
        if events[-1].type=='ENCODING_END':
            in_list = events.list==self._list
            list_stim_events = events[(events.type=='STIM_ON') & in_list]
            list_events= events[in_list]
            for duration in np.unique([x['stim_duration'] for x in self._stim_params.values()]):
                stim_off_events = deepcopy(list_stim_events)
                stim_off_events.eegoffset += int(duration*self._experiment_config['global_settings']['sampling_rate']/1000.)
                stim_off_events.type='STIM_OFF'
                list_events = np.concatenate([list_events,stim_off_events])
                list_events.sort(order='eegoffset')
            events = np.rec.array(np.concatenate([events[~in_list],list_events]))
        return events

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


class catFRHostPCLogParser(FRHostPCLogParser):

    _catFR_FIELDS = (
        ('category','X','S64'),
        ('category_num',-999,'int16')
    )

    _CATEGORY = 'category'
    _CATEGORY_NUM = 'category_num'

    _TESTS = FRHostPCLogParser._TESTS + [fr_tests.test_catfr_categories]

    def __init__(self,*args,**kwargs):
        super(catFRHostPCLogParser, self).__init__(*args,**kwargs)
        self._add_fields(*self._catFR_FIELDS)
        self._categories = np.unique([e[self._CATEGORY] for e in self._contents if self._CATEGORY in e])
        if os.path.splitext(self.files['wordpool'])[1]:
            self._wordpool = np.loadtxt(self.files['wordpool'],dtype=str)
        else:
            self._wordpool = np.loadtxt(self.files['wordpool'], dtype=[('category','|S256'),('item_name','|S256')])
            self._wordpool = self._wordpool['item_name']


    def event_word(self, event_json):
        event = super(catFRHostPCLogParser, self).event_word(event_json)
        event.category = event_json[self._CATEGORY]
        category_num = np.where(np.in1d(self._categories,event.category))
        if len(category_num):
            event.category_num=category_num[0][0]
        return event

    def clean_events(self, events):
        """
        Final processing of events
        Here we add categories and category numbers to all the non-ELI recalls
        :param events:
        :return:
        """
        events= super(catFRHostPCLogParser, self).clean_events(events).view(np.recarray)
        is_rec = (events.type == 'REC_WORD') & (events.intrusion != -1)
        rec_events = events[is_rec]
        categories = [events[(events.item_name==r['item_name']) & (events.type=='WORD')]['category'][0] for r in rec_events]
        category_nums = [events[events.item_name == r.item_name]['category_num'][0] for r in rec_events]
        rec_events['category']=categories
        rec_events['category_num']=category_nums
        events[is_rec] = rec_events
        return events


class TiclFRParser(FRHostPCLogParser):

    event_biomarker = BaseHostPCLogParser.event_biomarker

    def __init__(self,*args,**kwargs):
        super(TiclFRParser, self).__init__(*args,**kwargs)
        self._add_type_to_new_event(BIOMARKER=self.event_biomarker)

        self.fix_content_offsets()

    def fix_content_offsets(self):
        for event_json in self._contents:
            if np.isnan(event_json['offset']):
                event_json['offset']= event_json['msg_stub']['start_offset']

    def modify_biomarker(self,events):
        return events