from system3_log_parser import BaseSys3LogParser
from fr_log_parser import FRSessionLogParser
from base_log_parser import BaseLogParser
from collections import defaultdict
import json,sqlite3
import numpy as np
import pandas as pd
import os


def mark_beginning(suffix='START'):
    def with_beginning_marked(f):
        def new_f(parser,event_json):
            event = f(parser,event_json)
            try:
                if event_json['value']:
                    event.type = event.type+'_%s'%suffix
            except KeyError:
                pass
            return event
        return new_f
    return with_beginning_marked

def mark_end(suffix='END'):
    def with_beginning_marked(f):
        def new_f(parser,event_json):
            event = f(parser,event_json)
            try:
                if not event_json['value']:
                    event.type = event.type+'_%s'%suffix
            except KeyError:
                pass
            return event
        return new_f
    return with_beginning_marked




class FRSys3LogParser(BaseSys3LogParser,FRSessionLogParser):

    _STIM_FIELDS = BaseLogParser._STIM_FIELDS + (
        ('biomarker_value',-1,'float64'),
        ('id','','S64'),
        ('position','','S64')
    )

    _BASE_FIELDS = FRSessionLogParser._BASE_FIELDS + (
        ('phase','','S64'),
        ('recognized',-999,'int16'),
        ('rejected',-999,'int16'),
        ('recog_resp',-999,'int16'),
    )

    @staticmethod
    def persist_fields_during_stim(event):
        return FRSessionLogParser.persist_fields_during_stim(event)+('phase',)


    _STIME_FIELD = 'timestamp'
    _TYPE_FIELD = 'event'
    _ITEM_FIELD = 'word'
    _PHASE_TYPE_FIELD = 'phase_type'
    _SERIAL_POS_FIELD = 'serialpos'
    _ONSET_FIELD = 'start_offset'
    _ID_FIELD = 'hashsum'
    _RESPONSE_FIELD = 'yes'

    def _read_sql_logs(self):
        msgs = []
        for log in self._primary_log:
            msgs += self._read_sql_log(log)
        return msgs

    @staticmethod
    def _read_sql_log(log):
        conn = sqlite3.connect(log)
        query = 'SELECT msg FROM logs WHERE name = "events"'
        msgs = [json.loads(msg) for msg in pd.read_sql_query(query,conn).msg.values]
        return msgs


    def _read_primary_log(self):
        log = self._primary_log[0] if isinstance(self._primary_log,list) else self._primary_log
        if log is self._primary_log:
            return self._read_sql_log(log)
        else:
            return self._read_sql_logs()


    def event_default(self, event_json):
        event = self._empty_event
        event.mstime = event_json[self._STIME_FIELD]
        if self._PHASE_TYPE_FIELD in event_json:
            event.phase = event_json[self._PHASE_TYPE_FIELD]
        event.type = event_json[self._TYPE_FIELD]
        event.list = self._list
        event.stim_list = self._stim_list
        return event


    def __init__(self,protocol, subject, montage, experiment, session, files):
        super(FRSys3LogParser,self).__init__(protocol, subject, montage, experiment, session, files,
                                        primary_log='session_log',allow_unparsed_events=True)

        self._list = -999
        self._stim_list = False
        self._on = False
        self._recognition = False
        self._was_recognized = False

        self._type_to_new_event= defaultdict(lambda: self.event_default,
                                             WORD_START = self.event_word,
                                             WORD_END = self.event_word_off,
                                             TRIAL_START = self.event_trial,
                                             ENCODING_END = self.event_reset_serialpos,
                                             RETRIEVAL_START = self.event_recall_start,
                                             RETRIEVAL_END = self.event_recall_end,
                                             RECOGNITION_START = self._begin_recognition,
                                             KEYPRESS = self.event_recog,
                                             )
        self._add_type_to_modify_events(
            RETRIEVAL_START=self.modify_recalls,
            KEYPRESS = self.modify_recog,
        )

    def modify_recog(self,events):
        events = events.view(np.recarray)
        recog_event = events[-1]
        recog_word = recog_event.item_name
        rejected = not self._was_recognized if recog_event.phase =='LURE' else -999
        recognized = self._was_recognized if recog_event.phase != 'LURE' else -999
        word_mask = np.where(events.item_name==recog_word)
        new_events= events[word_mask]
        new_events.recog_resp = self._was_recognized
        new_events.rejected=rejected
        new_events.recognized = recognized
        new_events.rectime = self._recog_endtime
        events[word_mask]=new_events
        return events

    def modify_recalls(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        ann_outputs = self._parse_ann_file(str(self._list) if self._list > 0 else '0')
        for recall in ann_outputs:
            word = recall[-1]

            new_event = self._empty_event
            new_event.list = self._list
            new_event.stim_list = self._stim_list
            new_event.exp_version = self._version
            new_event.rectime = float(recall[0])
            new_event.mstime = rec_start_time + new_event.rectime
            new_event.msoffset = 20
            new_event.item_name = word
            new_event.item_num = recall[1]

            # If vocalization
            if word == '<>' or word == 'V' or word == '!':
                new_event.type = 'REC_WORD_VV'
            else:
                new_event.type = 'REC_WORD'

            # If XLI
            if recall[1] == -1:
                new_event.intrusion = -1
            else:  # Correct recall or PLI or XLI from latter list
                pres_mask = self.find_presentation(word, events)
                pres_list = np.unique(events[pres_mask].list)
                pres_mask = np.logical_and(pres_mask, events.list == self._list)

                # Correct recall or PLI
                if len(pres_list) >= 1:
                    new_event.intrusion = self._list - max(pres_list)
                    if new_event.intrusion == 0:
                        new_event.serial_pos = np.unique(events[pres_mask].serialpos)
                        new_event.recalled = True
                        if not any(events.recalled[pres_mask]):
                            events.recalled[pres_mask] = True
                            events.rectime[pres_mask] = new_event.rectime
                else:  # XLI
                    new_event.intrusion = -1

            events = np.append(events, new_event).view(np.recarray)

        return events


    def event_recog(self, event_json):
        self._was_recognized = event_json[self._RESPONSE_FIELD]
        return False

    def _begin_recognition(self,event_json):
        self._recognition = True
        return self.event_default(event_json)

    def event_reset_serialpos(self, split_line):
        return super(FRSys3LogParser, self).event_reset_serialpos(split_line)

    def event_recall_start(self,event_json):
        event = self.event_default(event_json)
        event.type = 'REC_START'
        return  event

    def event_recall_end(self,event_json):
        event = self.event_default(event_json)
        event.type = 'REC_END'
        return event

    def event_trial(self, event_json):
        if event_json[self._PHASE_TYPE_FIELD]=='PRACTICE':
            self._list=-1
        elif self._list == -1:
            self._list =1
        else:
            self._list+=1
        self._stim_list = event_json[self._PHASE_TYPE_FIELD]=='STIM'
        event = self.event_default(event_json)
        event.type='TRIAL'
        return event

    def event_word(self, event_json):
        event = self.event_default(event_json)
        event.serialpos  = event_json[self._SERIAL_POS_FIELD]
        self._word = event_json[self._ITEM_FIELD]
        event = self.apply_word(event)
        if self._recognition:
            self._recog_pres_mstime = event_json[self._STIME_FIELD]
            if event_json[self._PHASE_TYPE_FIELD] == 'LURE':
                event.type = 'RECOG_LURE'
            else:
                event.type = 'RECOG_TARGET'
        else:
            event.type='WORD'
        return event

    def event_word_off(self, event_json):
        event = self.event_default(event_json)
        event.serialpos  = event_json[self._SERIAL_POS_FIELD]
        self._word = event_json[self._ITEM_FIELD]
        event = self.apply_word(event)
        if self._recognition:
            event.type = 'RECOG_WORD_OFF'
            self._recog_endtime = event_json[self._STIME_FIELD]-self._recog_pres_mstime
        else:
            event.type = 'WORD_OFF'
        return event



class catFRSys3LogParser(FRSys3LogParser):
    _BASE_FIELDS = FRSys3LogParser._BASE_FIELDS + (
        ('category','X','S64'),
        ('category_num',-999,'int16')
    )

    _CATEGORY = 'category'
    _CATEGORY_NUM = 'category_num'


    def __init__(self,*args,**kwargs):
        super(catFRSys3LogParser,self).__init__(*args,**kwargs)
        pass

    def event_word(self, event_json):
        event = super(catFRSys3LogParser,self).event_word(event_json)
        event.category = event_json[self._CATEGORY]
        event.category_num = event_json[self._CATEGORY_NUM] if not(np.isnan(event_json[self._CATEGORY_NUM])) else -999
        return event

    def event_word_off(self, event_json):
        event = super(catFRSys3LogParser,self).event_word_off(event_json)
        event.category = event_json[self._CATEGORY]
        event.category_num = event_json[self._CATEGORY_NUM]
        return event

class RecognitionParser(BaseSys3LogParser):

    # Lures have phase-type "LURE"

    def __init__(self,*args,**kwargs):
        super(RecognitionParser, self).__init__(*args,**kwargs)

        self._type_to_new_event = defaultdict(lambda : self._event_skip)
        self._add_type_to_new_event(
            WORD = self.event_recog,
            RECOGNITION = self.begin_recognition
        )
        self._recognition = False

    def begin_recognition(self,event):
        self._recognition = True

    def event_recog(self,event_json):
        if self._recognition:
            event = self.event_default(event_json)
            event.type = 'RECOG_LURE' if event_json['phase_type']=='LURE' else 'RECOG_TARGET'
            event.response = event_json['value']

if __name__ == '__main__':
    files = {
        'session_log':'/Users/leond/Documents/R1111M/behavioral/FR5/session_0/session.log',
        'wordpool': '/Users/leond/Documents/R1111M/behavioral/FR5/RAM_wordpool.txt',
        'session_sql':'/Users/leond/Documents/R1111M/behavioral/FR5/session_0/session.sqlite'
    }

    frslp = FRSys3LogParser('r1', 'R1999X', 0.0, 'FR1', 0, files)
    events=  frslp.parse()
    pass
