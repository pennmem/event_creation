from system3_log_parser import BaseSys3LogParser
from fr_log_parser import FRSessionLogParser
from base_log_parser import BaseLogParser
from collections import defaultdict
import json,sqlite3
import pandas as pd

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


    _STIME_FIELD = 'timestamp'
    _TYPE_FIELD = 'event'
    _ITEM_FIELD = 'item'
    _PHASE_TYPE_FIELD = 'phase_type'
    _SERIAL_POS_FIELD = 'serialpos'
    _ONSET_FIELD = 'start_offset'
    _ID_FIELD = 'hashsum'

    def _read_primary_log(self):
        # path = "sqlite://{sqlite}".format(sqlite=self._primary_log)
        conn = sqlite3.connect(self._primary_log)
        query = 'SELECT msg FROM logs WHERE name = "events"'
        msgs = [json.loads(msg) for msg in pd.read_sql_query(query,
                                                             conn).msg.values]
        return msgs



    def event_default(self, event_json):
        event = self._empty_event
        event.mstime = event_json[self._STIME_FIELD] * 1000
        event.type = event_json[self._TYPE_FIELD]
        event.list = self._list
        event.stim_list = self._stim_list
        return event


    def __init__(self,protocol, subject, montage, experiment, session, files):
        super(FRSys3LogParser,self).__init__(protocol, subject, montage, experiment, session, files,
                                        primary_log='session_sql',allow_unparsed_events=True)

        self._list = -999
        self._stim_list = False
        self._on = False

        self._type_to_new_event= defaultdict(lambda: self.event_default,
                                             WORD_START = self.event_word,
                                             WORD_END = self.event_word_off,
                                             TRIAL = self.event_trial,
                                             ENCODING_END = self.event_reset_serialpos,
                                             RETRIEVAL_START = self.event_recall_start,
                                             RETRIEVAL_END = self.event_recall_end
                                             )
        self._add_type_to_modify_events(
            RETRIEVAL_START=self.modify_recalls
        )

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
        event.type = 'WORD'
        return event

    def event_word_off(self, event_json):
        event = super(FRSys3LogParser,self).event_word_off(event_json)
        event.type = 'WORD_OFF'
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
        'session_log':'/Users/leond/Documents/PS4_FR5/task/R1234M/session_0/session.log',
        'wordpool': '/Users/leond/Documents/PS4_FR5/task/R1234M/RAM_wordpool.txt',
        'session_sql':'/Users/leond/Documents/PS4_FR5/task/R1234M/session_0/session.sqlite'
    }

    frslp = FRSys3LogParser('r1', 'R1999X', 0.0, 'FR1', 0, files)
    events=  frslp.parse()
    pass
