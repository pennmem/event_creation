from .base_log_parser import BaseSys3LogParser
from .pal_log_parser import PALSessionLogParser
from collections import defaultdict
import sqlite3,json
import pandas as pd

class PALSys3LogParser(BaseSys3LogParser,PALSessionLogParser):
    def __init__(self,protocol, subject, montage, experiment, session, files):

        super(PALSys3LogParser,self).__init__(protocol, subject, montage, experiment, session, files,
                                        primary_log='session_log',allow_unparsed_events=True)

        self._type_to_new_event = defaultdict(lambda :self.event_default,

                                              STUDY_PAIR=self.event_study_pair,
                                              PROBE_START = self.event_test_probe,
                                              PROBE_END = self.event_test_probe,
                                              TRIAL = self.event_set_trial,
                                              )

        self._list = -999


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


    def event_study_pair(self, event_json):
        event=self.event_default(event_json)
        event.word1=event_json['word1']
        event.word2=event_json['word2']
        event.serialpos=event_json['serialpos']
        event.type='STUDY_PAIR' if event_json['value'] else 'STUDY_PAIR_OFF'

    def event_test_probe(self, split_line):
        event = self.event_default(split_line)
        event.probe = split_line['probe']
        event.expecting = split_line['expecting']
        event.direction  = split_line['direction']

    def event_set_trial(self,event_json):
        if self._list == -999:
            self._list = -1
        elif self._list == -1:
            self._list = 1
        else:
            self._list += 1
        return self.event_default(event_json)