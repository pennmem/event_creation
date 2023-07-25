from .base_log_parser import BaseSessionLogParser,BaseSys3_1LogParser
from .elemem_parsers import BaseElememLogParser
from .system2_log_parser import System2LogParser
import pandas as pd
import numpy as np
from ..exc import UnknownExperimentError
import re
from ..log import logger


def MathLogParser(protocol,subject,montage,experiment,session,files):
    """
    As with PSLogParser, we're using an additional level of indirection to choose which math log parser is
    most appropriate.
    :param protocol:
    :param subject:
    :param montage:
    :param experiment:
    :param session:
    :param files:
    :return: Union[MathSessionLogParser,MathUnityLogParser]
    """
    logger.debug("Files given to MathLogParser = {}".format(files))
    if 'math_log' in files:
        return MathSessionLogParser(protocol,subject,montage,experiment,session,files)
    elif 'session_log_json' in files:
        return MathUnityLogParser(protocol,subject,montage,experiment,session,files)
    elif 'elemem_files' in files:          # conditioning on 'event_log' would also match system 3 (ok because of logic order)
        return MathElememLogParser(protocol, subject, montage, experiment, session, files)
    else:
        raise UnknownExperimentError('Uknown math log file')


class MathSessionLogParser(BaseSessionLogParser):

    _STIM_PARAM_FIELDS = System2LogParser.sys2_fields()

    ADD_STIM_EVENTS = False

    @classmethod
    def _math_fields(cls):
        """
        Returns the template for a new FR field
        Has to be a method because of call to empty_stim_params, unfortunately
        :return:
        """
        return (
            ('list', -999, 'int16'),
            ('test', -999, 'int16', 3),
            ('answer', -999, 'int16'),
            ('iscorrect', -999, 'int16'),
            ('rectime', -999, 'int32'),
        )

    @classmethod
    def _math_fields_ltp(cls):
        """
        Used instead of _math_fields for LTP behavioral.
        """
        return (
            ('list', -999, 'int16'),
            ('test', -999, 'int16', 3),
            ('answer', -999, 'int16'),
            ('iscorrect', -999, 'int16'),
            ('rectime', -999, 'int32'),
            ('eogArtifact', -1, 'int8')
        )

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(MathSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
                                                   primary_log='math_log')
        self._list = -999
        self._test = ''
        self._answer = ''
        self._iscorrect = ''
        self._rectime = -999
        self._add_fields(*self._math_fields_ltp()) if protocol == 'ltp' else self._add_fields(*self._math_fields())
        self._add_type_to_new_event(
            START=self.event_start,
            STOP=self.event_default,
            PROB=self.event_prob
        )

    def event_default(self, split_line):
        """
        Override base class's default event to include list, serial position, and stimList
        :param split_line:
        :return:
        """
        event = BaseSessionLogParser.event_default(self, split_line)
        event.list = self._list
        return event

    def event_start(self, split_line):
        if self._list == -999:
            self._list = -1
        elif self._list ==-1:
            self._list=1
        else:
            self._list += 1
        return self.event_default(split_line)

    def event_prob(self, split_line):
        event = self.event_default(split_line)
        answer = int(split_line[4].replace('\'', ''))
        if np.iinfo(np.int16).min <= answer <= np.iinfo(np.int16).max:
            event.answer = answer
        test = split_line[3].replace('=', '').replace(' ', '').strip("'").split('+')
        event.test[0] = int(test[0])
        event.test[1] = int(test[1])
        event.test[2] = int(test[2])
        event.iscorrect = int(split_line[5])
        rectime = int(split_line[6])
        event.rectime = rectime
        return event



class MathUnityLogParser(BaseSys3_1LogParser):

    ADD_STIM_EVENTS = False

    ANSWER_FIELD='response'
    TEST_FIELD = 'problem'
    RECTIME_FIELD = 'response_time_ms'
    _STATE_NAMES = ['DISTRACT',]

    @classmethod
    def _read_unityepl_log(cls, filename):
        """
        Need to overload this method because the math events are formatted differently from the WORD events.
        :param filename: path to session.json
        :return: List of dicts containing necessary info
        """
        df = pd.read_json(filename, lines=True)
        messages = df[df['type'] == 'network'].data
        to_use = ['type' in msg['message'] and (
            msg['message']['type']=='MATH'
            or (msg['message']['type']=='STATE' and msg['message']['data']['name'] in cls._STATE_NAMES)
        )
                  for msg in messages ]
        used_messages = pd.DataFrame.from_records([msg['message']  for msg in messages.loc[to_use]])
        data = pd.DataFrame.from_records([data for data in used_messages.data])
        data[cls._MSTIME_FIELD] = used_messages.time.values.astype(int)
        data[cls._TYPE_FIELD] = data['name'].where(~data['name'].isnull(),used_messages['type'])
        return [e.to_dict() for _, e in data.iterrows()]


    def __init__(self,protocol,subject,montage,experiment,session,files):
        super(MathUnityLogParser, self).__init__(protocol,subject,montage,experiment,session,files,
                                                 primary_log='session_log_json')

        self._add_fields(*MathSessionLogParser._math_fields())
        self._add_type_to_new_event(
            MATH=self.events_math,
            DISTRACT = self.event_distract,
        )

        self._list = -1

    def event_default(self, event_json):
        event = super(MathUnityLogParser, self).event_default(event_json)
        event['list'] = self._list
        return event

    def event_distract(self, event_json):
        self._phase = event_json[self._PHASE_TYPE_FIELD]
        event = self.event_default(event_json)
        if event_json['value']:
            event['type'] = 'START'
        else :
            event['type']='STOP'
            if self._list == -1:
                self._list = 1
            else:
                self._list += 1
        return event

    def events_math(self,event_json):
        event = self.event_default(event_json)
        event['type'] = 'PROB'
        event['answer'] = int(event_json[self.ANSWER_FIELD])
        problem = [int(x.strip()) for x in event_json[self.TEST_FIELD].replace('=','').replace(' ','').split('+')]
        event['test']  = problem
        event['iscorrect'] = int(event['answer']==sum(problem))
        event['rectime'] = int(event_json[self.RECTIME_FIELD])
        event['mstime'] = event_json[self._MSTIME_FIELD] - event['rectime']

        return event


    @classmethod
    def test(cls,unity_log):
        files = {'session_log':unity_log}
        parser = MathUnityLogParser('r1','R1999X',0,'math_test',-1,files)
        events = parser.parse()
        return events
    

class MathElememLogParser(BaseElememLogParser):    # parse events.log for math/distractor events
    ANSWER_FIELD = 'response'
    TEST_FIELD = 'problem'
    RECTIME_FIELD = 'response_time_ms'
    ISCORRECT_FIELD = 'correct'
    _MSTIME_FIELD = 'time'
    

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(MathElememLogParser, self).__init__(protocol, subject, montage, experiment, session, files, 
                                                  primary_log='event_log')
        self._add_fields(*MathSessionLogParser._math_fields())             # use other classes method for math columns in events
        self._add_type_to_new_event(
            MATH = self.events_math,
            DISTRACT = self.events_distract,
        )

    # override method in BaseElememLogParser -> only reading MATH and DISTRACT events
    def _read_event_log(self, filename):
        df = pd.read_json(filename, lines=True)
        md = df[(df['type']=='MATH') | (df['type']=='DISTRACT')]       # math and distract events
        # have to convert from dataframe to grab necessary info
        md_ra = [{'answer': int(row.data[self.ANSWER_FIELD]), 
                  'test': [int(x) for x in re.findall(r'\d+', row.data[self.TEST_FIELD])], 
                  'iscorrect': int(bool(row.data[self.ISCORRECT_FIELD])), 
                  'rectime': int(row.data[self.RECTIME_FIELD]), 
                  'mstime': int(row.time - int(row.data[self.RECTIME_FIELD]))} if row.type == 'MATH' else 
                  {'answer': -999, 'test': [0,0,0], 'iscorrect': -999, 'mstime': int(row.time)} if row.type == 'DISTRACT' else 
                  {} for _, row in md.iterrows()]
        df_md = pd.DataFrame.from_records(md_ra)       # convert back to dataframe to add fields
        df_md['subject'] = self._subject
        df_md['experiment'] = self._experiment
        df_md['protocol'] = self._protocol
        df_md['montage'] = self._montage
        return [e.to_dict() for _, e in df_md.iterrows()]

        
    def events_distract(self, event_json):
        event = self.event_default(event_json)
        event['type'] = 'DISTRACT'                   # only distractor start (no stop) for system 4 logs
        return event
    
    def events_math(self, event_json):
        event = self.event_default(event_json)
        event['type'] = 'MATH'                       # probably not best that this is different from system 3 'PROB'
        event['answer'] = int(event_json[self.ANSWER_FIELD])
        event['test'] = [int(x) for x in re.findall(r'\d+', event_json[self.TEST_FIELD])]
        event['iscorrect'] = int(event['answer'] == sum(event['test']))
        event['rectime'] = int(event_json[self.RECTIME_FIELD])
        event['mstime'] = event_json[self._MSTIME_FIELD] - event['rectime']     # timestamp of start of math problem
        return event