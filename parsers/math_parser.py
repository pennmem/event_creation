from base_log_parser import BaseSessionLogParser, UnknownExperimentTypeException
from system2_log_parser import System2LogParser
import numpy as np
import os


class MathSessionLogParser(BaseSessionLogParser):

    _STIM_PARAM_FIELDS = System2LogParser.sys2_fields()


    @classmethod
    def _fr_fields(cls):
        """
        Returns the template for a new FR field
        Has to be a method because of call to empty_stim_params, unfortunately
        :return:
        """
        return (
            ('list', -999, 'int16'),
            ('test', '', 'U16'),
            ('answer', -999, 'int16'),
            ('iscorrect', -999, 'int16'),
            ('rectime', -999, 'int32'),
        )
        #return MatParser(MATFile)
    def __init__(self, subject, montage, files):
        super(MathSessionLogParser, self).__init__(subject, montage, files, primary_log='math_log')
        #self.fields = {field[0]: field[1] for field in fields)
        self._session = -999
        self._list = -999
        self._test = ''
        self._answer = ''
        self._iscorrect = ''
        self._rectime = -999
        self._add_fields(*self._fr_fields())
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
        event.session = self._session
        return event

    def event_E(self, split_line):
        self._list = -999
        return self.event_default(split_line)

    def event_B(self, split_line):
        if self._session == -999:
            self._session = 0
        else:
            self._session += 1
        event = self.event_default(split_line)
        return self.event_default(split_line)

    def event_start(self, split_line):
        event = self.event_default(split_line)
        if self._list == -999:
            self._list = 1
        else:
            self._list += 1
        return self.event_default(split_line)

    def event_prob(self, split_line):
        event = self.event_default(split_line)
        event.answer = int(split_line[4].replace('\'', ''))
        event.test = split_line[3].replace('=', '').replace(' ', '').replace('+', ';')
        event.iscorrect = int(split_line[5])
        rectime = int(split_line[6])
        event.rectime = rectime
        return event


def math_parser_wrapper(subject, session, experiment, base_dir='/data/eeg/', session_log_name='math.log'):

    exp_path = os.path.join(base_dir, subject, 'behavioral', experiment)
    session_log_path = os.path.join(exp_path, 'session_%d' % session, session_log_name)
    parser = MathSessionLogParser(session_log_path, subject)
    return parser


def parse_math_session_log(subject, session, experiment, base_dir='/data/eeg/', session_log_name='session.log',
                         wordpool_name='ram_wordpool_noacc.txt'):
    return math_parser_wrapper(subject, session, experiment, base_dir='/data/eeg/', session_log_name='math.log')
