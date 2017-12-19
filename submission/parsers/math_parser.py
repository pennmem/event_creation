from .base_log_parser import BaseSessionLogParser, UnknownExperimentError
from .system2_log_parser import System2LogParser
import numpy as np
import os


class MathLogParser(BaseSessionLogParser):

    _STIM_PARAM_FIELDS = System2LogParser.sys2_fields()

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

            ('badEpoch', -1, 'int8'),
            ('artifactChannels', -1, 'int8', 128),
            ('variance', np.nan, 'float', 128),
            ('medGradient', np.nan, 'float', 128),
            ('ampRange', np.nan, 'float', 128),
            ('iqrDevMax', np.nan, 'float', 128),
            ('iqrDevMin', np.nan, 'float', 128),
            ('eogArtifact', -1, 'int8')
        )

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(MathLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
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
        event.answer = int(split_line[4].replace('\'', ''))
        test = split_line[3].replace('=', '').replace(' ', '').strip("'").split('+')
        event.test[0] = int(test[0])
        event.test[1] = int(test[1])
        event.test[2] = int(test[2])
        event.iscorrect = int(split_line[5])
        rectime = int(split_line[6])
        event.rectime = rectime
        return event
