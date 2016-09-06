from base_log_parser import BaseSessionLogParser, UnknownExperimentTypeException
from system2_log_parser import System2LogParser
import numpy as np
import os


class LTPFRSessionLogParser(BaseSessionLogParser):

    _STIM_PARAM_FIELDS = System2LogParser.sys2_fields()

    @classmethod
    def empty_stim_params(cls):
        """
        Makes a recarray for empty stim params (no stimulation)
        :return:
        """
        return cls.event_from_template(cls._STIM_PARAM_FIELDS)

    @classmethod
    def _fr_fields(cls):
        """
        Returns the template for a new FR field
        Has to be a method because of call to empty_stim_params, unfortunately
        :return:
        """
        return (
            ('list', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('word', 'X', 'S16'),
            ('wordno', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('rectime', -999, 'int16'),
            ('intrusion', -999, 'int16'),
            ('expVersion', '', 'S16'),
            ('stimList', False, 'b1'),
            ('isStim', False, 'b1'),
            ('stimParams', cls.empty_stim_params(), cls.dtype_from_template(cls._STIM_PARAM_FIELDS)),
        )

    def __init__(self, session_log, wordpool_file, subject, ann_dir=None):
        BaseSessionLogParser.__init__(self, session_log, subject, ann_dir)
        self._wordpool = np.array([x.strip() for x in open(wordpool_file).readlines()])
        self._session = -999
        self._list = -999
        self._serialpos = -999
        self._stimList = False
        self._word = ''
        self._version = ''
        self._add_fields(*self._fr_fields())
        self._add_type_to_new_event(
            INSTRUCT_VIDEO=self.event_instruct_video,
            SESS_START=self.event_sess_start,
            FR_PRES=self.event_fr_pres,
            SLOW_MSG = self.event_default ,
            PRACTICE_WORD=self.event_practice_word,
            PRACTICE_WORD_OFF=self.event_practice_word_off,
            WORD=self.event_word,
            WORD_OFF=self.event_word_off,
            REC_START=self.event_rest,
            REST=self.event_skip,
            SESS_END=self.event_default,
            SESSION_SKIPPED=self.event_default
        )
        self._add_type_to_modify_events(
            SESS_START=self.modify_session,
            REC_START=self.modify_recalls
        )

    def event_default(self, split_line):
        """
        Override base class's default event to include list, serial position, and stimList
        :param split_line:
        :return:
        """
        event = BaseSessionLogParser.event_default(self, split_line)
        event.session = self._session
        event.stimList = self._stimList
        event.expVersion = self._version
        return event

    def event_fr_pres(self, split_line):
        event = self.event_default(split_line)
        event.type = 'WORD'
        return event

    def event_sess_start(self, split_line):
        self._session = int(split_line[3]) - 1
        return self.event_default(split_line)

    def modify_session(self, events):
        """
        applies session and expVersion to all previous events
        :param events: all events up until this point in the log file
        """
        events.session = self._session
        events.expVersion = self._version
        return events

    def event_practice_trial(self, split_line):
        self._list = -1
        self._serialpos = -1
        event = self.event_default(split_line)
        return event

    def apply_word(self, event):
        event.word = self._word
        if (self._wordpool == self._word).any():
            wordno = np.where(self._wordpool == self._word)
            event.wordno = wordno[0] + 1
        else:
            event.wordno = -1
        return event

    def event_practice_word(self, split_line):
        self._serialpos += 1
        event = self.event_default(split_line)
        self._word = split_line[3]
        event.serialpos = self._serialpos
        event = self.apply_word(event)
        return event

    def event_practice_word_off(self, split_line):
        event = self.event_default(split_line)
        event = self.apply_word(event)
        return event

    def event_trial(self, split_line):
        self._list = int(split_line[3])
        self._stimList = split_line[4] == 'STIM'
        return self.event_default(split_line)

    def event_word(self, split_line):
        self._word = split_line[4]
        self._serialpos = int(split_line[5]) + 1
        event = self.event_default(split_line)
        event = self.apply_word(event)
        event.serialpos = self._serialpos
        return event

    def event_word_off(self, split_line):
        event = self.event_default(split_line)
        event = self.apply_word(event)
        event.serialpos = self._serialpos
        return event

    def modify_recalls(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        ann_outputs = self._parse_ann_file(str(self._list - 1) if self._trial > 1 else str(self._trial))
        for recall in ann_outputs:
            word = recall[-1]

            new_event = self._empty_event
            new_event.list = self._list
            new_event.session = self._session
            new_event.stimList = self._stimList
            new_event.expVersion = self._version
            new_event.rectime = float(recall[0])
            new_event.mstime = rec_start_time + new_event.rectime
            new_event.msoffset = 20
            new_event.word = word
            new_event.wordno = recall[1]

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

                # Correct recall or PLI
                if len(pres_list) == 1:
                    new_event.intrusion = self._list - pres_list[0]
                    if new_event.intrusion == 0:
                        new_event.serialpos = np.unique(events[pres_mask].serialpos)
                        new_event.recalled = True
                        if not any(events.recalled[pres_mask]):
                            events.recalled[pres_mask] = True
                            events.rectime[pres_mask] = new_event.rectime
                else:  # XLI
                    new_event.intrusion = -1

            events = np.append(events, new_event).view(np.recarray)

        return events

    @staticmethod
    def find_presentation(word, events):
        events = events.view(np.recarray)
        return np.logical_and(events.word == word, events.type == 'WORD')