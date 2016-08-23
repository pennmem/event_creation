from base_log_parser import BaseSessionLogParser, UnknownExperimentTypeException
from system2_log_parser import System2LogParser
from viewers.view_recarray import strip_accents
import numpy as np
import os


class CatFRSessionLogParser(BaseSessionLogParser):

    _STIM_PARAM_FIELDS = System2LogParser.sys2_fields()

    @classmethod
    def empty_stim_params(cls):
        """
        Makes a recarray for empty stim params (no stimulation)
        :return:
        """
        return cls.event_from_template(cls._STIM_PARAM_FIELDS)

    @classmethod
    def _catfr_fields(cls):
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
            ('recalled', False, 'int16'),
            ('rectime', -999, 'int32'),
            ('expVersion', -1, 'S16'),
            ('intrusion', -999, 'int16'),
            ('isStim', False, 'b1'),
            ('category', 'X', 'S16'),
            ('categoryNum', -999, 'int16'),
            ('stimList', False, 'b1'),
            ('stimParams', cls.empty_stim_params(), cls.dtype_from_template(cls._STIM_PARAM_FIELDS))
        )

    def __init__(self, subject, montage, files):
        super(CatFRSessionLogParser, self).__init__(subject, montage, files)
        if 'no_accent_wordpool' in files:
            wordpool_type = 'no_accent_wordpool'
        else:
            wordpool_type = 'wordpool'
        self._wordpool = np.array([x.strip() for x in open(files[wordpool_type]).readlines()])
        self._session = -999
        self._list = -999
        self._serialpos = -999
        self._stimList = False
        self._word = ''
        self._version = -1
        self._add_fields(*self._catfr_fields())
        self._add_type_to_new_event(
            INSTRUCT_VIDEO=self.event_instruct_video,
            SESS_START=self.event_sess_start,
            MIC_TEST=self.event_default,
            PRACTICE_TRIAL=self.event_practice_trial,
            COUNTDOWN_START=self.event_default,
            COUNTDOWN_END=self.event_default,
            PRACTICE_ORIENT=self.event_default,
            PRACTICE_ORIENT_OFF=self.event_default,
            PRACTICE_WORD=self.event_practice_word,
            PRACTICE_WORD_OFF=self.event_practice_word_off,
            DISTRACT_START=self.event_default,
            DISTRACT_END=self.event_default,
            RETRIEVAL_ORIENT=self.event_default,
            PRACTICE_REC_START=self.event_default,
            PRACTICE_REC_END=self.event_default,
            PRACTICE_DISTRACT_START=self.event_default,
            PRACTICE_DISTRACT_END=self.event_default,
            TRIAL=self.event_trial,
            ORIENT=self.event_default,
            ORIENT_OFF=self.event_default,
            WORD=self.event_word,
            WORD_OFF=self.event_word_off,
            REC_START=self.event_default,
            REC_END=self.event_default,
            SESS_END=self.event_default,
            SESSION_SKIPPED=self.event_default,
            STIM_PARAMS=self._event_skip,
            STIM_ON = self.event_stim_on,
            ENCODING_END=self.event_default
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
        event.list = self._list
        event.session = self._session
        event.stimList = self._stimList
        event.expVersion = self._version
        return event

    def event_stim_on(self, split_line):
        self._stimList = True
        event = self.event_default(split_line)
        return event

    def event_instruct_video(self, split_line):
        event = self.event_default(split_line)
        if split_line[3] == 'ON':
            event.type = 'INSTRUCT_START'
        else:
            event.type = 'INSTRUCT_END'
        return event

    def event_sess_start(self, split_line):
        self._session = int(split_line[3]) - 1
        self._version = split_line[5].split('_')[1]
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
        self._word = self._word.upper()
        if (self._wordpool == self._word).any():
            wordno = np.where(self._wordpool == self._word)
            event.wordno = wordno[0] + 1
        else:
            event.wordno = -1
        return event

    def event_practice_word(self, split_line):
        self._word = strip_accents(split_line[3])
        event = self.event_default(split_line)
        event.serialpos = self._serialpos
        event.word = self._word

        return event

    def event_practice_word_off(self, split_line):
        event = self.event_default(split_line)
        event.word = self._word
        return event

    def event_trial(self, split_line):
        if float(self._version) < 2:
            self._list = int(split_line[3])
            self._stimList = int(split_line[5]) > 0
        else:
            self._list = int(split_line[3])
            self._stimList = not split_line[4] == 'NONSTIM'
        return self.event_default(split_line)

    def event_word(self, split_line):
        self._word = strip_accents(split_line[4])
        self._serialpos = int(split_line[5]) + 1
        event = self.event_default(split_line)
        event = self.apply_word(event)
        event.serialpos = self._serialpos
        event.isStim = split_line[6] == 'STIM'
        event.categoryNum = split_line[7]
        event.category = split_line[8]

        return event

    def event_word_off(self, split_line):
        event = self.event_default(split_line)
        event = self.apply_word(event)
        event.serialpos = self._serialpos
        return event

    def modify_recalls(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        ann_outputs = self._parse_ann_file(str(self._list - 1) if self._list > 0 else 'p')
        for recall in ann_outputs:
            word = recall[-1]

            new_event = self._empty_event
            new_event.list = self._list
            new_event.session = self._session
            new_event.stimList = self._stimList
            new_event.expVersion = self._version
            new_event.rectime = float(recall[0])
            new_event.mstime = rec_start_time + recall[0]
            new_event.msoffset = 20
            new_event.word = word
            new_event.wordno = recall[1]

            # If vocalization
            if word == '<>' or word == 'V' or word == '!':
                new_event.type = 'REC_WORD_VV'
            else:
                new_event.type = 'REC_WORD'

            # If XLI

            pres_mask = self.find_presentation(word, events)
            pres_list = np.unique(events[pres_mask].list)

            # Correct recall or PLI
            if len(pres_list) == 1:
                new_event.intrusion = self._list - pres_list[0]
                if new_event.intrusion == 0:
                    new_event.categoryNum = np.unique(events[pres_mask].categoryNum)
                    new_event.category = np.unique(events[pres_mask].category)
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
        return np.logical_and(np.logical_or(events.word == word,
                                            events.word == word.lower()),
                              events.type == 'WORD')


def catfr_log_parser_wrapper(subject, session, experiment, base_dir='/data/eeg/', session_log_name='session.log',
                          wordpool_name='CatFR_WORDS.txt'):

    exp_path = os.path.join(base_dir, subject, 'behavioral', experiment)
    session_log_path = os.path.join(exp_path, 'session_%d' % session, session_log_name)
    wordpool_path = os.path.join(exp_path, wordpool_name)
    parser = CatFRSessionLogParser(session_log_path, wordpool_path, subject)
    return parser


def parse_fr_session_log(subject, session, experiment, base_dir='/data/eeg/', session_log_name='session.log',
                         wordpool_name='RAM_wordpool_noAcc.txt'):
    return catfr_log_parser_wrapper(subject, session, experiment, base_dir='/data/eeg/', session_log_name='session.log',
                                 wordpool_name='CatFR_WORDS.txt').parse()
