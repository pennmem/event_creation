import re
import numpy as np

from .base_log_parser import BaseSessionLogParser
from ..viewers.recarray import strip_accents

class CatFRSessionLogParser(BaseSessionLogParser):
    """
    Handles parsing of catFR log files and production of catFR events
    """

    @classmethod
    def _catfr_fields(cls):
        """
        Returns the template for a new FR field
        :return:
        """
        return (
            ('list', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('item_name', 'X', 'S16'),
            ('item_num', -999, 'int16'),
            ('recalled', False, 'int16'),
            ('rectime', -999, 'int32'),
            ('exp_version', -1, 'S16'),
            ('intrusion', -999, 'int16'),
            ('is_stim', False, 'b1'),
            ('category', 'X', 'S16'),
            ('category_num', -999, 'int16'),
            ('stim_list', False, 'b1'),
        )

    # Params specific to FR2
    CATFR2_STIM_DURATION= 4600
    CATFR2_STIM_PULSE_FREQUENCY = 50
    CATFR2_STIM_N_PULSES = 250
    CATFR2_STIM_BURST_FREQUENCY = 1
    CATFR2_STIM_N_BURSTS = 1
    CATFR2_STIM_PULSE_WIDTH = 300

    def __init__(self, protocol, subject, montage, experiment, session, files):
        """
        constructor
        :param protocol: Protocol for this subject/session
        :param subject: Subject that ran in this session (no montage code)
        :param montage: Montage for this subject/session
        :param experiment: Experiment name and number for this session (e.g. FR3, catFR1)
        :param session: Session for this subject/session
        :param files: A dictionary of files (as returned from an instance of Transferer). Must include 'session_log'
                      and a list of files under the key 'annotations' pointing to .ann files
        :return:
        """

        super(CatFRSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
                                                    include_stim_params=True)
        if 'no_accent_wordpool' in files:
            wordpool_type = 'no_accent_wordpool'
        else:
            wordpool_type = 'wordpool'
        self._wordpool = np.array([x.strip() for x in open(files[wordpool_type]).readlines()])
        self._list = -999
        self._serialpos = -999
        self._stim_list = False
        self._word = ''
        self._version = -1
        self._catfr2_stim_params = {
            'pulse_freq': self.CATFR2_STIM_PULSE_FREQUENCY,
            'n_pulses': self.CATFR2_STIM_N_PULSES,
            'burst_freq': self.CATFR2_STIM_BURST_FREQUENCY,
            'n_bursts': self.CATFR2_STIM_N_BURSTS,
            'pulse_width': self.CATFR2_STIM_PULSE_WIDTH,
            'stim_duration': self.CATFR2_STIM_DURATION
        }
        self._is_fr2 = False
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
            STIM_PARAMS=self._set_stim_params,
            STIM_ON = self.event_stim_on,
            ENCODING_END=self.event_default
        )
        self._add_type_to_modify_events(
            SESS_START=self.modify_session,
            REC_START=self.modify_recalls
        )

    @staticmethod
    def persist_fields_during_stim(event):
        if event['type'] == 'WORD':
            return ('list', 'serialpos', 'item_name', 'item_num', 'recalled',
                    'intrusion', 'stim_list', 'subject', 'session', 'eegfile',
                    'rectime')
        else:
            return ('list', 'serialpos', 'stim_list', 'subject', 'session', 'eegfile', 'rectime')

    def _set_stim_params(self, split_line):
        """
        For catFR2 specifically, used to set the catFR2 stim parameters
        :param split_line: the line defining stim params, split on tabs
        :return:
        """
        self._is_fr2 = True
        if split_line[9] != '0':
            # Have to do this annoying thing because there were a few sessions where patients from these sites
            # had one less in their anode and cathode numbers
            if split_line[5].isdigit():
                if self._subject[-1] in ('M', 'W', 'E'):
                    offset = 1
                else:
                    offset = 0
                self._catfr2_stim_params['anode_number'] = int(split_line[5]) + offset
                self._catfr2_stim_params['cathode_number']= int(split_line[7]) + offset
            else:
                self._catfr2_stim_params['anode_label']= split_line[5]
                self._catfr2_stim_params['cathode_label'] = split_line[7]

        self._catfr2_stim_params['amplitude'] = float(split_line[9])
        if self._catfr2_stim_params < 5:
            self._catfr2_stim_params['amplitude'] *= 1000
        self._catfr2_stim_params['stim_on'] = True
        return False


    def event_default(self, split_line):
        """
        Override base class's default event to include list, serial position, and stimList
        :param split_line:
        :return:
        """
        event = BaseSessionLogParser.event_default(self, split_line)
        event.list = self._list
        event.stim_list = self._stim_list
        event.exp_version = self._version
        return event

    def event_stim_on(self, split_line):
        self._stim_list = True
        event = self.event_default(split_line)
        event.is_stim = True
        self.set_event_stim_params(event, jacksheet=self._jacksheet, **self._catfr2_stim_params)
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
        self._version = re.sub(r'[^\d.]', '',split_line[5])
        return self.event_default(split_line)

    def modify_session(self, events):
        """
        applies session and expVersion to all previous events
        :param events: all events up until this point in the log file
        """
        events.exp_version = self._version
        return events

    def event_practice_trial(self, split_line):
        self._list = -1
        self._serialpos = -1
        event = self.event_default(split_line)
        return event

    def apply_word(self, event):
        event.item_name = self._word.upper()
        self._word = self._word.upper()
        if (self._wordpool == self._word).any():
            wordno = np.where(self._wordpool == self._word)
            event.item_num = wordno[0] + 1
        else:
            event.item_num = -1
        return event

    def event_practice_word(self, split_line):
        self._word = strip_accents(split_line[3])
        event = self.event_default(split_line)
        event.serialpos = self._serialpos
        event.item_name = self._word

        return event

    def event_practice_word_off(self, split_line):
        event = self.event_default(split_line)
        event.item_name = self._word
        return event

    def event_trial(self, split_line):
        if float(self._version) < 2:
            self._list = int(split_line[3])
            self._stim_list = int(split_line[5]) > 0
        else:
            self._list = int(split_line[3])
            self._stim_list = not split_line[4] == 'NONSTIM'
        return self.event_default(split_line)

    def event_word(self, split_line):
        self._word = strip_accents(split_line[4])
        self._serialpos = int(split_line[5]) + 1
        event = self.event_default(split_line)
        event = self.apply_word(event)
        event.serialpos = self._serialpos
        event.is_stim = split_line[6] == 'STIM'
        if event.is_stim & self._is_fr2:
            self.set_event_stim_params(event, jacksheet=self._jacksheet, **self._catfr2_stim_params)
        event.category_num = split_line[7]
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
            new_event.stim_list = self._stim_list
            new_event.exp_version = self._version
            new_event.rectime = float(recall[0])
            new_event.mstime = rec_start_time + recall[0]
            new_event.msoffset = 20
            new_event.item_name = word
            new_event.item_num = recall[1]

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
                    new_event.category_num = np.unique(events[pres_mask].category_num)
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
        return np.logical_and(np.logical_or(events.item_name == word,
                                            events.item_name == word),
                              events.type == 'WORD')
