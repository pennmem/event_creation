from base_log_parser import BaseSessionLogParser, UnknownExperimentTypeException
from system2_log_parser import System2LogParser
from viewers.view_recarray import strip_accents
import numpy as np
import os
import re


class FRSessionLogParser(BaseSessionLogParser):

    @classmethod
    def _fr_fields(cls):
        """
        :return: the template for a new FR field
        """
        return (
            ('list', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('word', 'X', 'S64'),
            ('wordno', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('rectime', -999, 'int16'),
            ('intrusion', -999, 'int16'),
            ('exp_version', '', 'S64'),
            ('stim_list', False, 'b1'),
            ('is_stim', False, 'b1'),
        )

    # FR2 specific fields
    FR2_STIM_DURATION= 4600
    FR2_STIM_PULSE_FREQUENCY = 50
    FR2_STIM_N_PULSES = 230
    FR2_STIM_BURST_FREQUENCY = 1
    FR2_STIM_N_BURSTS = 1
    FR2_STIM_PULSE_WIDTH = 300

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(FRSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
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
        self._version = ''
        self._is_fr2 = False
        self._fr2_stim_params = {
            'pulse_freq': self.FR2_STIM_PULSE_FREQUENCY,
            'n_pulses': self.FR2_STIM_N_PULSES,
            'burst_freq': self.FR2_STIM_BURST_FREQUENCY,
            'n_bursts': self.FR2_STIM_N_BURSTS,
            'pulse_width': self.FR2_STIM_PULSE_WIDTH,
            'stim_duration': self.FR2_STIM_DURATION
        }
        self._fr2_stim_on_time = False
        self._add_fields(*self._fr_fields())
        self._add_type_to_new_event(
            INSTRUCT_VIDEO=self.event_instruct_video,
            SESS_START=self.event_sess_start,
            MIC_TEST=self.event_default,
            PRACTICE_TRIAL=self.event_practice_trial,
            COUNTDOWN_START=self.event_default,
            COUNTDOWN_END=self.event_reset_serialpos,
            PRACTICE_ORIENT=self.event_default,
            PRACTICE_ORIENT_OFF=self.event_default,
            PRACTICE_WORD=self.event_practice_word,
            PRACTICE_WORD_OFF=self.event_practice_word_off,
            PRACTICE_DISTRACT_START=self.event_default,
            PRACTICE_DISTRACT_END=self.event_default,
            DISTRACT_START=self.event_default,
            DISTRACT_END=self.event_default,
            RETRIEVAL_ORIENT=self.event_default,
            PRACTICE_REC_START=self.event_default,
            PRACTICE_REC_END=self.event_default,
            TRIAL=self.event_trial,
            ORIENT=self.event_default,
            ORIENT_OFF=self.event_default,
            WORD=self.event_word,
            WORD_OFF=self.event_word_off,
            REC_START=self.event_default,
            REC_END=self.event_default,
            SESS_END=self.event_default,
            SESSION_SKIPPED=self.event_default,
            STIM_PARAMS=self.stim_params_event,
            STIM_ON=self.stim_on_event,
        )
        self._add_type_to_modify_events(
            SESS_START=self.modify_session,
            REC_START=self.modify_recalls,
        )

    @staticmethod
    def persist_fields_during_stim(event):
        if event['type'] == 'WORD':
            return ('list', 'serialpos', 'word', 'wordno', 'recalled',
                    'intrusion', 'stim_list', 'subject', 'session', 'eegfile',
                    'rectime')
        else:
            return ('list', 'serialpos', 'stim_list', 'subject', 'session', 'eegfile', 'rectime')

    def event_default(self, split_line):
        """
        Override base class's default event to include list, serial position, and stimList
        :param split_line:
        :return:
        """
        event = BaseSessionLogParser.event_default(self, split_line)
        if self._is_fr2 and self._fr2_stim_on_time and self._fr2_stim_on_time + self.FR2_STIM_DURATION >= int(split_line[0]):
            event.is_stim = True
            self.set_event_stim_params(event, jacksheet=self._jacksheet, **self._fr2_stim_params)

        event.list = self._list
        event.stim_list = self._stim_list
        event.exp_version = self._version
        return event

    def event_reset_serialpos(self, split_line):
        event = self.event_default(split_line)
        self._serialpos = -1
        return event

    def event_instruct_video(self, split_line):
        event = self.event_default(split_line)
        if split_line[3] == 'ON':
            event.type = 'INSTRUCT_START'
        else:
            event.type = 'INSTRUCT_END'
        return event

    def event_sess_start(self, split_line):
        self._version = re.sub(r'[^\d.]', '', split_line[5])
        return self.event_default(split_line)

    def stim_on_event(self, split_line):
        self._fr2_stim_on_time = int(split_line[0])
        event = self.event_default(split_line)
        return event

    def stim_params_event(self, split_line):
        self._is_fr2 = True
        if split_line[9] != '0':
            if split_line[5].isdigit():
                if self._subject[-1] in ('M', 'W', 'E'):
                    offset = 1
                else:
                    offset = 0
                self._fr2_stim_params['anode_number'] = int(split_line[5]) + offset
                self._fr2_stim_params['cathode_number']= int(split_line[7]) + offset
            else:
                self._fr2_stim_params['anode_label']= split_line[5]
                self._fr2_stim_params['cathode_label'] = split_line[7]

        self._fr2_stim_params['amplitude'] = float(split_line[9])
        if self._fr2_stim_params < 5:
            self._fr2_stim_params *= 1000
        self._fr2_stim_params['stim_on'] = True
        return False

    def modify_session(self, events):
        """
        applies session and expVersion to all previous events
        :param events: all events up until this point in the log file
        """
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
        self._list = -1 # Have to set here as well, because sys1 events have no PRACTICE_WORD
        self._serialpos += 1
        event = self.event_default(split_line)
        self._word = strip_accents(split_line[3])
        event.serialpos = self._serialpos
        event = self.apply_word(event)
        return event

    def event_practice_word_off(self, split_line):
        event = self.event_default(split_line)
        event = self.apply_word(event)
        return event

    def event_trial(self, split_line):
        self._list = int(split_line[3])
        if split_line[4] == 'STIM_PARAMS':
            self._stim_list = split_line[5] == '1'
        else:
            self._stim_list = split_line[4] == 'STIM'
        return self.event_default(split_line)

    def event_word(self, split_line):
        self._word = strip_accents(split_line[4])
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
        ann_outputs = self._parse_ann_file(str(self._list - 1) if self._list > 0 else 'p')
        for recall in ann_outputs:
            word = recall[-1]

            new_event = self._empty_event
            new_event.list = self._list
            new_event.stim_list = self._stim_list
            new_event.exp_version = self._version
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

    @staticmethod
    def find_presentation(word, events):
        events = events.view(np.recarray)
        return np.logical_and(events.word == word, np.logical_or(events.type == 'WORD', events.type == 'PRACTICE_WORD'))

