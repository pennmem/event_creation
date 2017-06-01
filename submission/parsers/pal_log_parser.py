from .base_log_parser import BaseSessionLogParser, UnknownExperimentError
from .system2_log_parser import System2LogParser
import numpy as np
import os
import warnings
import re

class PALSessionLogParser(BaseSessionLogParser):

    @classmethod
    def _pal_fields(cls):
        """
        Returns the template for a new PAL field
        Has to be a method because of call to empty_stim_params, unfortunately
        :return:
        """
        return (
            ('list', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('probepos', -999, 'int16'),
            ('study_1', '', 'S16'),
            ('study_2', '', 'S16'),
            ('cue_direction', -999, 'int16'),
            ('probe_word', '', 'S16'),
            ('expecting_word', '', 'S16'),
            ('resp_word', '', 'S16'),
            ('correct', -999, 'int16'),
            ('intrusion', -999, 'int16'),
            ('resp_pass', 0 , 'int16'),
            ('vocalization', -999, 'int16'),
            ('RT', -999, 'int16'),
            ('exp_version', '', 'S16'),
            ('stim_type', '', 'S16'),
            ('stim_list', 0, 'b1'),
            ('is_stim', False, 'b1'),
        )


    PAL2_STIM_DURATION= 4600
    PAL2_STIM_PULSE_FREQUENCY = 50
    PAL2_STIM_N_PULSES = 250
    PAL2_STIM_BURST_FREQUENCY = 1
    PAL2_STIM_N_BURSTS = 1
    PAL2_STIM_PULSE_WIDTH = 300

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(PALSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
                                                  include_stim_params=True)
        self._list = -999
        self._serialpos = -999
        self._stimList = False
        self._study_1 = ''
        self._study_2 = ''
        self._version = ''
        self._probe_word = ''
        self._expecting_word = ''
        self._cue_direction = -999
        self._correct = -999
        self._stim_list = 0

        self._pal2_stim_params = {
            'pulse_freq': self.PAL2_STIM_PULSE_FREQUENCY,
            'n_pulses': self.PAL2_STIM_N_PULSES,
            'burst_freq': self.PAL2_STIM_BURST_FREQUENCY,
            'n_bursts': self.PAL2_STIM_N_BURSTS,
            'pulse_width': self.PAL2_STIM_PULSE_WIDTH,
            'stim_duration': self.PAL2_STIM_DURATION
        }

        self._pal2_stim_on_time = 0
        self._pal2_stim_serialpos = -1
        self._pal2_stim_list = -1
        self._pal2_stim_is_retrieval = False

        self._stim_type = ''
        self._add_fields(*self._pal_fields())
        self._add_type_to_new_event(
            SESS_START=self.event_sess_start,
            COUNTDOWN_START=self._event_skip,
            COUNTDOWN_END=self._event_skip,
            PRACTICE_ORIENT=self.event_practice_orient,
            PRACTICE_ORIENT_OFF=self.event_default,
            PRACTICE_RETRIEVAL_ORIENT=self.event_practice_retrieval_orient,
            PRACTICE_RETRIEVAL_ORIENT_OFF=self.event_default,
            PRACTICE_PAIR=self.event_practice_pair,
            PRACTICE_PAIR_OFF=self.event_practice_pair_off,
            STUDY_START=self.event_study_start,
            TRIAL=self.event_trial,
            ENCODING_START=self.event_encoding_start, # SYS2 EVENT
            ENCODING_END=self.event_encoding_end,
            STUDY_ORIENT=self.event_study_orient,
            STUDY_ORIENT_OFF=self.event_default,
            STUDY_PAIR=self.event_study_pair,
            PAIR_OFF=self.event_default,
            MATH_START=self.event_math_start,
            DISTRACT_START=self.event_distract_start,
            MATH_END=self.event_default,
            DISTRACT_END=self.event_distract_end,
            TEST_START=self.event_test_start,
            RECALL_START=self.event_recall_start,
            RECALL_END=self.event_recall_end,
            TEST_ORIENT=self.event_test_orient,
            RETRIEVAL_ORIENT=self.event_retrieval_orient,
            RETRIEVAL_ORIENT_OFF=self.event_default,
            TEST_PROBE=self.event_test_probe,
            PROBE_OFF=self.event_default,
            REC_START=self.event_default,
            REC_END=self.event_default,
            SESS_END=self.event_default,
            STIM_ON=self.event_stim_on,
            PRACTICE_TRIAL=self.event_practice_trial,
            INSTRUCT_VIDEO=self.event_instruct_video,
            MIC_TEST=self.event_default,
            STIM_PARAMS= self.event_stim_params,
            FEEDBACK_SHOW_ALL_PAIRS=self.event_default,
            FORCED_BREAK=self.event_default,
            SESSION_SKIPPED=self.event_default,
        )
        self._add_type_to_modify_events(
            SESS_START=self.modify_session,
            REC_START=self.modify_recalls,
            TEST_PROBE=self.modify_test,
            STUDY_PAIR=self.modify_study,
            PRACTICE_RETRIEVAL_ORIENT=self.modify_orient,
            RETRIEVAL_ORIENT=self.modify_orient,
            PRACTICE_PAIR=self.modify_practice_pair,
            STIM_ON=self.modify_stim_on
        )

    def event_default(self, split_line):
        """
        Override base class's default event to include list, serial position, and stimList
        :param split_line:
        :return:
        """
        event = BaseSessionLogParser.event_default(self, split_line)

        event.stimList = self._stimList
        event.exp_version = self._version
        event.stim_type = self._stim_type
        event.stim_list = self._stim_list
        event.list = self._list
        event.serialpos = self._serialpos

        self.check_apply_stim(event)

        return event

    def check_apply_stim(self, event):
        if self._pal2_stim_on_time and \
                self._pal2_stim_on_time + self.PAL2_STIM_DURATION >= event.mstime and \
                event.mstime >= self._pal2_stim_on_time:
            event.is_stim = True
            self.set_event_stim_params(event, jacksheet=self._jacksheet, stim_on=True, **self._pal2_stim_params)
        #elif self._pal2_stim_on_time and \
        #        event.list == self._pal2_stim_list and event.serialpos == self._pal2_stim_serialpos:
        #    item = event.type.item()
        #    if  ('TEST' in item or 'RETRIEVAL' in item or 'REC' in item) ^ (self._stim_type == 'ENCODING'):
        #        event.is_stim = True
        #        self.set_event_stim_params(event, jacksheet=self._jacksheet, stim_on=False, **self._pal2_stim_params)


    def event_stim_params(self, split_line):
        self._is_fr2 = True
        if split_line[9] != '0':
            if split_line[5].isdigit():
                if self._subject[-1] in ('M', 'W', 'E'):
                    offset = 1
                else:
                    offset = 0
                self._pal2_stim_params['anode_number'] = int(split_line[5]) + offset
                self._pal2_stim_params['cathode_number']= int(split_line[7]) + offset
            else:
                self._pal2_stim_params['anode_label']= split_line[5]
                self._pal2_stim_params['cathode_label'] = split_line[7]

        self._pal2_stim_params['amplitude'] = float(split_line[9])
        if self._pal2_stim_params['amplitude'] < 5:
            self._pal2_stim_params['amplitude'] *= 1000
        return False

    def event_instruct_video(self, split_line):
        event = self.event_default(split_line)
        event.type = 'INSTRUCT_VIDEO_' + split_line[3]
        return event

    def event_distract_start(self, split_line):
        self._serialpos = -999
        self._probepos = -999
        event = self.event_default(split_line)
        event.type = 'MATH_START'
        return event

    def event_distract_end(self, split_line):
        event = self.event_default(split_line)
        event.type = 'MATH_END'
        return event

    def modify_orient(self, events):
        retrieval_event = events[-1]
        mask = np.logical_or(events.study_1 == retrieval_event.probe_word, events.study_2 == retrieval_event.probe_word)
        events[mask].probepos = retrieval_event.probepos
        return events

    def event_practice_trial(self, split_line):
        self._list = -1
        event = self.event_default(split_line)
        self._serialpos = 0
        return event

    def event_stim_on(self, split_line):
        self._pal2_stim_on_time = int(split_line[0])
        event = self.event_default(split_line)
        self._stim_list = 1
        event.stim_list = self._stim_list
        event.serialpos = self._serialpos
        event.probepos = -999
        event.study_1 = ''
        event.study_2 = ''
        event.is_stim = True
        self.set_event_stim_params(event, jacksheet=self._jacksheet, **self._pal2_stim_params)
        return event

    def modify_stim_on(self, events):
        self._pal2_stim_serialpos = events[-2].serialpos
        self._pal2_stim_list = events[-2].list
        self._pal2_stim_is_retrieval = 'TEST' in events[-2].type
        return events

    def event_trial(self, split_line):
        self._list = int(split_line[3])
        event = self.event_default(split_line)
        self._stim_list = split_line[4] != 'NONSTIM'
        event.stim_list = self._stim_list
        return event

    def event_sess_start(self, split_line):
        self._version = float(re.sub(r'[^\d.]', '',split_line[5]))
        return self.event_default(split_line)

    def event_study_start(self, split_line):
        self._serialpos = 0
        event = self.event_default(split_line)
        event.type = 'ENCODING_START'
        trial_str = split_line[3]
        self._list = int(trial_str.split('_', 1)[1])+1
        event.list = self._list
        stimstr = split_line[5]
        if stimstr == 'none':
            self._stim_type = 'NONE'
            self._stim_list = 0
        elif stimstr == 'encoding':
           self._stim_type = 'ENCODING'
           self._stim_list =1
        elif stimstr == 'retrieval':
            self._stim_type = 'RETRIEVAL'
            self._stim_list = 1
        event.stim_type = self._stim_type
        event.stim_list = self._stim_list
        return event

    def event_encoding_start(self, split_line):
        self._serialpos = -999
        event = self.event_default(split_line)
        self._serialpos = 0
        return event

    def event_encoding_end(self, split_line):
        self._serialpos = -999
        event = self.event_default(split_line)
        event.list = self._list
        return event

    def event_practice_orient(self, split_line):
        self._serialpos += 1
        event = self.event_default(split_line)
        return event

    def event_study_orient(self, split_line):
        event = self.event_default(split_line)
        self._serialpos += 1
        event.serialpos = self._serialpos

        if self._version < 2:
            trial_str = split_line[4]
            self._list = int(trial_str.split('_', 1)[1])+1
            event.list = self._list

            stimstr = split_line[5]
            if stimstr == "NO_STIM":
                event.is_stim = 0
            elif stimstr =="STIM":
                event.is_stim = 1
                self.set_event_stim_params(event, self._jacksheet, **self._pal2_stim_params)
            event.stim_type = self._stim_type
            event.stim_list = self._stim_list
            event.correct = 0
        return event

    def event_practice_pair(self, split_line):

        self._serialpos = int(split_line[3]) + 1
        self._list = -1
        event = self.event_default(split_line)
        self._study_1 = split_line[4].split('_')[1]
        self._study_2 = split_line[5].split('_')[1]
        event.study_1 = self._study_1
        event.study_2 = self._study_2
        return event

    def modify_practice_pair(self, events):
        orient_on = np.where(events.type == 'PRACTICE_ORIENT')[0][-1]
        orient_off= np.where(events.type == 'PRACTICE_ORIENT_OFF')[0][-1]
        events[orient_on].study_1 = self._study_1
        events[orient_on].study_2 = self._study_2
        events[orient_off].study_1 = self._study_1
        events[orient_off].study_2 = self._study_2
        return events

    def event_practice_pair_off(self, split_line):
        event = self.event_default(split_line)
        event.study_1 = self._study_1
        event.study_2 = self._study_2
        event.list = self._list
        event.serialpos = self._serialpos
        return event

    def event_study_pair(self, split_line):
        event = self.event_default(split_line)

        event.serialpos = self._serialpos

        trial_str = split_line[4]
        self._list = int(trial_str.split('_', 1)[1])+1
        event.list = self._list

        word_1 = split_line[5]
        self._study_1 = word_1.split('_', 1)[1]
        event.study_1 = self._study_1

        word_2 = split_line[6]
        self._study_2 = word_2.split('_', 1)[1]
        event.study_2 = self._study_2

        stimstr = split_line[7]
        if self._version < 2:
            if stimstr == "NO_STIM":
                event.is_stim = 0
            elif stimstr =="STIM":
                event.is_stim = 1
                self.set_event_stim_params(event, self._jacksheet, **self._pal2_stim_params)
        event.stim_type = self._stim_type
        event.stim_list = self._stim_list
        event.correct = 0
        return event

    def event_recall_start(self, split_line):
        event = self.event_default(split_line)
        event.list = self._list
        self._probepos = 0
        return event

    def event_recall_end(self, split_line):
        self._serialpos = -999
        self._probepos = -999
        return self.event_default(split_line)

    def event_test_start(self, split_line):
        event = self.event_default(split_line)

        trial_str = split_line[3]
        self._list = int(trial_str.split('_', 1)[1])+1
        event.list = self._list
        self._probepos = 0
        return event

    def event_retrieval_orient(self, split_line):
        event = self.event_default(split_line)
        event.type = 'TEST_ORIENT'
        event.list = self._list
        self._probepos += 1
        event.probepos = self._probepos
        return event

    def event_practice_retrieval_orient(self, split_line):
        event = self.event_default(split_line)
        event.list = self._list
        self._probepos += 1
        event.probepos = self._probepos
        return event

    def event_retrieval_orient_off(self, split_line):
        event = self.event_default(split_line)
        event.type = 'TEST_ORIENT_OFF'
        return event

    def event_test_orient(self, split_line):
        event = self.event_default(split_line)

        stimstr = split_line[6]
        event.list = self._list
        if stimstr == "NO_STIM":
            event.is_stim = 0
        elif stimstr =="STIM":
            event.is_stim = 1
            self.set_event_stim_params(event, self._jacksheet, **self._pal2_stim_params)
        event.stim_type = self._stim_type
        event.stim_list = self._stim_list

        self._probepos +=1
        event.probepos = self._probepos

        serialstr = split_line[4]
        self._serialpos = int(serialstr.split('_', 1)[1])+1
        event.serialpos = self._serialpos
        event.correct = 0
        return event

    def event_test_probe(self, split_line):
        self._serialpos = int(split_line[4].split('_')[1]) + 1
        event = self.event_default(split_line)

        if self._list == -1:
            event.type = 'PRACTICE_PROBE'

        event.list = self._list
        probe_str = split_line[6]
        self._probe_word = probe_str.split('_', 1)[1]
        event.probe_word = self._probe_word

        expect_str = split_line[7]
        self._expecting_word = expect_str.split('_', 1)[1]
        event.expecting_word = self._expecting_word

        direction_str = split_line[8]
        self._cue_direction = int(direction_str.split('_', 1)[1])
        event.cue_direction = self._cue_direction
        event.probepos = self._probepos
        event.serialpos = self._serialpos

        event.study_1 = event.expecting_word if event.cue_direction == 1 else event.probe_word
        event.study_2 = event.expecting_word if event.cue_direction == 0 else event.probe_word

        event.stim_type = self._stim_type
        event.stim_list = self._stim_list

        event.correct = 0

        return event

    def event_math_start(self, split_line):
        self._serialpos=-999
        self._probepos = -999
        event = self.event_default(split_line)

        return event

    def modify_test(self, events):
        orient_ons = np.where(np.logical_or.reduce((events.type == 'STUDY_ORIENT',
                                                   events.type == 'TEST_ORIENT',
                                                   events.type == 'PRACTICE_RETRIEVAL_ORIENT')))[0]
        if len(orient_ons) > 0:
            orient_on = orient_ons[-1]
            events[orient_on].serialpos = self._serialpos

        try:
            orient_off = np.where(np.logical_or(events.type == 'TEST_ORIENT_OFF',
                                               events.type == 'PRACTICE_RETRIEVAL_ORIENT_OFF'))[0][-1]
            events[orient_off].serialpos = self._serialpos
        except:
            pass

        modify_mask = np.logical_and(events.list == events[-1].list, events.serialpos == self._serialpos)
        events.probepos[modify_mask] = events[-1].probepos
        events.probe_word[modify_mask] = events[-1].probe_word
        events.resp_word[modify_mask] = events[-1].resp_word
        events.cue_direction[modify_mask] = events[-1].cue_direction
        events.expecting_word[modify_mask] = events[-1].expecting_word
        events.study_1[modify_mask] = events[-1].study_1
        events.study_2[modify_mask] = events[-1].study_2

        return events

    def modify_study(self, events):
        orient_event_mask = events.type == 'STUDY_ORIENT'
        orient_event_value = np.where(orient_event_mask)[0][-1]
        last_orient = events[orient_event_value]
        last_orient.study_1 = self._study_1
        last_orient.study_2 = self._study_2
        return events

    def modify_session(self, events):
        """
        applies session and exp_version to all previous events
        :param events: all events up until this point in the log file
        """
        events.exp_version = self._version
        return events

    def modify_recalls(self, events):
        rec_start_event = events[-1]
        is_stim = rec_start_event.is_stim
        rec_start_time = float(rec_start_event.mstime)
        list_str = str(self._list-1) if self._list != -1 else 'p'
        ann_file = list_str + '_' + str(self._probepos-1)
        try:
            ann_outputs = self._parse_ann_file(ann_file)
        except IOError: # Will happen if no ann file
            warnings.warn("Ann file %s not parseable" % ann_file)
            return events
        response_spoken = False

        modify_events_mask = np.logical_and.reduce((events.serialpos == self._serialpos,
                                                    events.list == self._list,
                                                    events.type != 'REC_EVENT'))

        events.resp_pass[modify_events_mask] = 0
        events.correct[modify_events_mask] = 0

        for i, recall in enumerate(ann_outputs):
            word = recall[-1]
            new_event = self._empty_event
            new_event.list = self._list
            new_event.is_stim = is_stim
            new_event.serialpos = self._serialpos
            new_event.probepos = self._probepos
            new_event.probe_word = self._probe_word
            new_event.expecting_word = self._expecting_word
            new_event.cue_direction = self._cue_direction
            if self._cue_direction == 0:
                self._study_1 = self._probe_word
                self._study_2 = self._expecting_word
            elif self._cue_direction == 1:
                self._study_1 = self._expecting_word
                self._study_2 = self._probe_word
            new_event.study_1 = self._study_1
            new_event.study_2 = self._study_2
            new_event.resp_word = word
            new_event.stim_type = self._stim_type
            new_event.stim_list = self._stim_list
            new_event.exp_version = self._version
            new_event.RT = recall[0]
            new_event.mstime = rec_start_time + recall[0]
            new_event.msoffset = 20
            self._correct = 1 if word == self._expecting_word and word != self._probe_word else 0
            new_event.correct = self._correct
            new_event.resp_pass = 0
            new_event.intrusion = 0
            new_event.vocalization = 0

            modify_events_mask = np.logical_and.reduce((events.serialpos == self._serialpos,
                                                        events.list == self._list,
                                                        events.type != 'REC_EVENT'))

            pres_mask = self.find_presentation(word, events)
            pres_list = np.unique(events[pres_mask].list)

            same_word = i > 0 and ann_outputs[i-1][-1] == word

            if word != '<>' and word != 'v' and word != '!':
                is_vocalization = False
                events.vocalization[modify_events_mask] = 0
                events.resp_word[modify_events_mask] = word
                events.correct[modify_events_mask] = self._correct
                events.resp_pass[modify_events_mask] = 0
                if not same_word:
                    events.RT[modify_events_mask] = new_event.RT

                response_spoken = True
            else:
                if not response_spoken:
                    events.correct[modify_events_mask] = 0
                is_vocalization = True

            new_event.type = 'REC_EVENT'

            if word == 'PASS':
                new_event.resp_pass = 1
                events.resp_pass[modify_events_mask] = 1
            elif not is_vocalization:
                events.resp_pass[modify_events_mask] = 0
            # if xli

            if self._correct == 0:
                if word == '<>' or word == 'v' or word == '!':
                    new_event.vocalization = 1
                    new_event.intrusion = 0
                    new_event.resp_pass = 0
                    if not response_spoken or new_event.resp_pass:
                        events.resp_pass[modify_events_mask] = 0
                elif word == 'PASS':
                    new_event.resp_pass = 1
                    new_event.intrusion = 0
                    events.resp_pass[modify_events_mask] = 1
                    events.intrusion[modify_events_mask] = 0
                elif recall[1] == -1:
                    new_event.intrusion = -1
                    events.intrusion[modify_events_mask] = -1
                else:  # correct recall or pli or xli from latter list
                    # correct recall or pli
                    if len(pres_list) == 1:
                        new_event.intrusion = self._list - pres_list
                        events.intrusion[modify_events_mask] = self._list - pres_list
                    else:  # xli
                        new_event.intrusion = -1
                        events.intrusion[modify_events_mask] = -1

            if self._pal2_stim_serialpos == new_event.serialpos and \
                    self._pal2_stim_list == new_event.list and \
                    self._pal2_stim_is_retrieval:
                new_event.is_stim = True
                stim_on = new_event.mstime - self._pal2_stim_on_time < self.PAL2_STIM_DURATION
                self.set_event_stim_params(new_event, jacksheet=self._jacksheet, stim_on=stim_on, **self._pal2_stim_params)

            events = np.append(events, new_event).view(np.recarray)

        return events

    @staticmethod
    def find_presentation(word, events):
        events = events.view(np.recarray)
        # not quite sure why I need the last clause, but it doesn't work without it. Maybe the type changes somewhere
        # within the loop somehow?
        return np.logical_and(np.logical_and(np.logical_or(events.study_1 == word,
                                                           events.study_2 == word,),
                                             np.any([events.type == 'STUDY_ORIENT',
                                                     events.type == 'STUDY_PAIR'])),
                              events.type != 'REC_EVENT')

    @staticmethod
    def find_test(word, events):
        events = events.view(np.recarray)
        return np.logical_and(np.logical_and(events.probe_word == word,
                                             np.any([events.type == 'TEST_ORIENT',
                                                               events.type == 'TEST_PROBE'])),
                              events.type != 'REC_EVENT')

    @staticmethod
    def persist_fields_during_stim(event):
        if event['type'] == 'STUDY_PAIR':
            return ('resp_word', 'probe_word', 'probepos', 'cue_direction', 'subject', 'RT',
                    'stim_list', 'serialpos', 'resp_pass', 'correct', 'study_1', 'study_2',
                    'vocalization', 'stim_type', 'intrusion', 'list', 'expecting_word')
        else:
            return tuple()
