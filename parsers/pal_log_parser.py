from base_log_parser import BaseSessionLogParser, UnknownExperimentTypeException
from system2_log_parser import System2LogParser
import numpy as np
import os


class PALSessionLogParser(BaseSessionLogParser):

    _STIM_PARAM_FIELDS = System2LogParser.sys2_fields()

    @classmethod
    def empty_stim_params(cls):
        """
        Makes a recarray for empty stim params (no stimulation)
        :return:
        """
        return cls.event_from_template(cls._STIM_PARAM_FIELDS)

    @classmethod
    def _pal_fields(cls):
        """
        Returns the template for a new FR field
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
            ('resp_pass', -999, 'int16'),
            ('vocalization', -999, 'int16'),
            ('RT', -999, 'int16'),
            ('expVersion', '', 'S16'),
            ('stimType', '', 'S16'),
            ('stimTrial', 0, 'b1'),
            ('isStim', -999, 'int16'),
            ('stimParams', cls.empty_stim_params(), cls.dtype_from_template(cls._STIM_PARAM_FIELDS))
        )
        #return MatParser(MATFile)
    def __init__(self, session_log, subject, ann_dir=None):
        BaseSessionLogParser.__init__(self, session_log, subject, ann_dir)
        #self.fields = {field[0]: field[1] for field in fields)
        self._session = -999
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
        self._stimTrial = 0

        self._stimType = ''
        self._add_fields(*self._pal_fields())
        self._add_type_to_new_event(
            SESS_START=self.event_sess_start,
            COUNTDOWN_START=self._event_skip,
            COUNTDOWN_END=self._event_skip,
            STUDY_START=self.event_study_start,
            STUDY_ORIENT=self.event_study_orient,
            STUDY_PAIR=self.event_study_pair,
            MATH_START=self.event_math_start,
            MATH_END=self.event_default,
            TEST_START=self.event_test_start,
            TEST_ORIENT=self.event_test_orient,
            TEST_PROBE=self.event_test_probe,
            REC_START=self.event_default,
            REC_END=self.event_default,
            SESS_END=self.event_default,
            STIM_ON=self.event_stim_on,
            PRACTICE_TRIAL=self.event_default,
            STIM_PARAMS= self._event_skip
        )
        self._add_type_to_modify_events(
            SESS_START=self.modify_session,
            REC_START=self.modify_recalls,
            TEST_PROBE=self.modify_test,
            STUDY_PAIR=self.modify_study
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
        event.stimType = self._stimType
        event.stimTrial = self._stimTrial
        return event

    def event_stim_on(self, split_line):
        event = self.event_default(split_line)
        self._stimTrial = 1
        event.stimTrial = self._stimTrial
        event.serialpos = self._serialpos
        event.probepos = -999
        event.study_1 = ''
        event.study_2 = ''
        return event

    def event_sess_start(self, split_line):
        self._session = int(split_line[3]) - 1
        self._version = split_line[5]
        return self.event_default(split_line)


    def event_practice_trial(self, split_line):
        self._list = -1
        self._serialpos = -1
        event = self.event_default(split_line)
        return event

    def event_study_start(self, split_line):
        self._serialpos = 0
        event = self.event_default(split_line)
        trial_str = split_line[3]
        self._list = int(trial_str.split('_', 1)[1])+1
        event.list = self._list
        stimstr = split_line[5]
        if stimstr == 'none':
            self._stimType = 'NONE'
            self._stimTrial = 0
        elif stimstr == 'encoding':
           self._stimType = 'ENCODING'
           self._stimTrial =1
        elif stimstr == 'retrieval':
            self._stimType = 'RETRIEVAL'
            self._stimTrial = 1
        event.stimType = self._stimType
        event.stimTrial = self._stimTrial
        return event

    def event_study_orient(self, split_line):
        event = self.event_default(split_line)
        self._serialpos += 1
        event.serialpos = self._serialpos
        trial_str = split_line[4]
        self._list = int(trial_str.split('_', 1)[1])+1
        event.list = self._list

        stimstr = split_line[5]
        if stimstr == "NO_STIM":
            self._isStim = 0
        elif stimstr =="STIM":
            self._isStim = 1
        event.isStim = self._isStim
        event.stimType = self._stimType
        event.stimTrial = self._stimTrial
        event.correct = 0
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
        if stimstr == "NO_STIM":
            self._isStim = 0
        elif stimstr =="STIM":
            self._isStim = 1
        event.isStim = self._isStim
        event.stimType = self._stimType
        event.stimTrial = self._stimTrial
        event.correct = 0
        return event

    def event_test_start(self, split_line):
        event = self.event_default(split_line)

        trial_str = split_line[3]
        self._list = int(trial_str.split('_', 1)[1])+1
        event.list = self._list
        self._probepos = 0
        return event

    def event_test_orient(self, split_line):
        event = self.event_default(split_line)

        stimstr = split_line[6]
        event.list = self._list
        if stimstr == "NO_STIM":
            self._isStim = 0
        elif stimstr =="STIM":
            self._isStim = 1
        event.isStim = self._isStim
        event.stimType = self._stimType
        event.stimTrial = self._stimTrial

        self._probepos +=1
        event.probepos = self._probepos

        serialstr = split_line[4]
        self._serialpos = int(serialstr.split('_', 1)[1])+1
        event.serialpos = self._serialpos
        event.correct = 0
        return event

    def event_test_probe(self, split_line):
        event = self.event_default(split_line)
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

        event.isStim = self._isStim
        event.stimType = self._stimType
        event.stimTrial = self._stimTrial

        event.correct = 0

        return event

    def event_math_start(self, split_line):
        event = self.event_default(split_line)
        self._serialpos=-999
        return event

    def modify_test(self, events):
        orient_event_mask = events.type == 'TEST_ORIENT'
        orient_event_value = [i for i, x in enumerate(orient_event_mask) if x][-1]
        last_orient = events[orient_event_value]
        last_orient.probe_word = self._probe_word
        last_orient.expecting_word = self._expecting_word
        last_orient.cue_direction = self._cue_direction
        return events

    def modify_study(self, events):
        orient_event_mask = events.type == 'STUDY_ORIENT'
        orient_event_value = [i for i, x in enumerate(orient_event_mask) if x][-1]
        last_orient = events[orient_event_value]
        last_orient.study_1 = self._study_1
        last_orient.study_2 = self._study_2
        return events

    def modify_session(self, events):
        """
        applies session and expVersion to all previous events
        :param events: all events up until this point in the log file
        """
        events.session = self._session
        events.expVersion = self._version
        return events

    def modify_recalls(self, events):
        rec_start_event = events[-1]
        rec_start_time = float(rec_start_event.mstime)
        ann_outputs = self._parse_ann_file(str(self._list - 1) + '_' + str(self._probepos-1))
        for recall in ann_outputs:
            word = recall[-1]
            new_event = self._empty_event
            new_event.list = self._list
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
            new_event.session = self._session
            new_event.stimType = self._stimType
            new_event.stimTrial = self._stimTrial
            new_event.expVersion = self._version
            new_event.RT = recall[0]
            new_event.mstime = rec_start_time + recall[0]
            new_event.msoffset = 20
            self._correct = 1 if word == self._expecting_word and word != self._probe_word else 0
            new_event.correct = self._correct
            new_event.resp_pass = 0
            new_event.intrusion = 0
            new_event.vocalization = 0

            study_mask = self.find_presentation(self._study_1, events)
            test_mask = self.find_test(self._probe_word, events)
            pres_mask = self.find_presentation(word, events)
            pres_list = np.unique(events[pres_mask].list)

            if self._correct == 1:
                events.correct[study_mask] = self._correct
                events.correct[test_mask] = self._correct

            events.resp_pass[study_mask] = 0
            events.resp_pass[test_mask] = 0

            if word != '<>':
               events.resp_word[study_mask] = word
               #events.resp_word[test_mask] = word
            new_event.type = 'REC_EVENT'
            events.probepos[study_mask] = self._probepos
            for event in events[study_mask]:
                if event.list == self._list:
                    if word != '<>':
                        events.RT[test_mask] = new_event.RT
                        events.RT[study_mask] = new_event.RT
                        events.resp_word[test_mask] = word
            if word == 'PASS':
                new_event.resp_pass = 1
                events.resp_pass[study_mask] = 1  # move this later
                events.resp_pass[test_mask] = 1  # move this later
            # if xli

            if self._correct == 0:
                if word == '<>' or word == 'v' or word == '!':
                    new_event.vocalization = 1
                elif word == 'PASS':
                    new_event.resp_pass = 1
                    events.resp_pass[study_mask] = 1  # move this later
                    events.resp_pass[test_mask] = 1  # move this later
                elif recall[1] == -1:
                    new_event.intrusion = -1
                else:  # correct recall or pli or xli from latter list
                    # correct recall or pli
                    if len(pres_list) == 1:
                        new_event.intrusion = self._list - pres_list[0]
                    else:  # xli
                        new_event.intrusion = -1
            events = np.append(events, new_event).view(np.recarray)

        return events

    @staticmethod
    def find_presentation(word, events):
        events = events.view(np.recarray)
        #not quite sure why I need the last clause, but it doesn't work without it. Maybe the type changes somewhere
        # within the loop somehow?
        return np.logical_and(np.logical_and(events.study_1 == word,
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


def pal_log_parser_wrapper(subject, session, experiment, base_dir='/data/eeg/', session_log_name='session.log'):

    if experiment not in ('PAL1', 'PAL2', 'PAL3'):
        raise UnknownExperimentTypeException('experiment must be one of pal1, pal2, or pal3')

    exp_path = os.path.join(base_dir, subject, 'behavioral', experiment)
    session_log_path = os.path.join(exp_path, 'session_%d' % session, session_log_name)
    parser = PALSessionLogParser(session_log_path, subject)
    return parser


def parse_pal_session_log(subject, session, experiment, base_dir='/data/eeg/', session_log_name='session.log',
                         wordpool_name='ram_wordpool_noacc.txt'):
    return pal_log_parser_wrapper(subject, session, experiment, base_dir='/data/eeg/', session_log_name='session.log')
