from .base_log_parser import BaseSessionLogParser
import numpy as np


class LTPFR2SessionLogParser(BaseSessionLogParser):

    @classmethod
    def _ltpfr2_fields(cls):
        """
        Returns the templates for all LTPFR-specific fields (those not included under _BASE_FIELDS)
        :return:
        """
        return (
            ('trial', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('begin_distractor', -999, 'int16'),
            ('final_distractor', -999, 'int16'),
            ('begin_math_correct', -999, 'int16'),
            ('final_math_correct', -999, 'int16'),
            ('item_name', '', 'S16'),
            ('item_num', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('intruded', 0, 'int16'),
            ('rectime', -999, 'int32'),
            ('intrusion', -999, 'int16'),

            ('badEpoch', False, 'b1'),
            ('artifactChannels', False, 'b1', 128),
            ('variance', np.nan, 'float', 128),
            ('medGradient', np.nan, 'float', 128),
            ('ampRange', np.nan, 'float', 128),
            ('iqrDevMax', np.nan, 'float', 128),
            ('iqrDevMin', np.nan, 'float', 128),
            ('eogArtifact', -1, 'int8', 128)
        )

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(LTPFR2SessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files)
        self._wordpool = np.array([x.strip() for x in open(files['wordpool']).readlines()])
        self._session = -999
        self._trial = -999
        self._serialpos = -999

        # Holds distractor ID and performance info so it can be entered into the 'WORD' event following the distractor
        self._distractor = 0
        self._math_correct = -999

        self._add_fields(*self._ltpfr2_fields())
        self._add_type_to_new_event(
            SESS_START=self.event_sess_start,
            FR_PRES=self.event_fr_pres,
            REC_START=self.event_default,
            REST=self._event_skip,
            REST_REWET=self.event_default,
            SESS_END=self._event_skip,
            SESSION_SKIPPED=self._event_skip,
            DISTRACTOR_MATH=self.event_distractor,
            MATH_TOTAL_SCORE=self._event_skip
        )
        self._add_type_to_modify_events(
            SESS_START=self.modify_session,
            REC_START=self.modify_recalls
        )

    """=====EVENT CREATION====="""
    def event_default(self, split_line):
        """
        Override base class's default event to include list, serial position, and stimList
        :param split_line:
        :return:
        """
        event = BaseSessionLogParser.event_default(self, split_line)
        event.session = self._session
        event.trial = self._trial
        return event

    def event_sess_start(self, split_line):
        """
        Extracts the session number from the fourth column of SESS_START log events
        :param split_line:
        :return:
        """
        self._session = int(split_line[3]) - 1
        return False

    def event_fr_pres(self, split_line):
        self.update_trial(split_line)
        self._serialpos += 1
        event = self.event_default(split_line)
        event.type = 'WORD'
        event.serialpos = self._serialpos
        event.begin_distractor = self._distractor
        event.begin_math_correct = self._math_correct
        event.item_name = split_line[4]
        event.item_num = int(split_line[5])
        return event

    def event_distractor(self, split_line):
        """
        Collect math distractor information and create a DISTRACTOR event. NOTE: The original .mat files always have the
        distractor event's distractor and math_correct values logged as "begin_", even if it is a final distractor event
        """
        self.update_trial(split_line)
        self._distractor = int(split_line[6])
        self._math_correct = int(split_line[7])
        event = self.event_default(split_line)
        event.begin_distractor = self._distractor
        event.begin_math_correct = self._math_correct
        event.type = 'DISTRACTOR'
        return event

    """=====EVENT MODIFIERS====="""
    def modify_session(self, events):
        """
        Applies session number to all previous events.
        """
        events.session = self._session
        return events

    def modify_recalls(self, events):
        if self._distractor != 0:
            events = self.modify_final_distractor_info(self._trial, events).view(np.recarray)

        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        # Get list of recalls from the .ann file for the current list; each recall is (rectime, item_num, item_name)
        ann_outputs = self._parse_ann_file(str(self._trial - 1))

        for recall in ann_outputs:
            word = recall[-1]
            # Vocalizations are skipped in the .mat files for ltpFR2
            if word == '<>' or word == 'V' or word == '!':
                continue
            new_event = self._empty_event
            new_event.trial = self._trial
            new_event.session = self._session
            new_event.rectime = int(round(float(recall[0])))
            new_event.mstime = rec_start_time + new_event.rectime
            new_event.msoffset = 20
            new_event.item_name = word
            new_event.item_num = int(recall[1])
            new_event.type = 'REC_WORD'

            # If XLI
            if recall[1] == -1:
                new_event.intrusion = -1
            else:  # Correct recall or PLI or XLI from later list
                # Determines which list the recalled word was from (gives [] if from a future list)
                pres_mask = self.find_presentation(new_event.item_num, events)
                pres_trial = np.unique(events[pres_mask].trial)

                # Correct recall or PLI
                if len(pres_trial) == 1:
                    # Determines how many lists back the recalled word was presented
                    new_event.intrusion = self._trial - pres_trial[0]
                    # Retrieve the recalled word's serial position in its list, as well as the distractor(s) used
                    new_event.serialpos = np.unique(events[pres_mask].serialpos)
                    new_event.begin_distractor = np.unique(events[pres_mask].begin_distractor)
                    new_event.final_distractor = np.unique(events[pres_mask].final_distractor)
                    # Correct recall if word is from the most recent list
                    if new_event.intrusion == 0:
                        # Retroactively log on the word pres event that the word was recalled
                        if not any(events[pres_mask].recalled):
                            events.recalled[pres_mask] = True
                    else:
                        events.intruded[pres_mask] = new_event.intrusion
                else:  # XLI from later list
                    new_event.intrusion = -1

            # Add recall event to events array
            events = np.append(events, new_event).view(np.recarray)

        return events

    def modify_final_distractor_info(self, trial, events):
        """
        Retroactively adds final distractor information to the word presentations from the most recent list
        :param trial: The current list number
        :param events: The list of all prior events
        :return: The modified event list
        """
        events = events.view(np.recarray)
        this_trial = np.logical_and(events.trial == trial, events.type == 'WORD')
        events.final_distractor[this_trial] = self._distractor
        events.final_math_correct[this_trial] = self._math_correct
        self._distractor = 0
        self._math_correct = -999
        return events

    def update_trial(self, split_line):
        """
        Checks whether a new word list has been reached, and resets the serial position if so
        """
        current_trial = int(split_line[3]) + 1
        if self._trial != current_trial:
            self._serialpos = 0
            self._trial = current_trial

    @staticmethod
    def find_presentation(item_num, events):
        events = events.view(np.recarray)
        return np.logical_and(events.item_num == item_num, events.type == 'WORD')
