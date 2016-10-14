from base_log_parser import BaseSessionLogParser
import numpy as np
from ast import literal_eval
from loggers import logger


class LTPFRSessionLogParser(BaseSessionLogParser):

    @classmethod
    def _ltpfr_fields(cls):
        """
        Returns the templates for all LTPFR-specific fields (those not included under _BASE_FIELDS)
        :return:
        """
        return (
            ('trial', -999, 'int16'),
            ('studytrial', -999, 'int16'),
            ('listtype', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('distractor', -999, 'int16'),
            ('final_distractor', -999, 'int16'),
            ('math_correct', -999, 'int16'),
            ('final_math_correct', -999, 'int16'),
            ('task', -999, 'int16'),
            ('resp', -999, 'int16'),
            ('rt', -999, 'int16'),
            ('recog_resp', -999, 'int16'),
            ('recog_conf', -999, 'int16'),
            ('recog_rt', -999, 'int32'),
            ('word', '', 'S16'),  # Calling this 'item' will break things, due to the built-in recarray.item method
            ('wordno', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('intruded', 0, 'int16'),
            ('finalrecalled', False, 'b1'),
            ('recognized', False, 'b1'),
            ('rectime', -999, 'int32'),
            #('final_rectime', -999, 'int32'),  # Not present in .MAT, but could be helpful
            ('intrusion', -999, 'int16'),
            ('color_r', -999, 'float16'),
            ('color_g', -999, 'float16'),
            ('color_b', -999, 'float16'),
            ('font', '', 'S32'),
            ('case', '', 'S8'),
            ('rejected', False, 'b1'),
            ('rej_time', -999, 'int32'),

            ('artifactMS', -999, 'int32'),
            ('artifactNum', -999, 'int32'),
            ('artifactFrac', -999, 'float16'),
            ('artifactMeanMS', -999, 'float16'),
            ('badEvent', -999, 'int16'),
            ('badEventChannel', -999, list)
        )

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(LTPFRSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files)
        self._wordpool = np.array([x.strip() for x in open(files['wordpool']).readlines()])
        self._recog_ann = []  # During the recog portion, records the lines of the .ann file currently being used
        self._presented = set()
        self._session = -999
        self._trial = -999
        self._serialpos = -999
        self._is_finished_recall = False

        # Holds distractor ID and performance info so it can be entered into the 'WORD' event following the distractor
        self._distractor = 0
        self._math_correct = -999

        self._listtype = -1
        self._is_ffr = False

        self._recog_starttime = 0
        self._recog_endtime = 0
        self._recog_pres_mstime = -999
        self._mstime_recstart = -999
        self._recog_conf_mstime = -999
        self._rej_mstime = -999

        self._add_fields(*self._ltpfr_fields())
        self._add_type_to_new_event(
            SESS_START=self.event_sess_start,
            FR_PRES=self.event_fr_pres,
            SLOW_MSG=self.event_default,
            KEY_MSG=self.event_default,
            REC_START=self.event_recstart,
            REST=self._event_skip,
            REST_REWET=self.event_default,
            REJECT=self.get_rej_mstime,
            SESS_END=self._event_skip,
            SESSION_SKIPPED=self._event_skip,
            RECOG_START=self._event_skip,
            RECOG_END=self._event_skip,
            DISTRACTOR=self.event_distractor,
            FFR_START=self._event_skip,
            MATH_TOTAL_SCORE=self._event_skip,
            RECOG_PRES=self.event_recog,
            RECOG_CR_PRES=self._event_skip,
            RECOG_FEEDBACK=self.recog_end
        )
        self._add_type_to_modify_events(
            SESS_START=self.modify_session,
            REC_START=self.modify_recalls,
            FFR_START=self.modify_ffr,
            RECOG_START=self.end_recall,
            RECOG_FEEDBACK=self.modify_recog,
            REJECT=self.modify_rejection
        )


    """=======EVENT CREATION======="""
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

    def event_recstart(self, split_line):
        self._mstime_recstart = int(split_line[0])
        if self._is_finished_recall:
            # Update trial number, since each block of recogs begins with a REC_START event
            # Note that any erroneous REC_STARTs that occur between FFR_START and RECOG_START will have nonsensical
            # trial numbers
            self._trial += 1
            self._recog_ann = self.parse_recog_ann(str(self._trial - 1))
        return self.event_default(split_line)

    def event_fr_pres(self, split_line):
        self.update_trial(split_line)
        self._serialpos += 1
        event = self.event_default(split_line)
        event.type = 'WORD'
        event.word = split_line[4]
        event.wordno = int(split_line[5])
        task = int(split_line[6])
        event.task = task
        event.resp = int(split_line[7])
        rt = int(split_line[8])
        event.rt = rt if rt > 0 else -999
        event.color_r, event.color_g, event.color_b = (1, 1, 1) if (split_line[9] == 'white') else literal_eval(split_line[9])
        event.font = split_line[10]
        event.case = split_line[11]
        event.serialpos = self._serialpos
        event.distractor = self._distractor
        event.math_correct = self._math_correct
        # Add the ID number of the word to the set of all words that have been presented
        self._presented.add(int(split_line[5]))
        # Use the task type information to determine listtype; listtype is retroactively added on during modify_recall
        self._listtype = 2 if ((self._listtype == 0 and task == 1) or (self._listtype == 1 and task == 0) or self._listtype == 2) else task
        return event

    def event_distractor(self, split_line):
        """
        Extracts the distractor ID number and math_correct info from the session log, without creating an actual event
        :param split_line:
        :return:
        """
        self._distractor = split_line[6]
        self._math_correct = split_line[7]
        return False

    def event_recog(self, split_line):
        event = self.event_default(split_line)
        self._recog_pres_mstime = int(split_line[0])
        self._recog_starttime = self._recog_pres_mstime - self._mstime_recstart
        # Determine whether the word is a target or lure
        wordno = int(split_line[5])
        event.type = 'RECOG_TARGET' if wordno in self._presented else 'RECOG_LURE'
        # Fill in information available in split_line
        event.word = split_line[4]
        event.wordno = wordno
        return event

    def recog_end(self, split_line):
        self._recog_endtime = int(split_line[0]) - self._mstime_recstart
        return False

    def get_rej_mstime(self, split_line):
        self._rej_mstime = int(split_line[0])
        return False

    """=======EVENT MODIFIERS======="""
    def modify_session(self, events):
        """
        Applies session number to all previous events.
        """
        events.session = self._session
        return events

    def modify_recalls(self, events):
        # If the FFR recall has already been conducted or recog has begun, do not modify recalls when REC_START is read
        if self._is_finished_recall:
            return events

        if not self._is_ffr:
            events = self.modify_final_distractor_and_listtype(self._trial, events).view(np.recarray)

        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        if self._is_ffr:
            events[-1].type = 'REC_START'
            events[-1].trial = -999

        # Get list of recalls from the .ann file for the current list; each recall is (rectime, wordno, word)
        ann_outputs = self._parse_ann_file('ffr') if self._is_ffr else self._parse_ann_file(str(self._trial - 1))
        for recall in ann_outputs:
            word = recall[-1]
            # Skip vocalizations during free recall
            if word == '<>' or word == 'V' or word == '!':
                continue
            new_event = self._empty_event
            new_event.trial = -999 if self._is_ffr else self._trial
            new_event.session = self._session
            new_event.rectime = int(round(float(recall[0])))
            new_event.mstime = rec_start_time + new_event.rectime
            new_event.msoffset = 20
            new_event.word = word
            new_event.wordno = int(recall[1])
            new_event.type = 'REC_WORD'

            # Add FFR to the type if part of the FFR recall
            if self._is_ffr:
                new_event.type = 'FFR_' + str(new_event.type)
                # Sets field such that future REC_START events will be treated as default events
                # The FFR recall should be the final time that the full modify_recalls procedure is followed
                self._is_finished_recall = True

            # If XLI
            if recall[1] == -1:
                new_event.intrusion = -1
            else:  # Correct recall or PLI or XLI from latter list
                # Determines which list the recalled word was from (gives [] if from a future list)
                pres_mask = self.find_presentation(new_event.wordno, events)
                pres_trial = np.unique(events[pres_mask].trial)
                # Correct recall or PLI
                if len(pres_trial) == 1:
                    # Determines how many lists back the recalled word was presented
                    new_event.intrusion = -999 if self._is_ffr else self._trial - pres_trial[0]
                    # Retrieve information about which distractors were used during the word's presentation
                    new_event.distractor = np.unique(events[pres_mask].distractor)
                    new_event.final_distractor = np.unique(events[pres_mask].final_distractor)
                    # Retrieve the recalled word's serial position in its list, along with task information
                    new_event.serialpos = np.unique(events[pres_mask].serialpos)
                    new_event.listtype = np.unique(events[pres_mask].listtype)
                    new_event.task = np.unique(events[pres_mask].task)
                    new_event.resp = np.unique(events[pres_mask].resp)
                    new_event.rt = np.unique(events[pres_mask].rt)
                    # Correct recall, i.e. word is from the most recent list
                    if new_event.intrusion == 0:
                        new_event.recalled = True
                        # Retroactively log on the word pres event that the word was recalled, and when it was recalled
                        if not any(events[pres_mask].recalled):
                            events.recalled[pres_mask] = True
                            events.rectime[pres_mask] = new_event.rectime
                    elif self._is_ffr:
                        new_event.studytrial = pres_trial
                        events.finalrecalled[pres_mask] = True
                    else:
                        events.intruded[pres_mask] = new_event.intrusion
                else:  # XLI from later list
                    new_event.intrusion = -1

            # Add recall event to events array
            events = np.append(events, new_event).view(np.recarray)

        return events

    def modify_recog(self, events):
        """
        Retroactively adds whether a word was recognized to the original word presentation event.
        :param events: The list of all prior events
        :return: The modified event list
        """
        events = events.view(np.recarray)

        is_target = True if events[-1].type == 'RECOG_TARGET' else False
        was_recognized = False
        word = events[-1].word
        wordno = events[-1].wordno
        new_events = []
        current_block = []
        recog = (-999, None, None)
        conf = (None, -997, None)  # default conf response is -997 so when 2 is subtracted later it gives -999
        recog_rt = -999

        # Ignore any responses and vocalizations that occur before the presentation of the word
        while len(self._recog_ann) > 0 and int(self._recog_ann[0][0]) < self._recog_starttime:
            self._recog_ann = self._recog_ann[1:]

        # Get all lines from the .ann that occur between the presentation of the word and the feedback presentation
        while len(self._recog_ann) > 0 and int(self._recog_ann[0][0]) <= self._recog_endtime:
            current_block.append(self._recog_ann[0])
            self._recog_ann = self._recog_ann[1:]

        # Find the last conf response in the block and use it as the RECOG_CONF event; all earlier conf responses
        # will be treated as RECOG_RESP_VV events (in cases where the participant changes their answer)
        i = len(current_block) - 1
        while i >= 0:
            event = self._empty_event
            rectime = current_block[i][0]
            resp = current_block[i][1]

            # Set timing information
            event.msoffset = 20
            event.rectime = rectime
            event.mstime = self._mstime_recstart + rectime
            event.trial = self._trial
            # If the current line is a confidence response, create the RECOG_CONF event and stop searching
            if resp in range(3, 8):
                conf = current_block[i]
                event.type = 'RECOG_CONF'
                new_events = [event] + new_events
                i -= 1
                break
            # If the current line is a vocalization, create a RECOG_RESP_VV event
            elif resp == -1:
                # If the participant gave a confidence number other than 1 through 5, it will be marked with -1, but
                # still use whatever number they gave as the confidence response
                try:
                    resp = int(current_block[i][2])
                    conf = (current_block[i][0], resp + 2, resp)
                    event.type = 'RECOG_CONF'
                    new_events = [event] + new_events
                    i -= 1
                    break
                except ValueError:
                    event.type = 'RECOG_RESP_VV'
            # If a recognition response is found, continue to the next while loop
            elif resp in (1, 2):
                break
            else:
                raise Exception('Invalid integer response %d found during recognition of word %s' % (resp, word))
            # Add the event to new_events
            new_events = [event] + new_events
            i -= 1

        if conf == (None, None, None):
            logger.warn('Confidence response not found for word ' + word)

        # Continue searching up the current response block for the last recognition response and use it as the
        # RECOG_RESP event. All earlier recog responses will be treated as RECOG_RESP_VV events.
        while i >= 0:
            event = self._empty_event
            rectime = current_block[i][0]
            resp = current_block[i][1]

            # Set timing information
            event.msoffset = 20
            event.rectime = rectime
            event.mstime = self._mstime_recstart + rectime
            event.trial = self._trial
            # Create a RECOG_RESP event when the recognition response is found and stop searching
            if resp in (1, 2):
                recog = current_block[i]
                # Calculate recog_rt as the response rectime + mstime(recstart) - mstime(presentation)
                recog_rt = event.mstime - self._recog_pres_mstime
                event.type = 'RECOG_RESP'
                if resp == 1:
                    was_recognized = True
                new_events = [event] + new_events
                i -= 1
                break
            # Create a RECOG_RESP_VV event if the line is a confidence response other than the final one they made
            elif resp in range(3, 8):
                event.type = 'RECOG_RESP_VV'
                event.recog_conf = resp - 2
            # If the current line is a vocalization create a RECOG_RESP_VV event
            elif resp == -1:
                # If the participant gave a confidence number other than 1 through 5, it will be marked with -1, but
                # still treat whatever number they gave as a confidence response
                try:
                    resp = int(current_block[i][2])
                    event.recog_conf = resp
                    event.type = 'RECOG_RESP_VV'
                    new_events = [event] + new_events
                except ValueError:
                    event.type = 'RECOG_RESP_VV'
            else:
                raise Exception('Invalid integer response %d found during recognition of word %s' % (resp, word))
            # Add the event to new_events
            new_events = [event] + new_events
            i -= 1

        if recog_rt == -999:
            logger.warn('Recognition response not found for word ' + word)

        # Treat all responses and vocalizations before the RECOG_RESP event as RECOG_RESP_VV events
        while i >= 0:
            event = self._empty_event
            rectime = current_block[i][0]
            resp = current_block[i][1]

            # Set timing information
            event.msoffset = 20
            event.rectime = rectime
            event.mstime = self._mstime_recstart + rectime
            event.trial = self._trial
            if resp in (1, 2):
                event.type = 'RECOG_RESP_VV'
                event.recog_resp = 1 if resp == 1 else 0
                # The original presentation of a word gets marked as recognized even if the participant changed their
                # recognition response from yes to no
                if resp == 1:
                    was_recognized = True
            elif resp in range(3, 8):
                event.type = 'RECOG_RESP_VV'
                event.recog_conf = resp - 2
            elif resp == -1:
                # If the participant gave a confidence number other than 1 through 5, it will be marked with -1, but
                # still treat whatever number they gave as a confidence response
                try:
                    resp = int(current_block[i][2])
                    event.recog_conf = resp
                    event.type = 'RECOG_RESP_VV'
                    new_events = [event] + new_events
                except ValueError:
                    event.type = 'RECOG_RESP_VV'
            else:
                raise Exception('Invalid integer response %d found during recognition of word %s' % (resp, word))
            new_events = [event] + new_events
            i -= 1

        events[-1].rectime = recog[0]
        new_events = [events[-1]] + new_events

        for ev in new_events:
            if ev.type != 'RECOG_RESP_VV':
                ev.recog_rt = recog_rt
                ev.recog_conf = conf[1] - 2
                # if no recognition response was given, leave the recog_resp as -999, otherwise fill it in
                if recog_rt != -999:
                    ev.recog_resp = 1 if recog[1] == 1 else 0
            ev.word = word
            ev.wordno = wordno

        # For targets, extract appropriate info from the original word presentation event
        if is_target:
            pres_mask = self.find_presentation(wordno, events)
            studytrial = np.unique(events[pres_mask].trial)
            listtype = np.unique(events[pres_mask].listtype)
            serialpos = np.unique(events[pres_mask].serialpos)
            task = np.unique(events[pres_mask].task)
            resp = np.unique(events[pres_mask].resp)
            rt = np.unique(events[pres_mask].rt)
            recalled = np.unique(events[pres_mask].recalled)
            intruded = np.unique(events[pres_mask].intruded)
            final_recalled = np.unique(events[pres_mask].finalrecalled)

            # Fill in this information for the recog events
            events[-1].recalled = recalled
            events[-1].intruded = intruded
            events[-1].finalrecalled = final_recalled
            for ev in new_events:
                ev.studytrial = studytrial
                ev.listtype = listtype
                ev.serialpos = serialpos
                ev.task = task
                ev.resp = resp
                ev.rt = rt

            # Log in the word presentation event whether the word was recognized, even if the participant changed their
            # answer to no after already having said yes.
            events.recognized[pres_mask] = True if was_recognized else False

        # Add new events to the end of the events list
        for ev in new_events[1:]:
            events = np.append(events, ev)

        return events

    def modify_final_distractor_and_listtype(self, trial, events):
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
        events.listtype[this_trial] = self._listtype
        self._distractor = 0
        self._math_correct = -999
        self._listtype = -1
        return events

    def modify_ffr(self, events):
        """
        When final free recall is reached, set a flag to indicate that modify_recalls should treat this recall trial as
        an FFR recall instead of a standard recall.
        """
        self._is_ffr = True
        return events

    def modify_rejection(self, events):
        """
        When a rejection is made, find the event directly preceding the mstime of the rejection. If it is a recall
        event, modify the event with the relevant rejection info. If the rejection did not follow a recall event,
        it must have been erroneous, so do nothing.
        :param events: The list of all prior events
        :return: The modified event list
        """
        i = -1
        while events[i].mstime > self._rej_mstime:
            i -= 1
        if events[i].type in ('REC_WORD', 'REC_WORD_VV'):
            events[i].rej_time = self._rej_mstime - self._mstime_recstart
            events[i].rejected = True
        return events

    def end_recall(self, events):
        """
        Records that the recall portion of the session has ended, and resets the trial counter for the beginning of the
        recog portion of the session. Note that trial is set to 0 because it will be increased to 1 when the first recog
        list begins. The events list is not used, but must be passed through in order to match the format of a "modify"
        style function.
        """
        self._is_finished_recall = True
        self._trial = 0
        return events

    def update_trial(self, split_line):
        """
        Checks whether a new word list has been reached, and resets the serial position if so
        :param split_line:
        """
        current_trial = int(split_line[3]) + 1
        if self._trial != current_trial:
            self._serialpos = 0
            self._trial = current_trial

    def parse_recog_ann(self, trial):
        ann_id = 'r' + trial
        if ann_id not in self._ann_files:
            return []
        ann_file = self._ann_files[ann_id]
        lines = open(ann_file, 'r').readlines()
        data_lines = [line for line in lines if line[0] != '#']  # May need to add regex line-matcher if problems arise
        split_lines = [line.split() for line in data_lines if len(line.split()) == 3]
        # line[0] is rectime; line[1] is either the recog_resp, 2 greater than the recog_conf, or -1 for vocalizations
        # line[2] is redundant except in the case where the participant gives a confidence judgment outside of the range
        # 1 through 5. In such a case, the invalid confidence judgment will be taken as the subject's conf response.
        return [(int(round(float(line[0]))), int(line[1]), line[2]) for line in split_lines]

    @staticmethod
    def find_presentation(wordno, events):
        events = events.view(np.recarray)
        return np.logical_and(events.wordno == wordno, events.type == 'WORD')
