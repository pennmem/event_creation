from base_log_parser import BaseSessionLogParser
import numpy as np
from ast import literal_eval


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
            ('recog_rt', -999, 'int16'),
            ('item_name', '', 'S16'),  # Calling this 'item' will break things, due to the built-in recarray.item method
            ('item_num', -999, 'int16'),
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
        self._mstime_recstart = -999
        self._recog_conf_mstime = -999
        self._rej_mstime = -999

        self._add_fields(*self._ltpfr_fields())
        self._add_type_to_new_event(
            SESS_START=self.event_sess_start,
            FR_PRES=self.event_fr_pres,
            SLOW_MSG=self._event_skip,
            KEY_MSG=self._event_skip,
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
        event.item_name = split_line[4]
        event.item_num = int(split_line[5])
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
        # self.update_trial(split_line)  # This line would only be needed if REC_START did not appear before each list
        event = self.event_default(split_line)
        self._recog_starttime = int(split_line[0]) - self._mstime_recstart
        # Determine whether the word is a target or lure
        wordno = int(split_line[5])
        event.type = 'RECOG_TARGET' if wordno in self._presented else 'RECOG_LURE'
        # Fill in information available in split_line
        event.item_name = split_line[4]
        event.item_num = wordno
        return event

    def recog_end(self, split_line):
        self._recog_endtime = int(split_line[0]) - self._mstime_recstart
        return False

    def get_rej_mstime(self, split_line):
        self._rej_mstime = self._mstime_recstart
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

        events = self.modify_final_distractor_and_listtype(self._trial, events).view(np.recarray)

        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        if self._is_ffr:
            events[-1].type = 'REC_START'

        # Get list of recalls from the .ann file for the current list; each recall is (rectime, wordno, word)
        ann_outputs = self._parse_ann_file('ffr') if self._is_ffr else self._parse_ann_file(str(self._trial - 1))
        for recall in ann_outputs:
            word = recall[-1]
            new_event = self._empty_event
            new_event.trial = -999 if self._is_ffr else self._trial
            new_event.session = self._session
            new_event.rectime = int(round(float(recall[0])))
            # if self._is_ffr:
            #    new_event.final_rectime = new_event.rectime  # For FFR recalls, count as both rectime and final_rectime
            new_event.mstime = rec_start_time + new_event.rectime
            new_event.msoffset = 20  # Check why offset is 20 and whether this is true for FFR, too
            new_event.item_name = word
            new_event.item_num = int(recall[1])

            # If vocalization
            if word == '<>' or word == 'V' or word == '!':
                new_event.type = 'REC_WORD_VV'
            else:
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
                    new_event.intrusion = 0 if self._is_ffr else self._trial - pres_trial[0]
                    # Correct recall if word is from the most recent list
                    if new_event.intrusion == 0:
                        # Retrieve the recalled word's serial position in its list, as well as the distractor used
                        new_event.serialpos = np.unique(events[pres_mask].serialpos)
                        new_event.distractor = np.unique(events[pres_mask].distractor)
                        new_event.final_distractor = np.unique(events[pres_mask].final_distractor)
                        new_event.listtype = np.unique(events[pres_mask].listtype)
                        new_event.task = np.unique(events[pres_mask].task)
                        new_event.resp = np.unique(events[pres_mask].resp)
                        new_event.rt = np.unique(events[pres_mask].rt)

                        # Retroactively log on the word pres event that the word was recalled, and when it was recalled
                        if self._is_ffr:
                            if not any(events.finalrecalled[pres_mask]):
                                # For FFR recalls, set studytrial equal to the trial number of the original presentation
                                new_event.studytrial = np.unique(events[pres_mask].trial)
                                events.finalrecalled[pres_mask] = True
                                # events.final_rectime[pres_mask] = new_event.rectime
                                new_event.finalrecalled = True
                        else:
                            if not any(events.recalled[pres_mask]):
                                events.recalled[pres_mask] = True
                                events.rectime[pres_mask] = new_event.rectime
                                new_event.recalled = True
                    else:
                        events.intruded[pres_mask] = new_event.intrusion
                else:  # XLI
                    new_event.intrusion = -1

            # Add recall event to events array
            events = np.append(events, new_event).view(np.recarray)

        return events

    '''
    def old_modify_recog(self, events):
        """
        Retroactively adds whether a word was recognized to the original word presentation event.
        :param events: The list of all prior events
        :return: The modified event list
        """
        events = events.view(np.recarray)

        is_target = True if events[-1].type == 'RECOG_TARGET' else False

        # Create RECOG_RESP and RECOG_CONF events
        resp_event = self._empty_event
        conf_event = self._empty_event
        new_events = [events[-1]]

        # Handle any RECOG_RESP_VV events that occur before the actual response is given
        while int(self._recog_ann[0][1]) == -1:
            if float(self._recog_ann[0][0]) > 0:
                recog_vv = self._empty_event
                recog_vv.type = 'RECOG_RESP_VV'
                recog_vv.msoffset = 20
                new_events.append(recog_vv)
            self._recog_ann = self._recog_ann[1:]

        # The first line of the recog .ann is the recog response, the second line is the confidence rating
        # RECOG_RESP_VV events may occur in between the two
        recog = self._recog_ann[0]
        new_events.append(resp_event)
        self._recog_ann = self._recog_ann[1:]
        while int(self._recog_ann[0][1]) == -1:
            if float(self._recog_ann[0][0]) > 0:
                recog_vv = self._empty_event
                recog_vv.type = 'RECOG_RESP_VV'
                recog_vv.msoffset = 20
                new_events.append(recog_vv)
            self._recog_ann = self._recog_ann[1:]
        conf = self._recog_ann[0]
        new_events.append(conf_event)
        self._recog_ann = self._recog_ann[1:] if len(self._recog_ann) > 1 else []

        # Adjust fields that cannot be read from .ann
        resp_event.type = 'RECOG_RESP'
        conf_event.type = 'RECOG_CONF'
        resp_event.msoffset = 20
        conf_event.msoffset = 20
        word = events[-1].word
        wordno = events[-1].wordno

        # Retrieve rectime, recog_resp, and recog_conf from the .ann and enter it into the recog events
        events[-1].rectime = int(round(float(recog[0])))
        resp_event.rectime = int(round(float(recog[0])))
        conf_event.rectime = int(round(float(conf[0])))

        # Determine mstime for the responses based on the mstime of the REC_START preceding the trial
        resp_event.mstime = self._mstime_recstart + resp_event.rectime
        conf_event.mstime = self._mstime_recstart + conf_event.rectime

        # Calculate recog_rt as the response rectime - mstime(pres) + mstime(recstart)
        recog_rt = int(recog[0]) - resp_event.mstime + self._mstime_recstart
        for ev in new_events:
            if int(recog[1]) == 1 and ev.type != 'RECOG_RESP_VV':
                ev.recog_resp = True
            ev.recog_rt = recog_rt
            ev.recog_conf = int(conf[1]) - 2
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

            # Fill in this information for the recog events
            events[-1].recalled = recalled
            events[-1].intruded = intruded
            for ev in new_events:
                ev.studytrial = studytrial
                ev.listtype = listtype
                ev.serialpos = serialpos
                ev.task = task
                ev.resp = resp
                ev.rt = rt

            # Log in the word presentation event whether the word was recognized
            events.recognized[pres_mask] = True if int(recog[1]) == 1 else False

        # Add new events to the end of the events list
        events = np.append(events, new_events[1:]).view(np.recarray)

        return events
    '''
    def modify_recog(self, events):
        """
        Retroactively adds whether a word was recognized to the original word presentation event.
        :param events: The list of all prior events
        :return: The modified event list
        """
        events = events.view(np.recarray)

        is_target = True if events[-1].type == 'RECOG_TARGET' else False
        word = events[-1].word
        wordno = events[-1].wordno
        new_events = []
        current_block = []
        recog = ['', '']
        conf = ['', '']

        resp_event = self._empty_event
        conf_event = self._empty_event

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
            resp = int(current_block[i][1])
            # If the top of the block is reached before finding a confidence response, then there is either no
            # confidence response or no recognition response
            if i == 0:
                raise Exception('Recognition or confidence response not found for word ' + word)
            # If the current line is a confidence response, create the RECOG_CONF event and stop searching
            elif resp in range(3, 8):
                conf = current_block[i]
                conf_event.type = 'RECOG_CONF'
                conf_event.msoffset = 20
                conf_event.rectime = int(round(float(conf[0])))
                conf_event.mstime = self._mstime_recstart + conf_event.rectime
                new_events = [conf_event] + new_events
                i -= 1
                break
            # If the current line is a vocalization, create a RECOG_RESP_VV event
            elif resp == -1:
                recog_vv = self._empty_event
                recog_vv.type = 'RECOG_RESP_VV'
                recog_vv.msoffset = 20
                new_events = [recog_vv] + new_events
                i -= 1
            elif resp in (1, 2):
                raise Exception('Recogntion response found after confidence response for word ' + word)
            else:
                raise Exception('Invalid integer response found during recognition of word ' + word)

        # Continue searching up the current response block for the last recognition response and use it as the
        # RECOG_RESP event. All earlier recog responses will be treated as RECOG_RESP_VV events.
        while i >= 0:
            resp = int(current_block[i][1])
            # Create a RECOG_RESP event when the recognition response is found and stop searching
            if resp in (1, 2):
                recog = current_block[i]
                resp_event.type = 'RECOG_RESP'
                resp_event.msoffset = 20
                resp_event.rectime = int(round(float(recog[0])))
                resp_event.mstime = self._mstime_recstart + resp_event.rectime
                new_events = [resp_event] + new_events
                i -= 1
                break
            # If the current line is a vocalization or (erroneous) confidence response, create a RECOG_RESP_VV event
            elif resp == -1 or resp in range(3, 8):
                recog_vv = self._empty_event
                recog_vv.type = 'RECOG_RESP_VV'
                recog_vv.msoffset = 20
                new_events = [recog_vv] + new_events
                i -= 1
            else:
                raise Exception('Invalid integer response found during recognition of word ' + word)

        # Treat all responses and vocalizations before the RECOG_RESP event as RECOG_RESP_VV events
        while i >= 0:
            resp = int(current_block[i][1])
            if resp == -1 or resp in range(1, 8):
                recog_vv = self._empty_event
                recog_vv.type = 'RECOG_RESP_VV'
                recog_vv.msoffset = 20
                new_events = [recog_vv] + new_events
                i -= 1
            else:
                raise Exception('Invalid integer response found during recognition of word ' + word)

        events[-1].rectime = int(round(float(recog[0])))
        new_events = [events[-1]] + new_events

        # Calculate recog_rt as the response rectime - mstime(pres) + mstime(recstart)
        recog_rt = int(recog[0]) - resp_event.mstime + self._mstime_recstart
        for ev in new_events:
            if int(recog[1]) == 1 and ev.type != 'RECOG_RESP_VV':
                ev.recog_resp = True
            ev.recog_rt = recog_rt
            ev.recog_conf = int(conf[1]) - 2
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

            # Fill in this information for the recog events
            events[-1].recalled = recalled
            events[-1].intruded = intruded
            for ev in new_events:
                ev.studytrial = studytrial
                ev.listtype = listtype
                ev.serialpos = serialpos
                ev.task = task
                ev.resp = resp
                ev.rt = rt
                ev.trial = self._trial

            # Log in the word presentation event whether the word was recognized
            events.recognized[pres_mask] = True if int(recog[1]) == 1 else False

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
        if events[i].type == 'REC_WORD' or 'REC_WORD_VV':
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
        # line[0] is rectime, line[1] is either the recog_resp or 2 greater than the recog_conf, line[2] is redundant
        return [(int(round(float(line[0]))), int(round(float(line[1])))) for line in split_lines]

    @staticmethod
    def find_presentation(wordno, events):
        events = events.view(np.recarray)
        return np.logical_and(events.wordno == wordno, events.type == 'WORD')
