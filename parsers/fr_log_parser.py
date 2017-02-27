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
            ('item_name', 'X', 'S64'),
            ('item_num', -999, 'int16'),
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

    def __init__(self, protocol, subject, montage, experiment, session, files,**kwargs):
        super(FRSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
                                                 include_stim_params=True,**kwargs)
        if 'no_accent_wordpool' in files:
            wordpool_type = 'no_accent_wordpool'
        else:
            wordpool_type = 'wordpool'
        try:
            self._wordpool = np.array([x.strip() for x in open(files[wordpool_type]).readlines()])
        except KeyError as key_error:
            if type(self) is FRSessionLogParser:
                raise key_error
            else:
                #Subclasses are allowed to not have a word pool
                self._wordpool = None

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

        self._recog_starttime = 0
        self._recog_endtime = 0
        self._recog_pres_mstime = -999
        self._mstime_recstart = -999
        self._recog_conf_mstime = -999
        self._rej_mstime = -999


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
            VOICE=self.event_default,
            RECOG_START=self._event_skip,
            RECOG_END = self._event_skip,
            RECOG_PRES = self.event_recog,
            RECOG_FEEDBACK=self.recog_end, #TODO: add in recognition events

        )
        self._add_type_to_modify_events(
            SESS_START=self.modify_session,
            REC_START=self.modify_recalls,
            RECOG_START=self.end_recall,
            RECOG_FEEDBACK=self.modify_recog,
        )

    @staticmethod
    def persist_fields_during_stim(event):
        if event['type'] == 'WORD':
            return ('list', 'serialpos', 'item_name', 'item_num', 'recalled',
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

    def event_recog(self, split_line):
        event = self.event_default(split_line)
        self._recog_pres_mstime = int(split_line[0])
        self._recog_starttime = self._recog_pres_mstime - self._mstime_recstart
        # Determine whether the word is a target or lure
        item_num = int(split_line[5])
        event.type = 'RECOG_TARGET' if item_num in self._presented else 'RECOG_LURE'
        # Fill in information available in split_line
        event.item_name = split_line[4]
        event.item_num = item_num
        return event

    def recog_end(self,split_line):
        def recog_end(self, split_line):
            self._recog_endtime = int(split_line[0]) - self._mstime_recstart
            return False

    def get_rej_mstime(self,split_line):
        self._rej_mstime = int(split_line[0])
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
        event.item_name = self._word
        if (self._wordpool == self._word).any():
            wordno = np.where(self._wordpool == self._word)
            event.item_num = wordno[0] + 1
        else:
            event.item_num = -1
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

    def modify_recog(self,events):
        events = events.view(np.recarray)

        is_target = True if events[-1].type == 'RECOG_TARGET' else False
        was_recognized = False
        item_name = events[-1].item_name
        item_num = events[-1].item_num
        new_events = []
        current_block = []
        r_ind = None
        c_ind = None

        # Ignore any responses and vocalizations that occur before the presentation of the word
        while len(self._recog_ann) > 0 and int(self._recog_ann[0][0]) < self._recog_starttime:
            self._recog_ann = self._recog_ann[1:]

        # Get all lines from the .ann that occur between the presentation of the word and the feedback presentation
        while len(self._recog_ann) > 0 and int(self._recog_ann[0][0]) <= self._recog_endtime:
            current_block.append(self._recog_ann[0])
            self._recog_ann = self._recog_ann[1:]

        for i in range(len(current_block)):
            line = current_block[i]
            event = self._empty_event
            event.type = 'RECOG_RESP_VV'
            rectime = line[0]
            resp = line[2]
            event.msoffset = 20
            event.rectime = rectime
            event.mstime = self._mstime_recstart + rectime
            event.trial = self._trial

            if resp == 'Y':
                event.recog_resp = 1
                r_ind = len(new_events)
                was_recognized = True
            elif resp == 'N':
                event.recog_resp = 0
                r_ind = len(new_events)
            else:
                try:
                    event.recog_conf = int(resp)
                    c_ind = len(new_events)
                except ValueError:
                    continue

            new_events.append(event)

        # Mark final recog response and confidence judgment and take those responses as the participant's final answers
        if r_ind is not None:
            new_events[r_ind].type = 'RECOG_RESP'
            rectime = events[r_ind].rectime
            resp = new_events[r_ind].recog_resp
            rt = new_events[r_ind].mstime - self._recog_pres_mstime
            events[-1].rectime = rectime
            events[-1].recog_resp = resp
            events[-1].recog_rt = rt
        if c_ind is not None:
            new_events[c_ind].type = 'RECOG_CONF'
            conf = new_events[c_ind].recog_conf
            events[-1].recog_conf = conf
            # Fill in confidence info on the recog event and recog info on the confidence event
            if r_ind is not None:
                events[r_ind].recog_conf = conf
                events[c_ind].recog_resp = resp
                events[c_ind].rt = rt

        for ev in new_events:
            ev.item_name = item_name
            ev.item_num = item_num

        # For targets, extract appropriate info from the original word presentation event
        if is_target:
            pres_mask = self.find_presentation(item_num, events)
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
            for ev in [events[-1]] + new_events:
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
        for ev in new_events:
            events = np.append(events, ev)

    def modify_recalls(self, events): # TODO: include recognition words
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
            new_event.item_name = word
            new_event.item_num = recall[1]

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


    @staticmethod
    def find_presentation(word, events):
        events = events.view(np.recarray)
        return np.logical_and(events.item_name == word, np.logical_or(events.type == 'WORD', events.type == 'PRACTICE_WORD'))


if __name__ == '__main__':
    files = {
        'session_log':'/Volumes/PATRIOT/R1999X/behavioral/FR1/session_0/session.log',
        'wordpool': '/Volumes/PATRIOT/R1999X/behavioral/FR1/RAM_wordpool.txt'
    }

    frslp = FRSessionLogParser('r1', 'R1999X', 0.0, 'FR1', 0, files)
    events=  frslp.parse()
