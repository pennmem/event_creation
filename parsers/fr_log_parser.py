from base_log_parser import BaseSessionLogParser, UnknownExperimentTypeException
from system2_log_parser import System2LogParser
from viewers.view_recarray import strip_accents
import numpy as np
import os
import re
import json
import pandas as pd
import sqlite3

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
            ('intrusion', -999, 'int16'),
            ('exp_version', '', 'S64'),
            ('stim_list', False, 'b1'),
            ('is_stim', False, 'b1'),
            ('rectime',-999,'int16'),
        )

    # FR2 specific fields
    FR2_STIM_DURATION= 4600
    FR2_STIM_PULSE_FREQUENCY = 50
    FR2_STIM_N_PULSES = 230
    FR2_STIM_BURST_FREQUENCY = 1
    FR2_STIM_N_BURSTS = 1
    FR2_STIM_PULSE_WIDTH = 300

    def __init__(self, protocol, subject, montage, experiment, session, files,**kwargs):
        kwargs['include_stim_params'] = True
        super(FRSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
                                                 **kwargs)
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
        self._presented = False


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
            WORD_START =self.event_word,
            WORD_END = self.event_word_off,
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
        try:
            in_wordpool = bool((self._wordpool==self._word))
        except ValueError:
            in_wordpool = (self._wordpool==self._word).any()
        if in_wordpool:
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
                        new_event.serialpos = np.unique(events[pres_mask].serialpos)
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

    # def parse(self):
    #     events = super(FRSessionLogParser,self).parse()
    #     events = events.view(np.recarray)
    #     return self.add_baseline_events(events)

    @staticmethod
    def add_baseline_events(sess_events):
        '''
        Match recall events to matching baseline periods of failure to recall.
        Baseline events all begin at least 1000 ms after a vocalization, and end at least 1000 ms before a vocalization.
        Each recall event is matched, wherever possible, to a valid baseline period from a different list within 3 seconds
         relative to the onset of the recall period.

        Parameters:
        -----------
        events: The event structure in which to incorporate these baseline periods

        '''

        rec_events = sess_events[(sess_events.type == 'REC_WORD') & (sess_events.intrusion == 0)]
        voc_events = sess_events[((sess_events.type == 'REC_WORD') | (sess_events.type == 'REC_WORD_VV'))]
        starts = sess_events[(sess_events.type == 'REC_START')]
        ends = sess_events[(sess_events.type == 'REC_END')]
        rec_lists = tuple(np.unique(starts.list))
        times = [voc_events[(voc_events.list == lst)].mstime for lst in rec_lists]
        start_times = starts.mstime
        end_times = ends.mstime
        epochs = free_epochs(times, 500, 1000, 1000, start=start_times, end=end_times)
        rel_times = [t - i for (t, i) in
                     zip([rec_events[rec_events.list == lst].mstime for lst in rec_lists], start_times)]
        rel_epochs = epochs - start_times[:, None]
        full_match_accum = np.zeros(epochs.shape, dtype=np.bool)
        for (i, rec_times_list) in enumerate(rel_times):
            is_match = np.empty(epochs.shape, dtype=np.bool)
            is_match[...] = False
            for t in rec_times_list:
                is_match_tmp = np.abs((rel_epochs - t)) < 3000
                is_match_tmp[i, ...] = False
                good_locs = np.where(is_match_tmp & (~full_match_accum))
                if len(good_locs[0]):
                    choice_position = np.argmin(np.mod(good_locs[0] - i, len(good_locs[0])))
                    choice_inds = (good_locs[0][choice_position], good_locs[1][choice_position])
                    full_match_accum[choice_inds] = True
        matching_epochs = epochs[full_match_accum]
        new_events = np.zeros(len(matching_epochs), dtype=sess_events.dtype).view(np.recarray)
        for i, _ in enumerate(new_events):
            new_events[i].mstime = matching_epochs[i]
            new_events[i].type = 'REC_BASE'
        new_events.recalled = 0
        merged_events = np.concatenate((sess_events, new_events)).view(np.recarray)
        merged_events.sort(order='mstime')
        for (i, event) in enumerate(merged_events):
            if event.type == 'REC_BASE':
                merged_events[i].session = merged_events[i - 1].session
                merged_events[i].list = merged_events[i - 1].list
                merged_events[i].eegfile = merged_events[i - 1].eegfile
                merged_events[i].eegoffset = merged_events[i - 1].eegoffset + (
                merged_events[i].mstime - merged_events[i - 1].mstime)
        return merged_events

    @staticmethod
    def find_presentation(word, events):
        events = events.view(np.recarray)
        return np.logical_and(events.item_name == word, np.logical_or(events.type == 'WORD', events.type == 'PRACTICE_WORD'))


def free_epochs(times, duration, pre, post, start=None, end=None):
    # (list(vector(int))*int*int*int) -> list(vector(int))
    """
    Given a list of event times, find epochs between them when nothing is happening

    Parameters:
    -----------

    times:
        An iterable of 1-d numpy arrays, each of which indicates event times

    duration: int
        The length of the desired empty epochs

    pre: int
        the time before each event to exclude

    post: int
        The time after each event to exclude

    """
    n_trials = len(times)
    epoch_times = []
    for i in range(n_trials):
        ext_times = times[i]
        if start is not None:
            ext_times = np.append([start[i]], ext_times)
        if end is not None:
            ext_times = np.append(ext_times, [end[i]])
        pre_times = ext_times - pre
        post_times = ext_times + post
        interval_durations = pre_times[1:] - post_times[:-1]
        free_intervals = np.where(interval_durations > duration)[0]
        trial_epoch_times = []
        for interval in free_intervals:
            begin = post_times[interval]
            finish = pre_times[interval + 1] - duration
            interval_epoch_times = range(begin, finish, duration)
            trial_epoch_times.extend(interval_epoch_times)
        epoch_times.append(np.array(trial_epoch_times))
    epoch_array = np.empty((n_trials, max([len(x) for x in epoch_times])))
    epoch_array[...] = -np.inf
    for i, epoch in enumerate(epoch_times):
        epoch_array[i, :len(epoch)] = epoch
    return epoch_array





class FRSys31SessionParser(BaseSessionLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files,
                 allow_unparsed_events=False, include_stim_params=False):
        super(FRSys31SessionParser,self).__init__(protocol, subject, montage, experiment, session, files,
                 allow_unparsed_events=False, include_stim_params=False,primary_log='session_sql')

    # def _get_raw_event_type(self, split_line):

    def _read_primary_log(self):
        # path = "sqlite://{sqlite}".format(sqlite=self._primary_log)
        conn = sqlite3.connect(self._primary_log)
        query = 'SELECT msg FROM logs WHERE name = "events"'
        msgs = [json.loads(msg) for msg in pd.read_sql_query(query,
                                                             conn).msg.values]
        return msgs


if __name__ == '__main__':
    files = {
        'session_log':'/Users/leond/Documents/PS4_FR5/task/R1234M/session_0/session.log',
        'wordpool': '/Users/leond/Documents/PS4_FR5/task/R1234M/RAM_wordpool.txt',
        'session_sql':'/Users/leond/Documents/PS4_FR5/task/R1234M/session_0/session.sqlite'
    }

    frslp = FRSys31SessionParser('r1', 'R1999X', 0.0, 'FR1', 0, files)
    events=  frslp.parse()
