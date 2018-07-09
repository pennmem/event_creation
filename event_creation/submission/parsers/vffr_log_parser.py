import numpy as np
import pandas as pd
from . import dtypes
from .base_log_parser import BaseUnityLTPLogParser


class VFFRSessionLogParser(BaseUnityLTPLogParser):

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(VFFRSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files)
        self._trial = 0
        self._serialpos = 0
        self.practice = True
        self.current_word = ''
        self.current_num = -999

        # Infer wordpool for "first free recall" by finding all words presented in this session
        df = pd.read_json(self._primary_log, lines=True)
        df = df[df.type == 'stimulus cleared']['data']
        self.wordpool = set([row['word'].strip() for row in df])

        self._add_fields(*dtypes.vffr_fields)
        self._add_type_to_new_event(
            countdown=self.event_countdown,
            final_recall_start=self.event_ffr_start,  # Start of final free recall
            final_recall_stop=self.event_ffr_stop,  # End of final free recall
            recall_start=self.event_rec_start,  # Start of vocalization period
            recall_stop=self.event_rec_stop,  # End of vocalization period
            required_break_start=self.event_break_start,  # Start of mid-session break
            required_break_stop=self.event_break_stop,  # End of mid-session break
            stimulus=self.event_word_on,  # Start of word presentation
            stimulus_cleared=self.event_word_off,  # End of word presentation
        )
        self._add_type_to_modify_events(
            final_recall_start=self.modify_ffr,  # Parse FFR annotations and add recall information to events
            recall_start=self.modify_rec,  # Parse annotation for word vocalization and add a vocalization event
            recall_stop=self.modify_rec_stop,  # Mark events with whether the participant vocalized too early
            stimulus_cleared=self.modify_pres_dur  # Mark word presentation events with how long the word was onscreen
        )

    ###############
    #
    # EVENT CREATION FUNCTIONS
    #
    ###############

    def event_countdown(self, evdata):
        self._trial += 1
        event = self.event_default(evdata)
        event.type = 'COUNTDOWN'
        event.item_name = evdata['data']['displayed text']

    def event_word_on(self, evdata):
        # Get information on the word presentation and whether it is a practice item
        self._serialpos = evdata['data']['index'] + 1
        self.practice = evdata['data']['practice']
        self.current_word = evdata['data']['word'].strip()

        # Build event
        event = self.event_default(evdata)
        event.type = 'PRACTICE_WORD' if self.practice else 'WORD'
        event.item_name = self.current_word
        event.serialpos = self._serialpos
        if 'ltp word number' in evdata['data']:
            self.current_num = evdata['data']['ltp word number']
            event.item_num = self.current_num

        # Determine whether word was recalled during first free recall
        if np.any(self.events[self.events.type == 'FFR_REC_WORD'].item_name == event.item_name):
            event.recalled = True

        return event

    def event_word_off(self, evdata):
        event = self.event_default(evdata)
        event.type = 'PRACTICE_WORD_OFF' if self.practice else 'WORD_OFF'
        event.item_name = evdata['data']['word'].strip()
        event.serialpos = self._serialpos
        if event.item_name == self.current_word:
            event.item_num = self.current_num

        # Determine whether word was recalled during first free recall
        if np.any(self.events[self.events.type == 'FFR_REC_WORD'].item_name == event.item_name):
            event.recalled = True

        return event

    def event_rec_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'PRACTICE_REC_START' if self.practice else 'REC_START'
        event.item_name = self.current_word
        event.item_num = self.current_num
        event.serialpos = self._serialpos

        return event

    def event_rec_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'PRACTICE_REC_STOP' if self.practice else 'REC_STOP'
        event.item_name = self.current_word
        event.item_num = self.current_num
        event.serialpos = self._serialpos
        event.too_fast_msg = evdata['data']['too_fast']

        return event

    def event_ffr_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'FFR_START'
        return event

    def event_ffr_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'FFR_STOP'
        return event

    def event_break_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_START'
        return event

    def event_break_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_STOP'
        return event

    ###############
    #
    # EVENT MODIFIER FUNCTIONS
    #
    ###############

    def modify_pres_dur(self, events):
        # Determine how long the most recent word was on the screen
        pres_dur = events[-1].mstime - events[-2].mstime
        # Mark word off event with the presentation duration of that word
        events[-1].pres_dur = pres_dur
        # Mark word on event with the presentation duration of that word
        events[-2].pres_dur = pres_dur
        return events

    def modify_rec(self, events):

        # Access the REC_START event for this recall period to determine the timestamp when the recording began
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        # Get list of recalls from the .ann file for the current word, formatted as (rectime, item_num, item_name)
        # The first ten items are practice items, and their .ann files are marked as 0_practice through 9_practice
        ann_outputs = self._parse_ann_file(str(self._serialpos - 1) + '_practice') if self.practice \
            else self._parse_ann_file(str(self._serialpos - 1))

        # For each word in the annotation file (note: there should typically only be one word per .ann in VFFR)
        for recall in ann_outputs:

            # Get the vocalized word from the annotation
            word = recall[-1]

            # Create a new event for the recall
            new_event = self._empty_event
            evtype = 'REC_WORD_VV' if word == '<>' or word == 'V' or word == '!' else 'REC_WORD'
            if self.practice:
                evtype = 'PRACTICE_' + evtype
            new_event.type = evtype
            # Get onset time of vocalized word, both relative to the start of the recording, as well as in Unix time
            new_event.rectime = int(round(float(recall[0])))
            new_event.mstime = rec_start_time + new_event.rectime

            # Mark recall as being too early if vocalization began less than 1 second after the word left the screen
            new_event.too_fast = new_event.rectime < 1000

            # Get the vocalized word from the annotation and mark if it is not the most recently presented word
            new_event.item_name = word
            new_event.intrusion = word != self.current_word

            # Determine the serial position of the spoken word within the 576 item list
            pres_mask = (events.item_name == word) & (events.type == 'WORD')
            # Correct recall if word was previously presented
            if pres_mask.sum() == 1:
                new_event.serialpos = events[pres_mask].serialpos[0]
                new_event.item_num = events[pres_mask].item_num[0]
                events.recalled[pres_mask] = True
            # ELI if word was never presented
            elif pres_mask.sum() == 0:
                new_event.intrusion = True
            # If a word was presented multiple times, abort event creation, as this indicates an error with the session
            else:
                raise Exception('Word %s was presented multiple times during %s %s! Aborting event creation...' %
                                (word, self._subject, self._session))

            # Append the new event
            events = np.append(events, new_event).view(np.recarray)

        return events

    def modify_rec_stop(self, events):
        # Propagate information about whether the participant vocalized too quickly to all events related to that word
        # Meanwhile, check whether any of the vocalized words were the correct word. If so, mark the trial as correct.
        correct_trial = 0
        any_fast = False
        i = -2
        while events[i].type not in ('WORD_OFF', 'PRACTICE_WORD_OFF'):

            # If any vocalizations were too early, we will mark the word presentation and recall start/stop as too fast
            if events[i].too_fast:
                any_fast = True

            # Mark the REC_START event with whether the correct word was spoken and whether they vocalized too early
            if events[i].type == 'REC_START':
                events[i].correct = correct_trial
                events[i].too_fast = any_fast
            # If one of the recalls is the correct word, mark that recall as correct
            elif events[i].item_name == self.current_word:
                events[i].correct = 1
                correct_trial = 1

            # Mark all events with whether the "too fast" message was displayed on that trial
            events[i].too_fast_msg = events[-1].too_fast_msg
            i -= 1

        # Mark the word presentation on/off events as okay/too fast and correct/incorrect
        events[i].too_fast_msg = events[-1].too_fast_msg
        events[i].correct = correct_trial
        events[i].too_fast = any_fast
        events[i-1].too_fast_msg = events[-1].too_fast_msg
        events[i-1].correct = correct_trial
        events[i-1].too_fast = any_fast

        # Mark the REC_STOP event as okay/too fast and correct/incorrect
        events[-1].correct = correct_trial
        events[-1].too_fast = any_fast

        return events

    def modify_ffr(self, events):

        # Access the REC_START event for this recall period to determine the timestamp when the recording began
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        print 'Parsing recalls'
        # Get list of recalls from the .ann file for the current word, formatted as (rectime, item_num, item_name)
        ann_outputs = self._parse_ann_file('ffr')

        # For each word in the annotation file (note: there should typically only be one word per .ann in VFFR)
        for recall in ann_outputs:
            print recall

            # Create a new event for the recall
            new_event = self._empty_event

            # Get onset time of vocalized word, both relative to the start of the recording, as well as in Unix time
            new_event.rectime = int(round(float(recall[0])))
            new_event.mstime = rec_start_time + new_event.rectime
            print new_event.rectime
            new_event.item_num = int(recall[1])
            print new_event.item_num
            new_event.item_name = recall[2].strip()
            print new_event.item_name
            if new_event.item_name not in self.wordpool:
                new_event.intrusion = True
            print new_event.intrusion

            # If the word was a vocalization, mark it as such
            new_event.type = 'FFR_REC_WORD_VV' if new_event.item_num in ('<>', 'V', '!') else 'FFR_REC_WORD'

            # Append the new event
            events = np.append(events, new_event).view(np.recarray)

        return events
