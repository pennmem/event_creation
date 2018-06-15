import numpy as np
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

        self._add_fields(*dtypes.vffr_fields)
        self._add_type_to_new_event(
            final_recall_start=self.event_ffr_start,  # Start of final free recall
            final_recall_stop=self.event_ffr_stop,  # End of final free recall
            recall_start=self.event_rec_start,  # Start of vocalization period
            recall_stop=self.event_rec_stop,  # End of vocalization period
            optional_break_start=self.event_break_start,  # Start of manual break
            optional_break_stop=self.event_break_stop,  # End of manual break
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

    def event_word_on(self, evdata):
        # Get information on the word presentation and whether it is a practice item
        self._serialpos = evdata['data']['index'] + 1
        self.practice = evdata['data']['practice']
        self.current_word = evdata['data']['word']
        # Build event
        event = self.event_default(evdata)
        event.type = 'PRACTICE_WORD' if self.practice else 'WORD'
        event.item_name = self.current_word
        event.serialpos = self._serialpos
        if 'ltp word number' in evdata['data']:
            self.current_num = evdata['data']['ltp word number']
            event.item_num = self.current_num
        return event

    def event_word_off(self, evdata):
        event = self.event_default(evdata)
        event.type = 'PRACTICE_WORD_OFF' if self.practice else 'WORD_OFF'
        event.item_name = evdata['data']['word']
        event.serialpos = self._serialpos
        if event.item_name == self.current_word:
            event.item_num = self.current_num
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
        event.too_fast = evdata['data']['too_fast']

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
            new_event.too_early = new_event.rectime < 1000

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

    @staticmethod
    def modify_rec_stop(events):
        # Propagate information about whether the participant vocalized too quickly to all events related to that word
        i = -2
        while events[i].type not in ('WORD', 'PRACTICE_WORD'):
            events[i].too_fast = events[-1].too_fast
            i -= 1
        events[i].too_fast = events[-1].too_fast
        return events

    def modify_ffr(self, events):

        # Access the REC_START event for this recall period to determine the timestamp when the recording began
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        # Get list of recalls from the .ann file for the current word, formatted as (rectime, item_num, item_name)
        ann_outputs = self._parse_ann_file('ffr')

        # For each word in the annotation file (note: there should typically only be one word per .ann in VFFR)
        for recall in ann_outputs:

            # Get the recalled word from the annotation
            word = recall[-1]

            # Create a new event for the recall
            new_event = self._empty_event
            new_event.type = 'FFR_REC_WORD_VV' if word == '<>' or word == 'V' or word == '!' else 'FFR_REC_WORD'

            # Get onset time of vocalized word, both relative to the start of the recording, as well as in Unix time
            new_event.rectime = int(round(float(recall[0])))
            new_event.mstime = rec_start_time + new_event.rectime

            # Get the recalled word
            new_event.item_name = word

            # Determine where the item was presented
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
