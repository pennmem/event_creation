import numpy as np
import pandas as pd
from . import dtypes
from .base_log_parser import BaseUnityLTPLogParser

class RepFRSessionLogParser(BaseUnityLTPLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(RepFRLogParser, self).__init__(protocol, subject, montage, experiment, session, files)
        self._session = -999
        self._trial = 0
        self._serialpos = -999
        self.practice = True
        self.current_word = ''
        self.current_num = -999


        #TODO: error out if not existent
        if("wordpool" in files.keys()):
            with open(files["wordpool"]) as f:
                self.wordpool = f.readlines() 
        else:
            raise Exception("wordpool not found in transferred files")


        self._add_fields(*dtypes.vffr_fields)
        self._add_type_to_new_event(
            session_start=self.event_sess_start,
            countdown=self.event_countdown,  # Pre-trial countdown video
            microphone_test_recording=self.event_sess_start,  # Start of session (microphone test)
            orientation_stimulus=self._event_skip, # skip orientation events
            display_recall_text=self.event_rec_start,  # Start of vocalization period
            end_recall_period=self.event_rec_stop, # End of vocalization period
            word_stimulus=self.event_word_on,  # Start of word presentation
            clear_word_stimulus=self.event_word_off, # End of word presentation
            display_distractor_problem=self.event_disctract_start, # TODO: distractor presented
            distractor_answered=self.event_distract_stop # TODO: distractor answered
        )
        
        self._add_type_to_modify_events(
            display_recall_text=self.modify_recalls,
            session_start=self.modify_session
        )

    ###############
    #
    # EVENT CREATION FUNCTIONS
    #
    ###############

    def event_sess_start(self, evdata):
        self._session = evdata['data']['session']
        event = self.event_default(evdata)
        return event

    def event_countdown(self, evdata):
        event = self.event_default(evdata)
        event.type = 'COUNTDOWN'
        
        # reset serial position of presentation at start of list
        self._serialpos = 0

        return event

    def event_word_on(self, evdata):
        # Read the word that was presented
        self.current_word = evdata['data']['displayed text'].strip()

        # Build event
        event = self.event_default(evdata)
        event.type = "WORD"
        event.item_name = self.current_word
        event.serialpos = self._serialpos

        event.item_num = self.wordpool.index(event.item_name)

        return event

    def event_word_off(self, evdata):
        event = self.event_default(evdata)
        event.type = "WORD_OFF"
        event.serialpos = self._serialpos

        # increment serial position after word is processed 
        # to order words correctly
        self._serialpos += 1

        return event

    def event_rec_start(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_START"
        return event

    def event_rec_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_STOP"

        # increment list index at end of each list
        self._trial += 1

        return event

    def event_disctract_start(self, evdata):
        event = self.event_default(evdata)
        event.distractor = evdata['data']['problem']

    def event_distract_stop(self, evdata):
        event = self.event_default(evdata)
        event.distractor = evdata['data']['problem']
        event.distractor_answer = evdata['data']['answer']
        event.answer_correct = evdata['data']['correctness']


    ###############
    #
    # EVENT MODIFIER FUNCTIONS
    #
    ###############

    def modify_session(self, events):
        """
        Applies session number to all previous events.
        """
        events.session = self._session
        return events

    def modify_recalls(self, events):
        # Access the REC_START event for this recall period to determine the timestamp when the recording began
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        # Get list of recalls from the .ann file for the current list, formatted as (rectime, item_num, item_name)
        ann_outputs = self._parse_ann_file(str(self._trial))

        for recall in ann_outputs:

            # Get the vocalized word from the annotation
            word = recall[-1]
            new_event = self._empty_event
            new_event.trial = self._trial
            new_event.session = self._session
            new_event.rectime = int(round(float(recall[0])))
            new_event.mstime = rec_start_time + new_event.rectime
            new_event.msoffset = 20
            new_event.item_name = word
            new_event.item_num = int(recall[1])

            # Create a new event for the recall
            evtype = 'REC_WORD_VV' if word == '<>' else 'REC_WORD'
            new_event.type = evtype

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
                    # Correct recall if word is from the most recent list
                    if new_event.intrusion == 0:
                        # Retroactively log on the word pres event that the word was recalled
                        if not any(events[pres_mask].recalled):
                            events.recalled[pres_mask] = True
                    else:
                        events.intruded[pres_mask] = new_event.intrusion
                else:  # XLI from later list
                    new_event.intrusion = -1

            # Append the new event
            events = np.append(events, new_event).view(np.recarray)

        return events

    def modify_repeats(self, events):
        for w in events[events["type"] == "WORD"]["item_name"].unique():
            events[(events["type"] == "WORD") & (events["item_name"] == w)].repeats = events[(events["item_name"] == w) & (events["type"] == "WORD")].count()
        
        return events

    @staticmethod
    def find_presentation(item_num, events):
        events = events.view(np.recarray)
        return np.logical_and(events.item_num == item_num, events.type == 'WORD')
