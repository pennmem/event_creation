import numpy as np
import pandas as pd
from . import dtypes
from .base_log_parser import BaseUnityLogParser


#TODO: update default to use list rather than trial

class RepFRSessionLogParser(BaseUnityLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(RepFRSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files)

        self._session = -999
        self._trial = 0
        self._serialpos = -999
        self.practice = True
        self.current_num = -999
        self.protocol = protocol

        if("wordpool" in list(files.keys())):
            with open(files["wordpool"]) as f:
                self.wordpool = [line.rstrip() for line in f]

        else:
            raise Exception("wordpool not found in transferred files")

        if protocol=='ltp':
            self._add_fields(*dtypes.ltp_fields)
            self.add_type_to_new_event(participant_break=self.event_break_start) # Mid session break
        self._add_fields(*dtypes.repFR_fields)
        self._add_type_to_new_event(
            session_start=self.event_sess_start,
            start_trial=self.event_trial_start,
            countdown=self.event_countdown,  # Pre-trial countdown video
            orientation_stimulus=self._event_skip, # skip orientation events
            display_recall_text=self.event_rec_start,  # Start of vocalization period
            end_recall_period=self.event_rec_stop, # End of vocalization period
            word_stimulus=self.event_word_on,  # Start of word presentation
            clear_word_stimulus=self.event_word_off, # End of word presentation
            session_end=self.event_sess_end,
        )
        
        self._add_type_to_modify_events(
            display_recall_text=self.modify_recalls,

            session_start=self.modify_session,
        )

    ###############
    #
    # EVENT CREATION FUNCTIONS
    #
    ###############

    def event_sess_start(self, evdata):
        self._session = evdata['data']['session']
        event = self.event_default(evdata)
        event["list"] = self._trial
        event.type = 'SESS_START'
        return event
    
    # added SESS_START and SESS_END to satisfy lcf partitions
    def event_sess_end(self, evdata):
        # FIXME: no session # attribute in embedded data
        # like there is for session start above 
        event = self.event_default(evdata)
        event.type = 'SESS_END'
        return event

    def event_trial_start(self, evdata):
        event = self.event_default(evdata)
        self._trial = evdata['data']['trial']
        event.type = 'TRIAL_START'
        return event

    def event_countdown(self, evdata):
        event = self.event_default(evdata)
        event.type = 'COUNTDOWN'
        
        # reset serial position of presentation at start of list
        self._serialpos = 0

        return event
    
    def event_break_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_START'
        return event

    def event_word_on(self, evdata):

        # Build event
        event = self.event_default(evdata)
        event.type = "WORD"
        
        event.item_name = evdata['data']['displayed text'].strip()
        # FIXME: someone made a redundant "word stimulus" instead of 
        # "word stimulus info". This will make duplicate events. Not my (JR) fault. 
        event.serialpos = self._serialpos
        event["list"] = self._trial
        
        #FIXME: RepFR wordpool has "word" as the first item. Is this ok? 
        event.item_num = self.wordpool.index(str(event.item_name)) + 1

        return event

    def event_word_off(self, evdata):
        event = self.event_default(evdata)
        event.type = "WORD_OFF"
        event.serialpos = self._serialpos
        event["list"] = self._trial


        # increment serial position after word is processed 
        # to order words correctly
        self._serialpos += 1

        return event

    def event_rec_start(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_START"
        event["list"] = self._trial

        return event

    def event_rec_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_END"
        event["list"] = self._trial


        # increment list index at end of each list
        self._trial += 1

        return event

    def event_distract_start(self, evdata):
        event = self.event_default(evdata)
        event["list"] = self._trial

        # process numbers out of problem
        distractor = evdata['data']['displayed text']
        nums = [int(s) for s in distractor.split('=')[0].split() if s.isdigit()]
        event.distractor = nums

        return event

    def event_distract_stop(self, evdata):
        event = self.event_default(evdata)
        event["list"] = self._trial
        distractor = evdata['data']['problem']
        nums = [int(s) for s in distractor.split('=')[0].split() if s.isdigit()]
        event.distractor = nums
        event.distractor_answer = evdata['data']['answer'] if evdata['data']['answer'] != '' else -999
        event.answer_correct = evdata['data']['correctness']

        return event


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
        try:
            ann_outputs = self._parse_ann_file(str(self._trial))
        # final free recall is not names with integer 26, but internally is 
        # considered trial 26
        except:
            if self.protocol=='ltp':
                ann_outputs = self._parse_ann_file("ffr")
            else:
                # could happen if the session was not finished
                ann_outputs = []
                

        for recall in ann_outputs:

            # Get the vocalized word from the annotation
            word = recall[-1]
            new_event = self._empty_event

            new_event["list"] = self._trial
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
                pres_mask = (events.item_num == new_event.item_num) & (events.type == 'WORD')
                pres_trial = np.unique(events["list"][pres_mask])


                # Correct recall or PLI
                if len(pres_trial) == 1:
                    # Determines how many lists back the recalled word was presented

                    num_lists_back = self._trial - pres_trial[0]
                    # Retrieve the recalled word's serial position in its list, as well as the distractor(s) used
                    # Correct recall if word is from the most recent list
                    if num_lists_back == 0:
                        new_event.intrusion = 0

                        # Retroactively log on the word pres event that the word was recalled
                        if not any(events.recalled[pres_mask]):
                            events.recalled[pres_mask] = True
                    else:
                        new_event.intrusion = num_lists_back
                        events.intruded[pres_mask] = num_lists_back

                else:  # XLI from later list
                    new_event.intrusion = -1

            # Append the new event
            events = np.append(events, new_event).view(np.recarray)
            events = self.modify_repeats(events)


        return events

    def modify_repeats(self, events):
        # FIXME: this could be a lot more readable
        repeat_mask = [True if ev.item_num in events[:i][events[:i]["type"] =="WORD"].item_num else False for i, ev in enumerate(events)]
        events.is_repeat[repeat_mask] = True

        for w in np.unique(events[events["type"] == "WORD"]["item_name"]):
            events.repeats[((events["type"] == "WORD") | (events["type"] == "REC_WORD")) & (events["item_name"] == w)] = len(events[(events["item_name"] == w) & (events["type"] == "WORD")])


        return events

    @staticmethod
    def find_presentation(item_num, events):
        events = events.view(np.recarray)
        return np.logical_and(events.item_num == item_num, events.type == 'WORD')
