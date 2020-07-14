import numpy as np
import pandas as pd
from . import dtypes
from .base_log_parser import BaseUnityLTPLogParser

class CourierSessionLogParser(BaseUnityLTPLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(CourierSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files)
        self._session = -999
        self._trial = 0
        self._serialpos = -999

        self.practice = False
        self.current_num = -999


        if("wordpool" in files.keys()):
            with open(files["wordpool"]) as f:
                self.wordpool = [line.rstrip().encode('utf-8') for line in f]
        else:
            raise Exception("wordpool not found in transferred files")

        self._add_fields(*dtypes.courier_fields)

        # TODO
        self._add_type_to_new_event()


    def _new_rec_event(self, recall, rec_start):
        word = recall[-1]
        new_event = self._empty_event
        new_event.trial = self._trial
        new_event.session = self._session
        new_event.rectime = int(round(float(recall[0])))
        new_event.mstime = rec_start + new_event.rectime
        new_event.msoffset = 20
        new_event.item_name = word
        new_event.item_num = int(recall[1])

        return new_event

    def _identify_intrusion(self, events, new_event):
        words = events[events['type'] == 'WORD']
        word_event = words[(words['item'] == item)]

        if len(word_event.index):
            raise Exception("Repeat items not supported or expected. Please check your data.")
        elif word_event["trial"] == self.trial:
            new_event.intrusion = 0
            new_event.serialpos = words[words['item'] == item]['serialPos'][0]
            new_event.store  = words[words['item'] == item]['store'][0]
            new_event.storeX = words[words['item'] == item]['storeX'][0]
            new_event.storeZ = words[words['item'] == item]['storeZ'][0]

        elif len(word_event.index)>0: # PLI
            new_event.intrusion = self.trial - word_event["trial"][0]
            new_event.serialpos = words[words['item'] == item]['serialPos'][0]
            new_event.store  = words[words['item'] == item]['store'][0]
            new_event.storeX = words[words['item'] == item]['storeX'][0]
            new_event.storeZ = words[words['item'] == item]['storeZ'][0]

        else: #ELI
            new_event.intrusion  = -1

        return new_event

    ####################
    # Functions to add new events from a single line in the log
    ####################

    def add_player_transform(self, evdata):
        event = self.event_default(evdata)
        event.type = "XFORM"
        event.presX = evdata['positionX']
        event.presY = evdata['positionY']

        return event

    def add_object_presentation_begins(self, evdata):
        event = self.event_default(evdata)
        event.type = "WORD"
        event.serialpos = evdata["serial position"]
        event.store = '_'.join(data['store name'].split(' ')

        event.storeX = evdata['store positions'][0]
        event.storeY = evdata['store positions'][2]

        return event

    def add_pointing_begins(self, evdata):
        event = self.event_default(evdata)
        event.type = "pointing begins"
        return event

    def add_pointing_finished(self, evdata):
        event = self.event_default(evdata):
        event.type = "pointing finished"
        event.correct_direction = evdata['correct direction (degrees)']
        event.pointed_direction = evdata['pointed direction (degrees)']
        return event

    def add_store_mappings(self, evdata):
        event = self.event_default(evdata)
        event.type = "store mappings"
        event.mappings = {"from {k}".format(k=k): {"to store": , "storeX": , "storeY": } for k in STORES}
        return event 

    def add_familiarization_store_displayed(self, evdata):
        event = self.event_default(evdata)
        event.type = "STORE_FAM"
        event.store_name = "_".join(evdata['store name'])
        return event

    def add_cued_recall_recording_start(self, evdata):
        event = self.event_default(evdata)
        event.storeX = evdata['store position'][0]
        event.storeZ = evdata['store position'][2]
        event.type = 'CUED_REC_CUE'
        event.item = evdata['item'].lower().rstrip('.1')
        event.store_name = '_'.join(evdata['store'].split(' ')
        return event 

    def add_final_object_recall_recording_start(self, evdata):
        event = self.event_default(evdata)
        event.type = "FFR_START"
        return event
    
    def add_final_store_recall_recording_start(self, evdata):
        event = self.event_default(evdata)
        event.type = "SR_START"
        return event

    def add_object_recall_recording_start(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_START"
        return event

    def add_final_object_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "FFR_STOP"
        return event
    
    def add_final_store_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "SR_STOP"
        return event

    def object_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_STOP"
        return event


    ####################
    # Functions that modify the existing event structure
    # when a line in the log is encountered
    ####################

    def modify_rec_start(self, events):

        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        ann_outputs = self._parse_ann_file(str(self._trial))

        for recall in ann_outputs:
            new_event = _new_rec_event(recall, rec_start_time)

            # Create a new event for the recall
            evtype = 'REC_WORD_VV' if word == '<>' else 'REC_WORD'
            new_event.type = evtype
            new_event = _identify_intrusion(events, new_event)

            events = np.append(events, new_event).view(np.recarray) 
        return events

    def modify_cued_rec(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        store_name = rec_start_events["store"]

        ann_outputs = self._parse_ann_file(store_name + "-" + str(self._trial))

        for recall in ann_outputs:
            new_event = _new_rec_event(recall, rec_start_time)

            events = np.append(events, new_event).view(np.recarray) 
        return events

    def modify_store_recall(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        ann_outputs = self._parse_ann_file("store recall")

        for recall in ann_outputs:
            new_event = _new_rec_event(recall, rec_start_time)

            events = np.append(events, new_event).view(np.recarray) 
        return events

    def modify_free_recall(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        ann_outputs = self._parse_ann_file("final recall")

        words = events.query("type == 'WORD'")

        for recall in ann_outputs:
            new_event = _new_rec_event(recall, rec_start_time)

            # Create a new event for the recall
            evtype = 'FFR_REC_WORD_VV' if word == '<>' else 'FFR_REC_WORD'
            new_event.type = evtype
            new_event = _identify_intrusion(events, new_event)

            events = np.append(events, new_event).view(np.recarray) 
        return events

    def modify_proceed_to_next_day(self, events):
        self._trial += 1
        return events

    def modify_intruded(self, events):
        pass


