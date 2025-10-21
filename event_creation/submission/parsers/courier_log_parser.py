import numpy as np
import pandas as pd
from . import dtypes
from .base_log_parser import BaseUnityLogParser
import six

class CourierSessionLogParser(BaseUnityLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(CourierSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files)
        self._trial = 0
        self.subject = subject

        self.phase = ''
        self.practice = False
        self.current_num = -999

        self.storeX = -999
        self.storeZ = -999
        self.presX = -999
        self.presZ = -999

        self.STORES = ['gym', 'pet store', 'barber shop', 'florist', 'craft shop', 'jewelry store', 'grocery store', 'music store', 'cafe', 'pharmacy', 'clothing store', 'pizzeria', 'dentist', 'toy store', 'hardware store', 'bakery', 'bike shop']


        if("wordpool" in list(files.keys())):
            with open(files["wordpool"]) as f:
                self.wordpool = [line.rstrip().encode('utf-8') for line in f]
        else:
            raise Exception("wordpool not found in transferred files")

        self._add_fields(*dtypes.courier_fields)
        if protocol=='ltp':
            self._add_fields(*dtypes.ltp_fields)

        self._add_type_to_new_event(
         versions=self.add_experiment_version,
         #instruction_message_cleared=self.event_sess_start,
         familiarization_store_displayed=self.add_familiarization_store_displayed,
         proceed_to_next_day_prompt=self.add_proceed_to_next_day,
         store_mappings=self.add_store_mappings,
         pointing_begins=self.add_pointing_begins,
         Player_transform=self.add_player_transform,
         player_transform=self.add_player_transform,
         PlayerTransform=self.add_player_transform,
         playertransform=self.add_player_transform,
         Player_Transform=self.add_player_transform,
         pointing_finished=self.add_pointing_finished,
         object_presentation_begins=self.add_object_presentation_begins,
         cued_recall_recording_start=self.add_cued_recall_recording_start,
         object_recall_recording_start=self.add_object_recall_recording_start,
         final_store_recall_recording_start=self.add_final_store_recall_recording_start,
         final_object_recall_recording_start=self.add_final_object_recall_recording_start,
         cued_recall_recording_stop=self.add_cued_recall_recording_stop,
         object_recall_recording_stop=self.add_object_recall_recording_stop,
         final_store_recall_recording_stop=self.add_final_store_recall_recording_stop,
         final_object_recall_recording_stop=self.add_final_object_recall_recording_stop,
         end_text=self.event_sess_end,
        )

        self._add_type_to_modify_events(
           final_object_recall_recording_start=self.modify_free_recall,
           cued_recall_recording_start=self.modify_cued_rec,
           final_store_recall_recording_start=self.modify_store_recall,
           object_recall_recording_start=self.modify_rec_start,
        )

    def parse(self):
        events = super(CourierSessionLogParser, self).parse()

        if self.old_syncs:
            events.mstime = events.mstime + 500

        return events


    def event_default(self, evdata):
        event = super(CourierSessionLogParser, self).event_default(evdata)
        event.trial = self._trial
        event.session = self._session
        event.presX = self.presX
        event.presZ = self.presZ
        event.phase = 'practice' if self.practice else self.phase
        return event


    def _new_rec_event(self, recall, evdata):
        word = recall[-1].strip().rstrip(".1") # some annotations have a .1 at the end?
        new_event = self._empty_event
        new_event.trial = self._trial
        new_event.session = self._session
        new_event.phase = 'practice' if self.practice else self.phase
        new_event.presX = self.presX
        new_event.presZ = self.presZ
        # new_event.rectime = int(round(float(recall[0])))
        new_event.rectime = int(float(recall[0])) # old code did not round
        new_event.mstime = evdata.mstime + new_event.rectime
        new_event.msoffset = 20
        new_event["item"] = word.upper()
        new_event.itemno = int(recall[1])

        return new_event


    def _identify_intrusion(self, events, new_event):
        words = events[events['type'] == 'WORD']
        word_event = words[(words['item'] == new_event["item"])]

        if len(word_event) > 1:
            raise Exception("Repeat items not supported or expected. Please check your data.")

        elif len(word_event) == 0: #ELI
            new_event.intrusion  = -1

        elif word_event["trial"][0] == self._trial:
            new_event.intrusion = 0
            new_event.serialpos = words[words['item'] == new_event["item"]]['serialpos'][0]
            new_event.store  = words[words['item'] == new_event["item"]]['store'][0]
            new_event.storeX = words[words['item'] == new_event["item"]]['storeX'][0]
            new_event.storeZ = words[words['item'] == new_event["item"]]['storeZ'][0]

        elif word_event["trial"][0] >= 0: # PLI
            new_event.intrusion = self._trial - word_event["trial"][0]
            new_event.serialpos = words[words['item'] == new_event["item"]]['serialpos'][0]
            new_event.store  = words[words['item'] == new_event["item"]]['store'][0]
            new_event.storeX = words[words['item'] == new_event["item"]]['storeX'][0]
            new_event.storeZ = words[words['item'] == new_event["item"]]['storeZ'][0]
        else:
            raise Exception("Processing error, word event was not presented during experimental trial")

        return new_event


    ####################
    # Functions to add new events from a single line in the log
    ####################
    def add_experiment_version(self, evdata):
        # version numbers are either v4.x or v4.x.x depending on the era,
        # due to a lack of foresight. At this point, we only need to check
        # that the version is greater than 4.0
        try:
            minor_version = int(evdata["data"]["Experiment version"].split('.')[1])
        except (ValueError, AttributeError):
            minor_version = int(6)
        if minor_version == 0 \
           and self.subject.startswith('R'):
            self.old_syncs=True
        else:
            self.old_syncs=False
        return False

    def event_break_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_START'
        return event

    def event_break_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_STOP'
        return event

    def event_sess_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'SESS_START'
        return event

    def event_sess_end(self, evdata):
        event = self.event_default(evdata)
        event.type = 'SESS_END'
        return event

    def add_proceed_to_next_day(self, evdata):
        self._trial += 1
        return False

    def add_player_transform(self, evdata):
        self.presX = evdata['data']['positionX']
        self.presZ = evdata['data']['positionZ']
        return False

    def add_object_presentation_begins(self, evdata):
        self._trial = evdata['data']['trial number']
        event = self.event_default(evdata)
        event.type = "WORD" if not self.practice else "PRACTICE_WORD"
        event.serialpos = evdata['data']["serial position"]

        event.store = '_'.join(evdata['data']['store name'].split(' '))
        event.intruded = 0
        event.recalled = 0
        event.finalrecalled = 0

        if isinstance(evdata['data']['store position'], six.string_types):
            evdata['data']['store position'] = [float(p) for p in evdata['data']['store position'][1:-1].split(',')]

        event.storeX = evdata['data']['store position'][0]
        self.storeX = event.storeX

        event.storeZ = evdata['data']['store position'][2]
        self.storeZ = event.storeZ

        event.presX = self.presX
        event.presZ = self.presZ

        event["item"] = evdata['data']['item name'].upper().rstrip('.1')

        return event

    def add_pointing_begins(self, evdata):
        event = self.event_default(evdata)
        event.type = "pointing begins"

        event.storeX = self.storeX
        event.storeZ = self.storeZ
        event.presX = self.presX
        event.presZ = self.presZ

        return event

    def add_pointing_finished(self, evdata):
        event = self.event_default(evdata)
        event.type = "pointing finished"
        event.correctPointingDirection = evdata['data']['correct direction (degrees)']
        event.submittedPointingDirection = evdata['data']['pointed direction (degrees)']

        event.storeX = self.storeX
        event.storeZ = self.storeZ
        event.presX = self.presX
        event.presZ = self.presZ

        return event

    def add_store_mappings(self, evdata):
        event = self.event_default(evdata)
        event.type = "store mappings"

        # Different versions have different store mappings
        try:
            event.mappings = {"from {k}".format(k=k): {"to store": evdata['data'][k], "storeX": evdata['data']["{} position X".format(k)] , "storeZ": evdata['data']["{} position Z".format(k)] } for k in self.STORES}
        except:
            try:
                event.mappings = {"from {k}".format(k=k): {"to store": evdata['data']['_'.join(k.split())], "storeX": evdata['data']["{} position X".format(k)] , "storeZ": evdata['data']["{} position Z".format(k)] } for k in self.STORES}
            except:
                print("mappings is null")
                event.mappings = {}

        return event

    def add_familiarization_store_displayed(self, evdata):
        event = self.event_default(evdata)
        event.type = "STORE_FAM"
        event.store = "_".join(evdata['data']['store name'].lower().split(' '))
        return event

    def add_cued_recall_recording_start(self, evdata):
        event = self.event_default(evdata)

        if isinstance(evdata['data']['store position'], six.string_types):
            evdata['data']['store position'] = [float(p) for p in evdata['data']['store position'][1:-1].split(',')]

        event.storeX = evdata['data']['store position'][0]
        event.storeZ = evdata['data']['store position'][2]
        self.storeX = event.storeX
        self.storeZ = event.storeZ

        event.presX = self.presX
        event.presZ = self.presZ

        event.type = 'CUED_REC_CUE'
        event["item"] = evdata['data']['item'].rstrip('.1').upper()
        event.store = '_'.join(evdata['data']['store'].lower().split(' '))
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

        event.presX = self.presX
        event.presZ = self.presZ

        return event

    def add_final_object_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "FFR_STOP"
        return event

    def add_final_store_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "SR_STOP"
        return event

    def add_object_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_STOP"
        return event

    def add_cued_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "CUED_REC_STOP"
        return event


    ####################
    # Functions that modify the existing event structure
    # when a line in the log is encountered
    ####################

    def modify_rec_start(self, events):

        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        if self.practice:
            # Practice parsing not implemented for Courier / NICLS
            return events
        else:
            ann_outputs = self._parse_ann_file(str(self._trial))
        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)

            # Create a new event for the recall
            evtype = 'REC_WORD_VV' if "<>" in new_event["item"] else 'REC_WORD'
            new_event.type = evtype
            new_event = self._identify_intrusion(events, new_event)

            if new_event.intrusion > 0:
                events.intruded[(events["type"] == 'WORD') & (events["item"] == new_event["item"])] = 1
            elif new_event.intrusion == 0:
                events.recalled[(events["type"] == 'WORD') & (events["item"] == new_event["item"])] = 1
            print("old")
            # print(events)
            print(events.dtype)
            print("new")
            # print(new_event)
            print(new_event.dtype)
            events = np.append(events, new_event).view(np.recarray)

        return events

    def modify_cued_rec(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        store_name = rec_start_event.store

        try:
            ann_outputs = self._parse_ann_file(str(self._trial) + '-' + " ".join(store_name.lower().split('_')))

        except:
            ann_outputs = []
            print(("MISSING ANNOTATIONS FOR %s" % rec_start_event.store))

        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)
            # new_event = self._identify_intrusion(events, new_event)

            evtype = 'CUED_REC_WORD_VV' if "<>" in new_event["item"] else 'CUED_REC_WORD'
            new_event.type = evtype

            # TODO: this matches previous events, but without this CUED_REC events lack any annotation
            # if new_event.intrusion > 0:
            #     events[(events["type"] == 'WORD') & (events["item"] == new_event["item"])].intruded = 1
            # elif new_event.intrusion == 0:
            #     events[(events["type"] == 'WORD') & (events["item"] == new_event["item"])].recalled = 1

            new_event["intrusion"] = -999

            events = np.append(events, new_event).view(np.recarray)

        return events

    def modify_store_recall(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        ann_outputs = self._parse_ann_file("store recall")

        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)
            new_event = self._identify_intrusion(events, new_event)

            evtype = 'SR_REC_WORD_VV' if "<>" in new_event["item"] else 'SR_REC_WORD'
            new_event.type = evtype
            new_event.trial = -999 # to match old events
            new_event.intrusion = 0 if new_event["item"] in ["_".join(s.upper().split()) for s in self.STORES] else -1

            events = np.append(events, new_event).view(np.recarray)

        return events

    def modify_free_recall(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        try:
            ann_outputs = self._parse_ann_file("final recall")
        except:
            ann_outputs = self._parse_ann_file("final free")

        words = events[events["type"] == 'WORD']

        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)

            # Create a new event for the recall
            evtype = 'FFR_REC_WORD_VV' if "<>" in new_event["item"] else 'FFR_REC_WORD'
            new_event.type = evtype
            new_event = self._identify_intrusion(events, new_event)
            new_event.trial = -999 # to match old events

            if new_event.intrusion >= 0:
                new_event.intrusion = 0
                events.finalrecalled[(events["type"] == "WORD") & (events["item"] == new_event["item"])] = 1

            events = np.append(events, new_event).view(np.recarray)

        return events
