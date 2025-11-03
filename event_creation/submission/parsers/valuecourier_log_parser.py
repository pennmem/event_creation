import numpy as np
import pandas as pd
import six
from . import dtypes
from .courier_log_parser import CourierSessionLogParser



class ValueCourierSessionLogParser(CourierSessionLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        super().__init__(protocol, subject, montage, experiment, session, files)
        pd.set_option('display.max_columns', None)
        self.phase = '1'

        self._add_fields(*dtypes.vc_fields)


        self._add_type_to_new_event(
           start_movie=self.event_movie_start,
           stop_movie=self.event_movie_stop,# Sherlock videos
           start_video=self.event_video_start,#music videos (numbered)
           stop_video=self.event_video_stop,
           start_music_videos=self.event_music_videos_start,
           stop_music_videos=self.event_music_videos_stop,
           start_music_video_recall=self.event_start_music_videos_recall,
           stop_music_video_recall=self.event_stop_music_videos_recall,
           music_video_recall_recording_start=self.event_video_rec_start,
           music_video_recall_recording_stop=self.event_video_rec_stop,
           start_town_learning=self.event_town_learning_start,
           stop_town_learning=self.event_town_learning_end,
           start_practice_deliveries=self.event_practice_start,
           stop_practice_deliveries=self.event_practice_end,
           keypress=self.event_efr_mark,
            value_recall = self.add_value_recall,
           continuous_pointer=self.event_pointer_on,
           start_required_break=self.event_break_start,
           stop_required_break=self.event_break_stop,
           start_deliveries=self.event_trial_start,
           stop_deliveries=self.event_trial_end,
           final_compensation=self.add_receive_compensation,
           earned_tips=self.event_sess_end,
        )
        self._add_type_to_modify_events(
           stop_deliveries=self.modify_pointer_on,
              final_compensation=self.modify_after_final_compensation,
        )

    ####################
    # Functions to add new events from a single line in the log
    ####################
    def event_break_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_START'
        return event

    def event_sess_end(self, evdata):
        event = self.event_default(evdata)
        event.type = 'SESS_END'
        return event

    def event_break_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_END'
        return event

    def event_trial_start(self, evdata):
        self.practice = False
        self._trial = evdata['data']['trial number']
        event = self.event_default(evdata)
        event.type = 'TRIAL_START'
        return event

    def event_trial_end(self, evdata):
        event = self.event_default(evdata)
        event.type = 'TRIAL_END'
        return event

    def event_movie_start(self, evdata):
        self.phase = 'movie'
        event = self.event_default(evdata)
        event.type = 'MOVIE_START'
        return event

    def event_movie_stop(self, evdata):
        self.phase = '2'
        event = self.event_default(evdata)
        event.type = 'MOVIE_STOP'
        return event

    def event_music_videos_start(self, evdata):
        return evdata

    def event_music_videos_stop(self, evdata):
        return evdata


    def event_video_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'VIDEO_START'
        event.itemno = evdata["data"]["video number"]
        return event

    def event_video_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'VIDEO_STOP'
        event.itemno = evdata["data"]["video number"]
        return event

    def event_start_music_videos_recall(self, evdata):
        return evdata


    def event_stop_music_videos_recall(self, evdata):
        return evdata

    def event_video_rec_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'VIDEO_REC_START'
        event.itemno = evdata["data"]["video number"]
        return event

    def event_video_rec_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'VIDEO_REC_STOP'
        event.itemno = evdata["data"]["video number"]
        return event

    def event_town_learning_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'TL_START'
        return event

    def event_town_learning_end(self, evdata):
        event = self.event_default(evdata)
        event.type = 'TL_END'
        return event

    def event_practice_start(self, evdata):
        self.practice = True
        self.trial = 0
        event = self.event_default(evdata)
        event.type = 'PRACTICE_DELIVERY_START'
        return event

    def event_practice_end(self, evdata):
        self.practice = True
        event = self.event_default(evdata)
        event.type = 'PRACTICE_DELIVERY_END'
        return event

    def add_value_recall(self, evdata):
        event = self.event_default(evdata)
        event.type = "VALUE_RECALL" if not self.practice else "PRACTICE_VALUE_RECALL"
        event.trial = evdata['data']['trial number']
        value_recall = self.stringify_list(evdata['data']['typed response'])
        event.valuerecall = int(value_recall)
        if 'actual value' in evdata['data']:
            event.actualvalue = evdata['data']['actual value']
        else:
            event.actualvalue = -1
            print(
                f"Missing 'actual value' field in VALUE_RECALL event for subject " +
                f"{self._subject}, session {self._session}, trial {event.trial}"
            )
        return event

    def event_pointer_on(self, evdata):
        event = self.event_default(evdata)
        event.type = 'POINTER_ON'
        return event

    def event_efr_mark(self, evdata):
        event = self.event_default(evdata)
        event.type = 'EFR_MARK'
        event.efr_mark = evdata['data']['response']=='correct'
        return event

    def add_receive_compensation(self, evdata):
        event = self.event_default(evdata)
        event.type = 'FINAL_COMPENSATION'
        event.compensation = evdata['data']['compensation']
        event.multiplier = evdata['data']['multiplier']
        return event

    def stringify_list(self, input_val):
        """
        Convert a list of characters or other types into a single string.
        If already a string, returns unchanged. Otherwise casts to string.
        """
        if isinstance(input_val, list):
            return ''.join(map(str, input_val))
        elif not isinstance(input_val, str):
            return str(input_val)
        else:
            return input_val

    # overwrite normal courier object presentation to add fields
    def add_object_presentation_begins(self, evdata):
        self._trial = evdata['data']['trial number']
        event = self.event_default(evdata)
        event.type = "WORD" if not self.practice else "PRACTICE_WORD"
        event.serialpos = evdata['data']["serial position"]

        store_name = self.stringify_list(evdata['data']['store name'])
        event.store = '_'.join(store_name.split(' '))

        event.intruded = 0
        event.recalled = 0

        # Convert store position to string
        store_pos_str = self.stringify_list(evdata['data']['store position'])

        # Parse string into floats
        store_position = [float(p) for p in store_pos_str.strip('()').replace(' ', '').split(',')]

        event.storeX = store_position[0] if len(store_position) > 0 else None
        event.storeZ = store_position[2] if len(store_position) > 2 else None

        player_pos_str = self.stringify_list(evdata['data']['player position'])
        player_position = [float(p) for p in player_pos_str.strip('()').replace(' ', '').split(',')]
        event.presX = player_position[0] if len(player_position) > 0 else None
        event.presZ = player_position[2] if len(player_position) > 2 else None

        if 'player orientation' in evdata['data']:
            player_rot_str = self.stringify_list(evdata['data']['player orientation'])
            player_orientation = [float(o) for o in player_rot_str.strip('()').replace(' ', '').split(',')]

            event.playerrotY = player_orientation[1] if len(player_orientation) > 1 else None
        
        event.primacybuf = evdata['data']['primacy buffer']
        event.recencybuf = evdata['data']['recency buffer']
        event.itemvalue = evdata['data']['store value']

        event.numingroupchosen = evdata['data']['number of in group chosen']

        item_name = self.stringify_list(evdata['data']['item name'])
        event["item"] = item_name.upper().rstrip('.1')
        if 'store point type' in evdata['data']:
            event.storepointtype = self.stringify_list(evdata['data']['store point type'])

        return event

# Overwrite normal Courier FFR and store recall
    def modify_store_recall(self, events): 
        return evdata

    # overwrite
    def modify_free_recall(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        try:
            ann_outputs = self._parse_ann_file("final recall")
        except:
            ann_outputs = self._parse_ann_file("final free-0")
            ann_outputs = ann_outputs + self._parse_ann_file("final free-1")
        words = events[events["type"] == 'WORD']

        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)

            # Create a new event for the recall
            evtype = 'REC_WORD_VV' if "<>" in new_event["item"] else 'REC_WORD'
            new_event.type = evtype
            new_event = self._identify_intrusion(events, new_event)
            new_event.trial = -999 # to match old events

            events = np.append(events, new_event).view(np.recarray)

        return events

    def modify_pointer_on(self, events):
        full_evs = pd.DataFrame.from_records(events)
        part_idx = full_evs[(full_evs.type=='WORD')|(full_evs.type=='TL_END')].index.values
        it = np.nditer(part_idx)
        preserve_idx = []
        last = int(it.value)
        while it.iternext():
            subset = full_evs[last:int(it.value)].type.eq('POINTER_ON')
            if subset.sum()>0:
                preserve_idx.append(subset.idxmax())
            last = int(it.value)
        point_idx = full_evs[full_evs.type=='POINTER_ON'].index.values
        clipped_evs = full_evs.drop(index=[i for i in point_idx if i not in preserve_idx])
        return clipped_evs.to_records(index=False,
                column_dtypes={x:str(y[0]) for x,y in events.dtype.fields.items()})

    # copies over every contextual value to all events after final compensation, we also merge on trials to hand over value recall, actuall value and storepointtype
    def modify_after_final_compensation(self, events):
        # Convert to DataFrame
        full_evs = pd.DataFrame.from_records(events)

        # merge on trials to hand over value recall, actual value and storepointtype
        value_recalls = full_evs[full_evs.type == "VALUE_RECALL"] 
        words = full_evs[full_evs.type == "WORD"]
        rec_words = full_evs[full_evs.type == "REC_WORD"]
        rec_vv_words = full_evs[full_evs.type == "REC_WORD_VV"]

        # WORD --> storepointtype --> VALUE_RECALL, REC_WORD, REC_WORD_VV
        word_trial_to_storepointtype = words.set_index("trial")["storepointtype"].to_dict()
        for event_type in ["VALUE_RECALL", "REC_WORD", "REC_WORD_VV"]:
            subset = full_evs[full_evs.type == event_type]
            for idx, row in subset.iterrows():
                trial = row["trial"]
                if trial in word_trial_to_storepointtype:
                    full_evs.at[idx, "storepointtype"] = word_trial_to_storepointtype[trial]

        # VALUE_RECALL --> actualvalue, valuerecall --> WORD, `REC_WORD`, REC_WORD_VV
        valuerecall_trial_to_actualvalue = value_recalls.set_index("trial")["actualvalue"].to_dict()
        valuerecall_trial_to_valuerecall = value_recalls.set_index("trial")["valuerecall"].to_dict()

        # --- Apply to multi-row event types ---
        for event_type in ["WORD", "REC_WORD", "REC_WORD_VV"]:
            subset = full_evs[full_evs.type == event_type]
            for idx, row in subset.iterrows():
                trial = row["trial"]

                # actualvalue
                if trial in valuerecall_trial_to_actualvalue:
                    full_evs.at[idx, "actualvalue"] = valuerecall_trial_to_actualvalue[trial]

                # valuerecall
                if trial in valuerecall_trial_to_valuerecall:
                    full_evs.at[idx, "valuerecall"] = valuerecall_trial_to_valuerecall[trial]


        # copy over universal values
        final_comp_ev = full_evs[full_evs.type == "FINAL_COMPENSATION"]
        mult = final_comp_ev["multiplier"].values
        comp = final_comp_ev["compensation"].values

        
        full_evs["multiplier"] = mult[0]
        full_evs["compensation"] = comp[0]

        word_ev = full_evs[full_evs.type == "WORD"]
        pbuf = word_ev["primacybuf"].values
        rbuf = word_ev["recencybuf"].values
        numingroup = word_ev["numingroupchosen"].values
        
        full_evs["primacybuf"] = pbuf[0]
        full_evs["recencybuf"] = rbuf[0]
        full_evs["numingroupchosen"] = numingroup[0]
        return full_evs.to_records(index=False,
                column_dtypes={x:str(y[0]) for x,y in events.dtype.fields.items()})

    #overwrite
    def add_pointing_finished(self, evdata):
        return evdata

    def event_classifier_wait(self, evdata):
        return evdata

    def event_classifier_result(self, evdata):
        return evdata

