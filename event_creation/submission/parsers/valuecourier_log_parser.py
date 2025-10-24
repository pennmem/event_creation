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

        self._add_fields(*dtypes.efr_fields)
        self._add_fields(*dtypes.nicls_fields)

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
        #    Object_Reinstatement = self.object_reinstatement,
            value_recall = self.add_value_recall,
           continuous_pointer=self.event_pointer_on,
           start_required_break=self.event_break_start,
           stop_required_break=self.event_break_stop,
           start_deliveries=self.event_trial_start,
           stop_deliveries=self.event_trial_end,
           receive_compensation=self.add_receive_compensation,
        )
        self._add_type_to_modify_events(
           stop_deliveries=self.modify_pointer_on,
              value_recall=self.modify_word_with_value_recall,
        )

    ####################
    # Functions to add new events from a single line in the log
    ####################
    def event_break_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_START'
        return event

    def event_break_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_STOP'
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
        # self.phase = 'video'
        # event = self.event_default(evdata)
        # event.type = 'MUSIC_VIDEOS_START'
        return evdata

    def event_music_videos_stop(self, evdata):
        # self.phase = '2'
        # event = self.event_default(evdata)
        # event.type = 'MUSIC_VIDEOS_STOP'
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
        # self.phase = 'video'
        # event = self.event_default(evdata)
        # event.type = 'MUSIC_VIDEOS_REC_START'
        return evdata


    def event_stop_music_videos_recall(self, evdata):
        # self.phase = '2'
        # event = self.event_default(evdata)
        # event.type = 'MUSIC_VIDEOS_REC_STOP'
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

    # def object_reinstatement(self, evdata):
    #     self._trial = evdata['data']['trial number']
    #     self._reinstated_item = evdata['data']['previously delivered item seen']
    #     self._angle_difference = evdata['data']['Angle of Approach']
    #     self.presX = evdata['data']['positionX']
    #     self.presZ = evdata['data']['positionZ']
    #     event = self.event_default(evdata)
    #     event.type = "REINSTATEMENT"
    #     event["item"] = evdata['data']['previously delivered item seen'].upper().rstrip('.1')

    #     event.intruded = 0
    #     event.recalled = 0
    #     # event.finalrecalled = 0

    #     return event

    def add_value_recall(self, evdata):
        event = self.event_default(evdata)
        event.type = 'VALUE_RECALL'
        event.trial = evdata['data']['trial number']
        value_recall = self.stringify_list(evdata['data']['typed response'])
        event.value_recall = int(value_recall)
        print(value_recall)
        if 'actual value' in evdata['data']:
            print(evdata['data']['actual value'])
            event.actual_value = evdata['data']['actual value']
            # event.actual_value = -1
        else:
            event.actual_value = -1
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
        event.value = evdata['data']['store value']

        # Convert store position to string
        store_pos_str = self.stringify_list(evdata['data']['store position'])

        # Parse string into floats
        store_position = [float(p) for p in store_pos_str.strip('()').replace(' ', '').split(',')]

        event.storeX = store_position[0] if len(store_position) > 0 else None
        event.storeZ = store_position[2] if len(store_position) > 2 else None

        event.presX = self.presX
        event.presZ = self.presZ

        if 'player orientation' in evdata['data']:
            player_rot_str = self.stringify_list(evdata['data']['player orientation'])
            player_orientation = [float(o) for o in player_rot_str.strip('()').replace(' ', '').split(',')]

            event.playerRotY = player_orientation[1] if len(player_orientation) > 1 else None
            event.playerRotX = player_orientation[0] if len(player_orientation) > 0 else None
            event.playerRotZ = player_orientation[2] if len(player_orientation) > 2 else None
        
        event.primacyBuf = evdata['data']['primacy buffer']
        event.recencyBuf = evdata['data']['recency buffer']

        event.numInGroupChosen = evdata['data']['number of in group chosen']

        item_name = self.stringify_list(evdata['data']['item name'])
        event["item"] = item_name.upper().rstrip('.1')
        if 'store point type' in evdata['data']:
            event.store_point_type = self.stringify_list(evdata['data']['store point type'])

        return event


    def modify_word_with_value_recall(self, events):
    # Convert to DataFrame for easier manipulation
        full_evs = pd.DataFrame.from_records(events)
        print(full_evs.head())

        # # Ensure required columns exist
        # for col in ["value_recall", "actual_value"]:
        #     if col not in full_evs.columns:
        #         full_evs[col] = np.nan

        # Separate WORD and VALUE_RECALL events
        words = full_evs[full_evs.type == "WORD"]
        recalls = full_evs[full_evs.type == "VALUE_RECALL"]

        print(words.head())
        print(recalls.head())

        if words.empty or recalls.empty:
            print("No WORD or VALUE_RECALL events found; skipping modification.")
            return events  # nothing to do

        # Only keep the relevant recall columns if they exist
        # recall_cols = [c for c in ["trial", "value_recall", "actual_value"] if c in recalls.columns]
        # recalls = recalls[recall_cols].dropna(subset=["trial"], how="any")

        # # Merge recall data into word events (trial-based merge)
        # words_reset = words.reset_index()  # preserve mapping to original indices
        # merged = words_reset.merge(recalls, on="trial", how="left", suffixes=("", "_rec"))

        # # Apply updates back to full_evs
        # for _, row in merged.iterrows():
        #     orig_idx = int(row["index"])
        #     if pd.notna(row.get("value_recall")):
        #         full_evs.at[orig_idx, "value_recall"] = row["value_recall"]
        #     if pd.notna(row.get("actual_value")):
        #         full_evs.at[orig_idx, "actual_value"] = row["actual_value"]

        # Return as record array with same dtype as input
        # full_evs.to_records(index=False)
        return events



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

    #overwrite
    def add_pointing_finished(self, evdata):
        return evdata

    def event_classifier_wait(self, evdata):
        return evdata

    def event_classifier_result(self, evdata):
        return evdata

    #overwrite
    # def modify_rec_start(self, events):
    #     print("modify rec start called")
    #     rec_start_event = events[-1]
    #     rec_start_time = rec_start_event.mstime

    #     if self.practice:
    #         # Practice parsing not implemented for Courier / NICLS
    #         return events
    #     else:
    #         ann_outputs = self._parse_ann_file(str(self._trial))

    #     for recall in ann_outputs:
    #         new_event = self._new_rec_event(recall, rec_start_event)

    #         # Create a new event for the recall
    #         evtype = 'REC_WORD_VV' if "<>" in new_event["item"] else 'REC_WORD'
    #         new_event.type = evtype
    #         new_event = self._identify_intrusion(events, new_event)

    #         # Update recalled/intruded fields on matching WORDs
    #         if new_event.intrusion > 0:
    #             events.intruded[(events["type"] == 'WORD') & (events["item"] == new_event["item"])] = 1
    #         elif new_event.intrusion == 0:
    #             events.recalled[(events["type"] == 'WORD') & (events["item"] == new_event["item"])] = 1

    #         # --- 🔧 Coerce new_event to match events dtype ---
    #         new_event_casted = np.zeros(1, dtype=events.dtype).view(np.recarray)
    #         for name in events.dtype.names:
    #             if hasattr(new_event, name):
    #                 new_event_casted[name][0] = getattr(new_event, name)
    #             elif isinstance(new_event, dict) and name in new_event:
    #                 new_event_casted[name][0] = new_event[name]
    #             else:
    #                 # leave default (0, '', etc.)
    #                 pass

    #         # Append safely
    #         print("old")
    #         # print(events)
    #         print(events.dtype)
    #         print("new")
    #         # print(new_event)
    #         print(new_event_casted.dtype)
    #         events = np.concatenate([events, new_event_casted]).view(np.recarray)

    #     return events

    def modify_rec_start(self, events):
        print("modify_rec_start called")
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        # Skip practice trials
        if self.practice:
            return events

        # Load annotations for the current trial
        try:
            ann_outputs = self._parse_ann_file(str(self._trial))
        except Exception as e:
            print(f"⚠️ Missing or unreadable annotation file for trial {self._trial}: {e}")
            return events

        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)

            # Label the recall type
            evtype = 'REC_WORD_VV' if "<>" in new_event["item"] else 'REC_WORD'
            new_event.type = evtype

            # Identify intrusion and link back to word presentation
            new_event = self._identify_intrusion(events, new_event)

            # Mark recalled/intruded items
            if new_event.intrusion > 0:
                events.intruded[(events["type"] == 'WORD') & (events["item"] == new_event["item"])] = 1
            elif new_event.intrusion == 0:
                events.recalled[(events["type"] == 'WORD') & (events["item"] == new_event["item"])] = 1

            # --- Ensure dtype match ---
            new_event_casted = np.zeros(1, dtype=events.dtype).view(np.recarray)
            for name in events.dtype.names:
                if isinstance(new_event, np.recarray) and name in new_event.dtype.names:
                    new_event_casted[name][0] = new_event[name]
                elif isinstance(new_event, dict) and name in new_event:
                    new_event_casted[name][0] = new_event[name]
                elif hasattr(new_event, name):
                    new_event_casted[name][0] = getattr(new_event, name)

            # # Append safely
            # print("old")
            # # print(events)
            # print(events.dtype)
            # print("new")
            # # print(new_event)
            # print(new_event_casted.dtype)
            # # Append new recall event
            events = np.append(events, new_event_casted).view(np.recarray)

        return events
