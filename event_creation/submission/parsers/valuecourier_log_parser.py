import numpy as np
import pandas as pd
import six
from . import dtypes
from .courier_log_parser import CourierSessionLogParser


class ValueCourierSessionLogParser(CourierSessionLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        super().__init__(protocol, subject, montage, experiment, session, files)

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
           start_classifier_wait=self.event_classifier_wait,
           stop_classifier_wait=self.event_classifier_result,
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
        self.phase = 'video'
        event = self.event_default(evdata)
        event.type = 'MUSIC_VIDEOS_START'
        return event

    def event_music_videos_stop(self, evdata):
        self.phase = '2'
        event = self.event_default(evdata)
        event.type = 'MUSIC_VIDEOS_STOP'
        return event

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
        self.phase = 'video'
        event = self.event_default(evdata)
        event.type = 'MUSIC_VIDEOS_REC_START'
        return event

    def event_stop_music_videos_recall(self, evdata):
        self.phase = '2'
        event = self.event_default(evdata)
        event.type = 'MUSIC_VIDEOS_REC_STOP'
        return event

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
        event.value_recall = evdata['data']['typed response']
        event.actual_value = evdata['data']['actual value']
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
        event.finalrecalled = 0
        event.value = evdata['data']['store value']

        # Convert store position to string
        store_pos_str = self.stringify_list(evdata['data']['store position'])

        # Parse string into floats
        store_position = [float(p) for p in store_pos_str.strip('()').replace(' ', '').split(',')]

        event.storeX = store_position[0] if len(store_position) > 0 else None
        event.storeZ = store_position[2] if len(store_position) > 2 else None

        event.presX = self.presX
        event.presZ = self.presZ

        event.primacyBuf = evdata['data']['primacy buffer']
        event.recencyBuf = evdata['data']['recency buffer']

        event.numInGroupChosen = evdata['data']['number of in group chosen']

        item_name = self.stringify_list(evdata['data']['item name'])
        event["item"] = item_name.upper().rstrip('.1')

        return event


    def modify_word_with_value_recall(self, events):
        full_evs = pd.DataFrame.from_records(events)

        # Only keep WORD and VALUE_RECALL events
        words = full_evs[full_evs.type == "WORD"].copy()
        recalls = full_evs[full_evs.type == "VALUE_RECALL"].copy()

        if words.empty or recalls.empty:
            return events  # nothing to do

        # Reset index on words to preserve mapping back to full_evs
        words_reset = words.reset_index()  # 'index' column maps to original full_evs index
        merged = words_reset.merge(
            recalls[["trial", "value_recall", "actual_value"]],
            on="trial",
            how="left",
            suffixes=("", "_rec")
        )

        # Update WORD events in full_evs using the preserved original index
        for _, row in merged.iterrows():
            orig_idx = int(row["index"])
            if pd.notna(row.get("value_recall")):
                full_evs.at[orig_idx, "value_recall"] = row["value_recall"]
            if pd.notna(row.get("actual_value")):
                full_evs.at[orig_idx, "actual_value"] = row["actual_value"]

        # Return as numpy recarray; keep it simple and let dtype coercion happen
        return full_evs.to_records(index=False)


# Overwrite normal Courier FFR and store recall
    def modify_store_recall(self, events):
        return events

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

    def modify_store_recall(self, events):
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

    def event_classifier_wait(self, evdata):
        event = self.event_default(evdata)
        event.type = 'CLASSIFIER_WAIT'
        return event

    def event_classifier_result(self, evdata):
        # Event indicates that the task stopped waiting for the classifier
        # This can either mean it received the desired value, timed out, or was
        # a sham
        event = self.event_default(evdata)
        event.type = 'CLASSIFIER'
        event.classifier = evdata['data']['type']
        try:
            # Sham events do not time out, and don't have "timed out" field
            if evdata['data']['timed out']==0:
                return event
            # Don't return timed out events
            else:
                event.type = 'TIMEOUT'
                return event
        except:
            return event
