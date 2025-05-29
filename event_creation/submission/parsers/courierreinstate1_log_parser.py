import numpy as np
import pandas as pd
from . import dtypes
from .courier_log_parser import CourierSessionLogParser

class CourierReinstate1SessionLogParser(CourierSessionLogParser):
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
           Object_Reinstatement = self.object_reinstatement,
           continuous_pointer=self.event_pointer_on,
           start_required_break=self.event_break_start,
           stop_required_break=self.event_break_stop,
           start_deliveries=self.event_trial_start,
           stop_deliveries=self.event_trial_end,
           start_classifier_wait=self.event_classifier_wait,
           stop_classifier_wait=self.event_classifier_result,
        )
        self._add_type_to_modify_events(
           stop_deliveries=self.modify_pointer_on,
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

    def object_reinstatement(self, evdata):
        self._trial = evdata['data']['trial number']
        self._reinstated_item = evdata['data']['previously delivered item seen']
        self._angle_difference = evdata['data']['Angle of Approach']
        self.presX = evdata['data']['positionX']
        self.presZ = evdata['data']['positionZ']
        event = self.event_default(evdata)
        event.type = "REINSTATEMENT"
        event["item"] = evdata['data']['previously delivered item seen'].upper().rstrip('.1')

        event.intruded = 0
        event.recalled = 0
        event.finalrecalled = 0

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
            evtype = 'FFR_REC_WORD_VV' if "<>" in new_event["item"] else 'FFR_REC_WORD'
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
