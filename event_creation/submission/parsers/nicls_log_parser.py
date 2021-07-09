import numpy as np
import pandas as pd
from . import dtypes
from .base_log_parser import BaseUnityLogParser
from .courier_log_parser import CourierSessionLogParser
import six

class NICLSSessionLogParser(CourierSessionLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(NICLSSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files)

        self._add_fields(*dtypes.efr_fields)
        self._add_fields(*dtypes.courier_fields)
        self._add_fields(*dtypes.ltp_fields)

        self._add_type_to_new_event(
           start_movie=self.event_start_movie,
           stop_movie=self.event_stop_movie,
           start_town_learning=self.event_town_learning_start,
           stop_town_learning=self.event_town_learning_end,
           keypress=self.event_efr_mark,
           continuous_pointer=event_pointer_on,
           start_required_break=self.event_break_start,
           stop_required_break=self.event_break_stop,
           start_deliveries=self.event_trial_start,
           stop_deliveries=self.event_trial_end,
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
        self._trial = evdata['trial number']
        event = self.event_default(evdata)
        event.type = 'TRIAL_START'
        return event
    
    def event_trial_end(self, evdata):
        event = self.event_default(evdata)
        event.type = 'TRIAL_END'
        return event

    def event_movie_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'MOVIE_START'
        return event
    
    def event_movie_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'MOVIE_STOP'
        return event
    
    def event_town_learning_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'TL_START'
        return event

    def event_town_learning_end(self, evdata):
        event = self.event_default(evdata)
        event.type = 'TL_END'
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

