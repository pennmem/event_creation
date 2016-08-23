from base_log_parser import BaseSessionLogParser
import numpy as np
from copy import deepcopy


class System2LogParser:

    _STIM_PARAMS_FIELD = 'stimParams'
    _STIM_ON_FIELD = 'isStim'
    _TASK_SORT_FIELD = 'mstime'
    _HOST_SORT_FIELD = 'hostTime'

    _SYS2_FIELDS = (
        ('hostTime', -1, 'int32'),
        ('anode_number', -1, 'int16'),
        ('cathode_number', -1, 'int16'),
        ('anode_label', '', 'S64'),
        ('cathode_label', '', 'S64'),
        ('amplitude', -1, 'int16'),
        ('pulseFreq', -1, 'int16'),
        ('nPulses', -1, 'int16'),
        ('burstFreq', -1, 'int16'),
        ('nBursts', -1, 'int16'),
        ('pulseWidth', -1, 'int16'),
        ('stimOn', False, bool),
        ('stimDuration', -1, 'int16'),
        ('fileIndex', -1, 'int16')
    )

    _LOG_TO_FIELD = {
        'E1': 'anode_number',
        'E2': 'cathode_number',
        'AMP': 'amplitude',
        'BFREQ': 'burstFreq',
        'NBURST': 'nBursts',
        'PFREQ': 'pulseFreq',
        'NPULSE': 'nPulses',
        'WIDTH': 'pulseWidth'
    }

    def __init__(self, host_logs):
        stim_events = self._empty_event()
        for i, log in enumerate(host_logs):
            stim_lines = self.get_rows_by_type(log, 'STIM')
            for line in stim_lines:
                if len(line) > 4:
                    stim_events = np.append(stim_events, self.make_stim_event(line))
                    stim_events[-1]['fileIndex'] = i
        self._stim_events = stim_events[1:] if stim_events.shape else stim_events

    @classmethod
    def sys2_fields(cls):
        return cls._SYS2_FIELDS

    @property
    def stim_events(self):
        return self._stim_events

    def merge_events(self, events, event_template, event_to_sort_value, persistent_field_fn):

        merged_events = events[:]

        # If no stim events available
        if len(self.stim_events.shape) == 0:
            return merged_events
        for i, stim_event in enumerate(self.stim_events):
            # Get the mstime for this host event
            sort_value = event_to_sort_value(stim_event)
            # Determine where to insert it in the full events structure
            after_events = (merged_events[self._TASK_SORT_FIELD] > sort_value)
            if after_events.any():
                insert_index = after_events.nonzero()[0][0]
            else:
                insert_index = 0
            # Copy the persistent fields from the previous event, modify the remaining fields
            if insert_index > 0:
                event_to_copy = merged_events[insert_index-1]
            else:
                event_to_copy = BaseSessionLogParser.event_from_template(event_template)
            new_event = self.partial_copy(event_to_copy, event_template, persistent_field_fn)
            new_event.type = 'STIM'
            new_event[self._STIM_ON_FIELD] = True
            new_event[self._TASK_SORT_FIELD] = sort_value
            new_event[self._STIM_PARAMS_FIELD] = stim_event
            # Insert into the events structure
            merged_events = np.append(merged_events[:insert_index],
                                      np.append(new_event, merged_events[insert_index:]))

            # Do the same for the stim_off_event
            stim_off_sub_event = deepcopy(stim_event)
            stim_off_sub_event.stimOn = False
            stim_off_sub_event.hostTime += stim_off_sub_event.stimDuration
            stim_off_value = event_to_sort_value(stim_off_sub_event)
            modify_indices = np.where(np.logical_and(merged_events[self._TASK_SORT_FIELD] > sort_value,
                                                     merged_events[self._TASK_SORT_FIELD] < stim_off_value))[0]

            # Modify the events between STIM and STIM_OFF to show that stim was applied
            for modify_index in modify_indices:
                merged_events[modify_index][self._STIM_PARAMS_FIELD] = stim_event
                merged_events[modify_index][self._STIM_ON_FIELD] = True

            # Insert the STIM_OFF event after the modified events if any, otherwise directly after the STIM event
            insert_index = modify_indices[-1] + 1 if len(modify_indices) > 0 else insert_index + 1
            stim_off_event = self.partial_copy(merged_events[insert_index-1], event_template, persistent_field_fn)
            stim_off_event.type = 'STIM_OFF'
            stim_off_event[self._STIM_ON_FIELD] = False
            stim_off_event[self._STIM_PARAMS_FIELD] = stim_off_sub_event
            stim_off_event[self._TASK_SORT_FIELD] = stim_off_value

            # Merge the
            merged_events = np.append(merged_events[:insert_index],
                                      np.append(stim_off_event, merged_events[insert_index:]))
        return merged_events

    @staticmethod
    def partial_copy(event_to_copy, event_template, persistent_field_fn):
        new_event = BaseSessionLogParser.event_from_template(event_template)
        for field in persistent_field_fn(event_to_copy):
            new_event[field] = deepcopy(event_to_copy[field])
        return new_event

    @staticmethod
    def _get_duration(event):
        duration_sec = (float(event.nBursts - 1) / event.burstFreq) + \
                        (float(event.nPulses - 1) / event.pulseFreq) + \
                        (float(event.pulseWidth)/1000000 * 2)
        return round(duration_sec*1000, -1)

    @classmethod
    def make_stim_event(cls, row):
        stim_event = cls._empty_event()
        stim_event.hostTime = row[0]
        for item in row:
            split_item = item.split(':')
            if split_item[0] in cls._LOG_TO_FIELD:
                stim_event[cls._LOG_TO_FIELD[split_item[0]]] = float(split_item[1])
        stim_event.stimDuration = cls._get_duration(stim_event)
        stim_event.stimOn = True
        return stim_event

    @classmethod
    def _empty_event(cls):
        return BaseSessionLogParser.event_from_template(cls._SYS2_FIELDS).view(np.recarray)

    @classmethod
    def get_rows_by_type(cls, host_log_file, line_type, columns=None, cast=None):
        split_lines = [x.strip().split('~') for x in open(host_log_file, 'r').readlines()]
        rows = []
        for split_line in split_lines:
            if len(split_line) > 1 and split_line[1] == line_type:
                if columns:
                    if cast:
                        rows.append([cast(split_line[index]) for index in columns])
                    else:
                        rows.append([split_line[index] for index in columns])
                else:
                    rows.append(split_line)
        return rows

    @classmethod
    def get_columns_by_type(cls, host_log_file, line_type, columns=None, cast=None):
        output = zip(*cls.get_rows_by_type(host_log_file, line_type, columns, cast))
        if (columns and not output):
            output = [[]] * len(columns)
        return output

'''
def test_merge_events():
    import random
    System2LogParser._TASK_SORT_FIELD = 'index'
    s2lp = System2LogParser([])
    s2lp._stim_events.stimDuration = 4
    s2lp._stim_events.hostTime = 5
    for i in range(10,40,5):
        new_event = s2lp._empty_event()
        new_event.hostTime = i
        new_event.stimDuration = 4
        new_event.stimOn = True
        s2lp._stim_events  = np.append(s2lp._stim_events, new_event)

    template = (
        ('type', 'task', 'S20'),
        ('index', 0, 'int16'),
        ('random', 0, 'int16'),
        ('nocopy', -1, 'int16'),
        ('stimParams',
         BaseSessionLogParser.event_from_template(System2LogParser.sys2_fields()),
         BaseSessionLogParser.dtype_from_template(System2LogParser.sys2_fields()))
    )

    old_events = BaseSessionLogParser.event_from_template(template)
    for i in range(2,40,2):
        new_event = BaseSessionLogParser.event_from_template(template)
        new_event.index = i
        new_event.random = random.randint(0, 100)
        new_event.nocopy = random.randint(0,100)
        old_events = np.append(old_events, new_event)

    new_events = s2lp.merge_events(old_events, template, lambda ev: ev.hostTime, ('random',))

    from viewers.view_recarray import pprint_rec
    pprint_rec(new_events)
    print(new_events)
'''
