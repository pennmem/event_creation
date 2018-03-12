from .base_log_parser import BaseSessionLogParser
import numpy as np
from copy import deepcopy
import re

class System2LogParser:

    _STIM_PARAMS_FIELD = 'stim_params'
    _STIM_ON_FIELD = 'is_stim'
    _TASK_SORT_FIELD = 'mstime'
    _HOST_SORT_FIELD = 'hosttime'

    _SYS2_FIELDS = BaseSessionLogParser._STIM_FIELDS + (
        ('hosttime', -1, 'int64'),
        ('file_index', -1, 'int16')
    )


    _LOG_TO_FIELD = {
        'E1': 'anode_number',
        'E2': 'cathode_number',
        'AMP': 'amplitude',
        'BFREQ': 'burst_freq',
        'NBURST': 'n_bursts',
        'PFREQ': 'pulse_freq',
        'NPULSE': 'n_pulses',
        'WIDTH': 'pulse_width'
    }

    _EXPERIMENT_TO_ITEM_FIELD = {
        'FR':'item_name',
        'PAL': 'study_1',
        'TH': 'item_name'
    }
    _EXPERIMENT_ITEM_OFF_TYPE={
        'FR':'WORD_OFF',
        'PAL': 'STUDY_PAIR_OFF',
        'TH': 'CHEST'
    }


    def __init__(self, host_logs, jacksheet=None):
        stim_events = self._empty_event()
        for i, log in enumerate(host_logs):
            stim_lines = self.get_rows_by_type(log, 'STIM')
            for line in stim_lines:
                if len(line) > 4:
                    stim_events = np.append(stim_events, self.make_stim_event(line, jacksheet))
                    stim_events[-1].stim_params['file_index'][0] = i
        self._stim_events = stim_events[1:] if stim_events.shape else stim_events

    @classmethod
    def sys2_fields(cls):
        return cls._SYS2_FIELDS

    @property
    def stim_events(self):
        return self._stim_events

    @classmethod
    def empty_stim_params(cls):
        return BaseSessionLogParser.event_from_template(cls._SYS2_FIELDS).view(np.recarray)

    @classmethod
    def stim_params_template(cls):
        return (('stim_params',
                cls.empty_stim_params(),
                BaseSessionLogParser.dtype_from_template(cls._SYS2_FIELDS),
                BaseSessionLogParser.MAX_STIM_PARAMS),)

    @classmethod
    def _empty_event(cls):
        return BaseSessionLogParser.event_from_template(cls.stim_params_template())

    def mark_stim_items(self,events):
        tasks = np.unique(events['experiment'])
        task = tasks[tasks != ''][0]
        task = re.sub(r'[\d.]', '',task)
        item_field = self._EXPERIMENT_TO_ITEM_FIELD[task]

        marked_stim_items = events[(events['is_stim']==True) &
                                   (events['type']==self._EXPERIMENT_ITEM_OFF_TYPE[task])][item_field]
        is_stim_item = np.in1d(events[item_field],marked_stim_items)
        events['is_stim'][is_stim_item] = np.ones(events[is_stim_item].shape).astype(bool)
        return events

    def merge_events(self, events, event_template, event_to_sort_value, persistent_field_fn):

        merged_events = events[:]

        # If no stim events available
        if len(self.stim_events.shape) == 0:
            return merged_events
        for i, stim_event in enumerate(self.stim_events):
            # Get the mstime for this host event
            sort_value = event_to_sort_value(stim_event.stim_params)
            # Determine where to insert it in the full events structure
            after_events = (merged_events[self._TASK_SORT_FIELD] > sort_value)
            if after_events.any():
                insert_index = after_events.nonzero()[0][0]
            else:
                insert_index = merged_events.shape[0]
            # Copy the persistent fields from the previous event, modify the remaining fields
            if insert_index > 0:
                event_to_copy = merged_events[insert_index-1]
            else:
                event_to_copy = BaseSessionLogParser.event_from_template(event_template)
            new_event = self.partial_copy(event_to_copy, event_template, persistent_field_fn)
            new_event.type = 'STIM_ON' if stim_event.stim_params['n_pulses'][0] > 1 else 'STIM_SINGLE_PULSE'
            new_event[self._STIM_ON_FIELD] = True
            new_event[self._TASK_SORT_FIELD] = sort_value
            new_event[self._STIM_PARAMS_FIELD] = stim_event.stim_params
            # Insert into the events structure
            merged_events = np.append(merged_events[:insert_index],
                                      np.append(new_event, merged_events[insert_index:]))

            if stim_event.stim_params['n_pulses'][0] > 1:
                # Do the same for the stim_off_event
                stim_off_sub_event = deepcopy(stim_event.stim_params).view(np.recarray)
                stim_off_sub_event.stim_on = False
                stim_off_sub_event.hosttime += stim_off_sub_event.stim_duration
                stim_off_value = event_to_sort_value(stim_off_sub_event)
                modify_indices = np.where(np.logical_and(merged_events[self._TASK_SORT_FIELD] > sort_value,
                                                         merged_events[self._TASK_SORT_FIELD] < stim_off_value))[0]

                # Modify the events between STIM and STIM_OFF to show that stim was applied
                for modify_index in modify_indices:
                    merged_events[modify_index][self._STIM_PARAMS_FIELD][0] = stim_event[0]
                    merged_events[modify_index][self._STIM_ON_FIELD] = True

                # Insert the STIM_OFF event after the modified events if any, otherwise directly after the STIM event
                insert_index = modify_indices[-1] + 1 if len(modify_indices) > 0 else insert_index + 1
                stim_off_event = self.partial_copy(merged_events[insert_index-1], event_template, persistent_field_fn)
                stim_off_event.type = 'STIM_OFF'
                stim_off_event[self._STIM_ON_FIELD] = False
                stim_off_event[self._STIM_PARAMS_FIELD] = stim_off_sub_event
                stim_off_event[self._TASK_SORT_FIELD] = stim_off_value

                # Merge the stim off event
                merged_events = np.append(merged_events[:insert_index],
                                          np.append(stim_off_event, merged_events[insert_index:]))
        merged_events = self.mark_stim_items(merged_events)
        return merged_events

    @staticmethod
    def partial_copy(event_to_copy, event_template, persistent_field_fn):
        new_event = BaseSessionLogParser.event_from_template(event_template)
        for field in persistent_field_fn(event_to_copy):
            new_event[field] = deepcopy(event_to_copy[field])
        return new_event

    @staticmethod
    def get_duration(params):
        duration_sec = (float(params['n_bursts'] - 1) / params['burst_freq']) + \
                        (float(params['n_pulses'] - 1) / params['pulse_freq']) + \
                        (float(params['pulse_width'])/1000000 * 2)
        return round(duration_sec*1000, -1)

    @classmethod
    def make_stim_event(cls, row, jacksheet=None):
        stim_event = cls._empty_event()
        stim_params = {}
        stim_params['hosttime'] = int(row[0])
        for item in row:
            split_item = item.split(':')
            if split_item[0] in cls._LOG_TO_FIELD:
                stim_params[cls._LOG_TO_FIELD[split_item[0]]]= float(split_item[1])

        #stim_params['amplitude'] = stim_params['amplitude']
        stim_params['n_bursts'] = stim_params['n_bursts'] or 1
        stim_params['burst_freq'] = stim_params['burst_freq'] or 1
        stim_params['stim_duration'] = cls.get_duration(stim_params)
        stim_params['pulse_freq'] = stim_params['pulse_freq'] if stim_params['n_pulses'] > 1 else 1
        stim_params['stim_on'] = True
        BaseSessionLogParser.set_event_stim_params(stim_event, jacksheet=jacksheet, **stim_params)
        return stim_event


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

