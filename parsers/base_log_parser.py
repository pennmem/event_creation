import numpy as np
from viewers.view_recarray import pformat_rec
import os
import re
from loggers import log
import codecs

class UnparsableLineException(Exception):
    pass


class UnknownExperimentTypeException(Exception):
    pass


class IncomparableFieldException(Exception):
    pass


class BaseSessionLogParser(object):
    """
    BaseSessionLogParser contains the basic structure for creating events from session.log files

    Three functions should be called within __init__ of overriding functions:

    _add_fields(field_template):
        field_template should be a tuple of the same format as _BASE_FIELDS. That is, ((NAME, DEFAULT, DTYPE), ...)
        This sets the default event type. Fields must be added in order to be modified in other functions.

    _add_type_to_new_event(**kwargs)
        keyword arguments should provide a function to be called for each "TYPE" of log line, where "TYPE" is the
        "_TYPE_INDEX"th part of session.log when split by tabs.
        The function accepts the line split by tabs, and returns a new event or False if no event is to be added

        arguments should have format:
        TYPE = fn
        where fn is of format:
        def fn(split_line):
            return new_event

    _add_type_to_modify_event(**kwargs)
        keyword arguments should provide a function to be called for each "TYPE" of log, where "TYPE" is the
        "_TYPE_INDEX"th part of session.log when split by tabs.
        The function accepts all events that have been created up until that point, and returns all events after
        modification

        arguments should have the format:
        TYPE = fn
        where fn is of format:
        def fn(events):
            return events_modified
    """

    # Index in split line corresponding to these fields
    _MSTIME_INDEX = 0
    _MSOFFSET_INDEX = 1
    _TYPE_INDEX = 2

    # Token on which to split lines of session.log
    _SPLIT_TOKEN = '\t'

    # FORMAT: (NAME, DEFAULT, DTYPE)
    _BASE_FIELDS = (
        ('subject', '', 'S64'),
        ('montage', '', 'S64'),
        ('session', -1, 'int16'),
        ('type', '', 'S64'),
        ('mstime', -1, 'int64'),
        ('msoffset', -1, 'int16'),
        ('eegoffset', -1, 'int64'),
        ('eegfile', '', 'S256')
    )

    START_EVENT = 'SESS_START'

    MAX_ANN_LENGTH = 600000

    def __init__(self, subject, montage, files, primary_log='session_log', allow_unparsed_events=False):
        """

        :param session_log: path to session.log
        :param subject: subject for this session
        :param ann_dir: directory containing annotation files (defaults to containing dir of session.log)
        :param allow_unparsed_events: True if it is acceptable to leave lines unparsed
        :return:
        """
        self._session_log = files[primary_log]
        self._allow_unparsed_events = allow_unparsed_events
        self._ann_files = {os.path.basename(os.path.splitext(ann_file)[0]): ann_file for ann_file in files['annotations']}

        self._subject = subject
        self._montage = montage
        self._contents = [line.strip().split(self._SPLIT_TOKEN)
                          for line in codecs.open(self._session_log, encoding='utf-8').readlines()]
        self._fields = self._BASE_FIELDS
        self._type_to_new_event = {
            'B': self._event_skip,
            'E': self._event_skip
        }
        self._type_to_modify_events = {}
        pass

    @staticmethod
    def persist_fields_during_stim(event):
        """
        Defines the field for a given event which should persist into the stim event
        """
        return []

    @property
    def event_template(self):
        return self._fields

    def _add_fields(self, *args):
        """
        Adds fields to events structure
        :param *args: args of format (name, default_value, dtype)
        """
        init_fields = list(self._fields)
        init_fields.extend(args)
        self._fields = tuple(init_fields)

    def _add_type_to_new_event(self, **kwargs):
        """
        Adds type->function mapping
        :param kwargs: TYPE = function
        """
        for (key, value) in kwargs.items():
            self._type_to_new_event[key] = value

    def _add_type_to_modify_events(self, **kwargs):
        """
        Adds type-> modify mapping
        :param kwargs:  TYPE = function
        """
        for (key, value) in kwargs.items():
            self._type_to_modify_events[key] = value

    @classmethod
    def event_from_template(cls, template):
        """
        Creates events out of template of type ( (name1, default1, dtype1), (name2, ...), ...)
        :param template:
        :return: recarray of these names, defaults, and types
        """
        defaults = tuple(field[1] for field in template)
        dtypes = cls.dtype_from_template(template)
        return np.rec.array(defaults, dtype=dtypes)

    @classmethod
    def dtype_from_template(cls, template):
        dtypes = {'names': [field[0] for field in template],
                  'formats': [field[2] for field in template]}
        return dtypes

    @property
    def _empty_event(self):
        """
        Returns an event with fieldnames and defaults from self._fields
        :return:
        """
        event = self.event_from_template(self._fields)
        event.subject = self._subject
        event.montage = self._montage
        return event

    @staticmethod
    def _event_skip(*_):
        return False

    def event_default(self, split_line):
        """
        Returns a default event with mstime, msoffset, and type filled in
        :param split_line:
        :return:
        """
        event = self._empty_event
        event.mstime = int(split_line[self._MSTIME_INDEX])
        event.msoffset = int(split_line[self._MSOFFSET_INDEX])
        event.type = split_line[self._TYPE_INDEX]
        return event

    def parse(self):
        """
        Makes all events for the passed in file
        :return:
        """
        events = self._empty_event
        for split_line in self._contents:
            this_type = split_line[self._TYPE_INDEX]
            if this_type in self._type_to_new_event:
                new_event = self._type_to_new_event[this_type](split_line)
                if not isinstance(new_event, np.recarray) and not (new_event is False):
                    raise Exception('Event not properly provided from log parser')
                elif isinstance(new_event, np.recarray):
                    events = np.append(events, new_event)
            elif self._allow_unparsed_events:
                log('Warning: type %s not found' % this_type)
            else:
                raise UnparsableLineException("Event type %s not parseable" % this_type)
            if this_type in self._type_to_modify_events:
                events = self._type_to_modify_events[this_type](events.view(np.recarray))

        # Remove first (empty) event
        return events[1:]

    # Used to find relevant lines in .ann files
    MATCHING_ANN_REGEX = r'\d+\.\d+\s+-?\d+\s+(([A-Z]+)|(<>))'

    def _parse_ann_file(self, ann_id):
        if ann_id not in self._ann_files:
            return []
        ann_file = self._ann_files[ann_id]
        lines = open(ann_file, 'r').readlines()
        matching_lines = [line for line in lines if line[0] != '#' and re.match(self.MATCHING_ANN_REGEX, line.strip())]
        # Remove events with rectimes greater than 10 minutes, because they're probably a mistake
        split_lines = [line.split() for line in matching_lines if float(line.split()[0]) < self.MAX_ANN_LENGTH]
        return [(float(line[0]), int(line[1]), line[2]) for line in split_lines]


class EventComparator:
    """
    Compares two sets of np.recarray events
    """

    SHOW_FULL_EVENTS = False

    def __init__(self, events1, events2, field_switch=None, field_ignore=None, exceptions=None, type_ignore=None):
        """
        :param events1:
        :param events2:
        :param field_switch: {'event1_field' : 'matching_event2_field', ...}
        :param field_ignore: ('fields1_to_ignore', 'field2_to_ignore') (ignores the field in both sets of events)
        :param exceptions: function that allows you to ignore a discrepancy
        :param type_ignore: ('type1_to_ignore', 'type2_to_ignore' ...) (ignores events with events.type in type_ignore)
        :return:
        """
        self.events1 = events1
        self.events2 = events2
        self.field_switch = field_switch if field_switch else {}
        self.field_ignore = field_ignore if field_ignore else []
        self.type_ignore = type_ignore if type_ignore else []
        self.exceptions = exceptions if exceptions else lambda *_: False

        ev1_names = events1.dtype.names
        ev2_names = events2.dtype.names

        for name in ev1_names:
            if name not in ev2_names and name not in field_switch and name not in field_ignore:
                raise IncomparableFieldException(name)

        for name in ev2_names:
            if name not in ev1_names and name not in field_switch.values() and name not in field_ignore:
                raise IncomparableFieldException(name)

        # Get rid of fields to ignore
        for name in ev1_names:
            if name not in self.field_ignore and name not in self.field_switch:
                self.field_switch[str(name)] = str(name)
        self.events1 = self.events1[self.field_switch.keys()]
        self.events2 = self.events2[self.field_switch.values()]

        # Name all fields the same in both events
        self.events2.dtype.names = self.field_switch.keys()
        self.names = self.field_switch.keys()

    def _get_field_mismatch(self, event1, event2, subfield=None):
        """
        Returns the names of fields that do not match between event1 and event2
        :return:
        """
        mismatch = []
        if subfield:
            ev1 = event1[subfield]
            ev2 = event2[subfield]
        else:
            ev1 = event1
            ev2 = event2

        names = set(ev1.dtype.names).intersection(set(ev2.dtype.names))
        for field in names:
            if isinstance(ev1[field], np.void) and ev2[field].dtype.names: # Why is this typing as void?
                mismatch.extend(self._get_field_mismatch(event1, event2, field))
            elif ev1[field] != ev2[field] and not self.exceptions(event1, event2, field, subfield):
                mismatch.append('%s: %s v. %s' % (field, ev1[field], ev2[field]))
        return mismatch

    def compare(self):
        found_bad = False
        err_msg = ''
        mask2 = np.ones(len(self.events2), dtype=bool)

        for this_ignore in self.type_ignore:
            mask2[self.events2['type'] == this_ignore] = False

        bad_events1 = self.events1[0] # Have to initialize with something to fill it
        for i, event1 in enumerate(self.events1):
            this_mask2 = np.logical_and(
                    np.abs(event1['mstime'] - self.events2['mstime']) <= 4, event1['type'] == self.events2['type'])
            if not this_mask2.any() and not event1['type'] in self.type_ignore:
                bad_events1 = np.append(bad_events1, event1)
            elif event1['type'] not in self.type_ignore:
                mismatches = self._get_field_mismatch(event1, self.events2[this_mask2])
                if len(mismatches) > 0:
                    found_bad = True
                    bad_events1 = np.append(bad_events1, event1)
                    for mismatch in mismatches:
                        err_msg += 'mismatch: %d %s\n' % (i, mismatch);

            mask2[this_mask2] = False

        if bad_events1.size > 1:
            for bad_event1 in bad_events1[1:]:
                if not self.exceptions(bad_event1, None, None):
                    found_bad = True
                    err_msg += '\n---\n' + pformat_rec(bad_event1)

        if mask2.any():
            for bad_event2 in self.events2[mask2]:
                if not self.exceptions(None, bad_event2, None):
                    found_bad = True
                    err_msg += '\n---\n' + pformat_rec(bad_event2)

        return found_bad, err_msg


def get_version_num(session_log_file):
    with open(session_log_file, 'r') as f:
        for line in f:
            split_line = line.split()
            if split_line[2] == 'SESS_START':
                version = float(split_line[-1].replace('v_', ''))
                return version