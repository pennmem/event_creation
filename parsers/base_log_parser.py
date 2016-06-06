import numpy as np
from viewers.view_recarray import pprint_rec, pformat_rec
import os
import re


class UnparsableLineException(Exception):
    pass


class UnknownExperimentTypeException(Exception):
    pass


class IncomparableFieldException(Exception):
    pass


class BaseSessionLogParser:
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
            return events_mofidied
    """


    # Index in split line corresponding to these fields
    _MSTIME_INDEX = 0
    _MSOFFSET_INDEX = 1
    _TYPE_INDEX = 2

    # Token on which to split lines of session.log
    _SPLIT_TOKEN = '\t'

    # FORMAT: (NAME, DEFAULT, DTYPE)
    _BASE_FIELDS = (
        ('subject', '', 'S20'),
        ('session', -1, 'int16'),
        ('type', '', 'S20'),
        ('mstime', -1, 'int64'),
        ('msoffset', -1, 'int16'),
        ('eegoffset', -1, 'int64'),
        ('eegfile', '', 'S64')
    )

    def __init__(self, session_log, subject, ann_dir=None, allow_unparsed_events=False):
        """

        :param session_log: path to session.log
        :param subject: subject for this session
        :param ann_dir: directory containing annotation files (defaults to containing dir of session.log)
        :param allow_unparsed_events: True if it is acceptable to leave lines unparsed
        :return:
        """
        self._session_log = session_log
        self._allow_unparsed_events = allow_unparsed_events
        if not ann_dir:
            self._ann_dir = os.path.dirname(session_log)
        self._subject = subject
        self._contents = [line.strip().split(self._SPLIT_TOKEN) for line in file(session_log, 'r').readlines()]
        self._fields = self._BASE_FIELDS
        self._type_to_new_event = {
            'B': self._event_skip,
            'E': self._event_skip
        }
        self._type_to_modify_events = {}
        pass

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
    def _event_from_template(cls, template):
        """
        Creates events out of template of type ( (name1, default1, dtype1), (name2, ...), ...)
        :param template:
        :return: recarray of these names, defaults, and types
        """
        defaults = tuple(field[1] for field in template)
        dtypes = cls._dtype_from_template(template)
        return np.rec.array(defaults, dtype=dtypes)

    @classmethod
    def _dtype_from_template(cls, template):
        dtypes = {'names': [field[0] for field in template],
                  'formats': [field[2] for field in template]}
        return dtypes

    @property
    def _empty_event(self):
        """
        Returns an event with fieldnames and defaults from self._fields
        :return:
        """
        event = self._event_from_template(self._fields)
        event.subject = self._subject
        return event

    def _event_skip(self, *_):
        return False

    def _event_default(self, split_line):
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
            type = split_line[self._TYPE_INDEX]
            if type in self._type_to_new_event:
                new_event = self._type_to_new_event[type](split_line)
                if not isinstance(new_event, (np.recarray)) and new_event!=False:
                    raise Exception()
                elif isinstance(new_event, (np.recarray)):
                    events = np.append(events, new_event)
            elif self._allow_unparsed_events:
                print('Warning: type %s not found' % type)
            else:
                raise UnparsableLineException("Event type %s not parseable" % type)
            if type in self._type_to_modify_events:
                events = self._type_to_modify_events[type](events.view(np.recarray))

        # Remove first (empty) event
        return events[1:]

    # Used to find relevant lines in .ann files
    MATCHING_ANN_REGEX = r'\d+\.\d+\s+-?\d+\s(([A-Z]+)|(<>))'

    def _parse_ann_file(self, ann_id):
        ann_file = os.path.join(self._ann_dir, '%s.ann' % ann_id)
        lines = open(ann_file, 'r').readlines()
        matching_lines = [line for line in lines if re.match(self.MATCHING_ANN_REGEX, line.strip())]
        split_lines = [line.split() for line in matching_lines]
        return [(float(line[0]), int(line[1]), line[2]) for line in split_lines]



class EventComparator():
    """
    Compares two sets of np.recarray events
    """

    SHOW_FULL_EVENTS = True

    def __init__(self, events1, events2, field_switch = None, field_ignore = None, exceptions = None, type_ignore = None):
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
        self.exceptions = exceptions if exceptions else lambda *_ : False

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
            if name not in field_ignore and name not in field_switch:
                self.field_switch[name] = name
        self.events1 = self.events1[self.field_switch.keys()]
        self.events2 = self.events2[self.field_switch.values()]

        # Name all fields the same in both events
        self.events2.dtype.names = self.field_switch.keys()
        self.names = self.field_switch.keys()

    def _get_field_mismatch(self, event1, event2):
        """
        Returns the names of fields that do not match between event1 and event2
        :return:
        """
        mismatch = []
        for field in self.names:
            if (event1[field] != event2[field]):
                if self.exceptions(event1, event2, field):
                    continue
                mismatch.append(field)
        return mismatch

    def compare(self):
        found_bad = False
        err_msg = ''
        mask2 = np.ones(len(self.events2), dtype=bool)

        for this_ignore in self.type_ignore:
            mask2[self.events2['type'] == this_ignore] = False

        bad_events1 = self.events1[0]
        for event1 in self.events1:
            this_mask2 = np.logical_and(np.abs(event1['mstime'] - self.events2.mstime) <= 1, event1['type'] == self.events2.type)
            if not this_mask2.any() and not event1['type'] in self.type_ignore:
                bad_events1 = np.append(bad_events1, event1)
            else:
                mismatch = self._get_field_mismatch(event1, self.events2[this_mask2])
                if len(mismatch) > 0:
                    found_bad = True
                    bad_events1 = np.append(bad_events1, event1)
                    if not self.SHOW_FULL_EVENTS:
                        for compare in mismatch:
                            err_msg += '\n' + compare + '\n' + event1[compare] + '\n' + self.events2[this_mask2][compare]
                    else:
                        err_msg += "\n\n"+ str(mismatch)+ ":"
                        err_msg += '\n' + pformat_rec(event1)
                        err_msg += '\n' + pformat_rec(self.events2[this_mask2][0])


            mask2[this_mask2] = False
        if bad_events1.size > 1:
            found_bad = True
            err_msg += '\n' + pformat_rec(bad_events1[1:])

        if mask2.any():
            found_bad = True
            err_msg += '\n' + pformat_rec(self.events2[mask2])

        return found_bad, err_msg
