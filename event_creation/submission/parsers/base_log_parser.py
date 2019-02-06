import codecs
import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd

from ..log import logger
from ..exc import LogParseError, UnknownExperimentError, EventFieldError
from ..readers.eeg_reader import read_jacksheet
from ..viewers.recarray import pformat_rec, to_dict, from_dict
from ..exc import NoAnnotationError
from . import dtypes


class BaseLogParser(object):

    # Maximum length of stim params
    MAX_STIM_PARAMS = 10

    # Set to false if stim events are added inside the log parser
    ADD_STIM_EVENTS = True

    # Set to false if EEG offsets are added inside the log parser
    DO_ALIGNMENT = True

    # FORMAT: (NAME, DEFAULT, DTYPE)
    _BASE_FIELDS = dtypes.base_fields

    # These fields are to be added if include_stim_params is true
    _STIM_FIELDS = dtypes.stim_fields

    # Maximum amount of time after which a valid annotation can appear in a .ann file
    MAX_ANN_LENGTH = 600000

    # Tests to run in order to validate output
    _TESTS = []

    def __init__(self, protocol, subject, montage, experiment, session, files,
                 primary_log='session_log', allow_unparsed_events=False, include_stim_params=False):
        """
        constructor
        :param protocol: Protocol for this subject/session
        :param subject: Subject that ran in this session (no montage code)
        :param montage: Montage for this subject/session
        :param experiment: Experiment name and number for this session (e.g. FR3, catFR1)
        :param session: Session for this subject/session
        :param files: A dictionary of files (as returned from an instance of Transferer). Must include an entry with
                      the key as primary_log, and may include a list of files under the key 'annotations' pointing to
                      .ann files
        :param primary_log: The key to the 'files' dictionary in which the log to be parsed resides
        :param allow_unparsed_events: If false, parser will throw an exception when a line cannot be parsed
        :param include_stim_params: If true, events are initialized with an empty stim_params sub rec-array
        :return:
        """
        self._allow_unparsed_events = allow_unparsed_events
        self._protocol = protocol
        self._experiment = experiment
        self._subject = subject
        self._session = session
        self._montage = montage

        # Read the contents of the primary log file
        self._primary_log = files[primary_log]

        # Read the contents of the primary log file
        self._contents = self._read_primary_log()

        # _fields defines the structure of the final recarray
        self._fields = self._BASE_FIELDS
        if include_stim_params:
            self._fields += (self.stim_params_template(),)

        self._type_to_new_event = {}
        self._type_to_modify_events = {}

        # Try to read the jacksheet if it is present
        if 'contacts' in files:
            self._jacksheet = read_jacksheet(files['contacts'])
        elif 'jacksheet' in files:
            self._jacksheet = read_jacksheet(files['jacksheet'])
        else:
            self._jacksheet = None

        # Try to read annotation files if they are present
        try:
            self._ann_files = {os.path.basename(os.path.splitext(ann_file)[0]): ann_file
                               for ann_file in files['annotations']}
        except KeyError:
            self._ann_files = []

    @classmethod
    def empty_stim_params(cls):
        """
        :return: A record array of just the stimulation paramers
        """
        return cls.event_from_template(cls._STIM_FIELDS)

    @classmethod
    def stim_params_template(cls):
        """
        Maximum of 10 stimulated electrodes at once, completely arbitrarily
        """
        return 'stim_params', cls.empty_stim_params(), cls.dtype_from_template(cls._STIM_FIELDS), cls.MAX_STIM_PARAMS

    def clean_events(self, events):
        """
        Called after parsing events to make any final modifications that might need to be made. To be overridden in
        subclasses
        :param events: fully parsed events structure
        :return: modified events structure
        """
        order = np.argsort(events['mstime'])
        events = events[order]
        events = events.view(np.recarray)
        events.protocol = self._protocol
        events.experiment = self._experiment
        events.subject = self._subject
        events.session = self._session
        events.montage = self._montage
        return events

    def check_event_quality(self, events, files):
        """
        Called at the end of event creation to make sure that the events look like we expect them to.
        :param events: A complete events structure
        :param files: A dictionary of paths
        :return: A list of messages from failing tests
        """
        msgs = []
        for test in self._TESTS:
            try:
                test(events, files)
            except Exception as e:
                msgs.append('{}.{}: {}'.format(type(self).__name__, test.__name__, e.message))
        return msgs

    @staticmethod
    def persist_fields_during_stim(event):
        """
        Defines the field for a given event which should persist into the stim event
        :param event: The event immediately preceding the stim event
        :return: A list of the fields that should maintain their current value in the upcoming stim event
        """
        return []

    @property
    def event_template(self):
        """
        Returns the template for an empty event
        :return:
        """
        return self._fields

    def _add_fields(self, *args):
        """
        Adds fields to events structure
        :param *args: args of format [(name1, default_value1, dtype1), ...]
        """
        init_fields = list(self._fields)
        init_fields.extend(args)
        self._fields = tuple(init_fields)

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
        """
        Creates a dtype (for use in the creation of a recarray) based on the input template
        :param template:
        :return:
        """
        dtypes = [(entry[0], entry[2], entry[3] if len(entry) > 3 else 1) for entry in template]
        return dtypes

    @property
    def _empty_event(self):
        """
        Returns an event with fieldnames and defaults from self._fields
        Additionally adds the fields which persist across every event in the structure (subject, montage ...)
        :return:
        """
        event = self.event_from_template(self._fields)
        event.protocol = self._protocol
        event.subject = self._subject
        event.montage = self._montage
        event.experiment = self._experiment
        event.session = self._session

        return event

    @staticmethod
    def set_event_stim_params(event, jacksheet, index=0, **params):
        """
        Sets stimulation parameters for a given event, also applying label or number for stimulated contact
        :param event: The event to be modified
        :param jacksheet: Mapping of channel # -> channel name
        :param index: The index of the stimulation parameter (if 2 stims were applied, call once with 0 and once with 1)
        :param params: Keyword/value pairs setting individual parameters
        """

        for param, value in params.items():
            if param == 'amplitude' and value < 5:
                value *= 1000  # Put in uA. Ugly fix...
            if param == 'pulse_freq' and value > np.iinfo(event.stim_params[index][param].dtype).max:
                value /= 1000  # same fix, but for Hz
            if param in event.stim_params.dtype.names:
                event.stim_params[index][param] = value

        if 'anode_label' in params and 'anode_number' not in params:
            reverse_jacksheet = {v: k for k, v in jacksheet.items()}
            event.stim_params[index]['anode_number'] = reverse_jacksheet.get(params['anode_label'].upper(),
                                                                             reverse_jacksheet[params['anode_label']]
                                                                             )

        if 'cathode_label' in params and 'cathode_number' not in params:
            reverse_jacksheet = {v: k for k, v in jacksheet.items()}
            event.stim_params[index]['cathode_number'] = reverse_jacksheet.get(
                params['cathode_label'].upper(), reverse_jacksheet[params['cathode_label']])

        if 'anode_number' in params and 'anode_label' not in params:
            event.stim_params[index]['anode_label'] = jacksheet[params['anode_number']].upper()

        if 'cathode_number' in params and 'cathode_label' not in params:
            event.stim_params[index]['cathode_label'] = jacksheet[params['cathode_number']].upper()

        event.stim_params[index]._remove = False

    # Used to find relevant lines in .ann files
    # Note from Jesse Pazdera: I added '[???]' to the accepted strings that can appear in the third column in order to
    # resolve a conflict that arose in a strange case where an annotation file had this listed as the word, and the
    # MATLAB-generated events structure counted the [???] as an XLI word recall (ltpFR->LTP123->session_18->ffr.ann)
    MATCHING_ANN_REGEX = r'\d+(\.\d+)?\s+-?\d+\s+(([A-Z]+)|(<>)|(\[\?\?\?\]))'

    def _parse_ann_file(self, ann_id):
        """
        Parses a single annotation file (for use in subclasses)
        :param ann_id: The key with which to retrieve the specific annotation file from files['annotations']
        :return: (float, int, str) corresponding to (time, word number, word)
        """
        # Worry if we can't find the annotation file
        if ann_id not in self._ann_files:
            raise NoAnnotationError("Missing %s.ann"%ann_id)

        # Read the annotation file, getting everything that matches the regular expression
        ann_file = self._ann_files[ann_id]
        lines = codecs.open(ann_file, encoding='latin1').readlines()
        matching_lines = [line for line in lines if line[0] != '#' and re.match(self.MATCHING_ANN_REGEX, line.strip())]

        # Remove events with rectimes greater than 10 minutes, because they're probably a mistake
        split_lines = [line.split() for line in matching_lines if float(line.split()[0]) < self.MAX_ANN_LENGTH]
        return [(float(line[0]), int(line[1]), ' '.join(line[2:])) for line in split_lines]

    def _read_primary_log(self):
        """
        Creates the list of entries from the primary log file
        :return: A list in which each item corresponds to one entry in the primary log file
        """
        raise NotImplementedError("BaseLogParser is to be extended in classes that specify log format")

    def _get_raw_event_type(self, raw_event):
        """
        Given an entry returned from _read_primary_log, returns the type of that entry
        :return: A string, the type of the entry from the primary log
        """
        raise NotImplementedError("BaseLogParser is to be extended in classes that specify log format")

    def _add_type_to_new_event(self, **kwargs):
        """
        Defines the types which should correspond to the creation of a new event
        Adds type-> creation function mapping
        :param kwargs: TYPE = function
        """
        for (key, value) in kwargs.items():
            self._type_to_new_event[key] = value

    def _add_type_to_modify_events(self, **kwargs):
        """
        Defines the types which should correspond to the modification of existing events
        Adds type-> modifying function mapping
        :param kwargs:  TYPE = function
        """
        for (key, value) in kwargs.items():
            self._type_to_modify_events[key] = value

    @staticmethod
    def _event_skip(*_):
        """
        Called to skip event creation for a given line
        :param _: ignores arguments
        :return: False
        """
        return False

    def parse(self):
        """
        Makes all events for the primary log file
        :return: all events
        """
        # Start with a single empty event
        events = self._empty_event
        # Loop over the contents of the log file
        for raw_event in self._contents:
            this_type = self._get_raw_event_type(raw_event)
            # Check if the line is parseable
            try:
                handler = self._type_to_new_event[this_type]
            except KeyError:
                if self._allow_unparsed_events:
                    # Fine to skip lines if specified
                    continue
                else:
                    raise LogParseError("Event type %s not parseable" % this_type)

            new_event = handler(raw_event)
            if not isinstance(new_event, np.recarray) and not (new_event is False):
                raise Exception('Event not properly provided from log parser for raw event {}'.format(raw_event))
            elif isinstance(new_event, np.recarray):
                events = np.append(events, new_event)

            # Modify existing events if necessary
            if this_type in self._type_to_modify_events:
                events = self._type_to_modify_events[this_type](events.view(np.recarray))

        # Remove first (empty) event
        if events.ndim > 0:
            return events[1:]
        else:
            return events


class BaseSessionLogParser(BaseLogParser):
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
    _SPLIT_DELIMITER = '\t'

    # Alignment doesn't have to start until this event is seen
    START_EVENT = 'SESS_START'

    def __init__(self, protocol, subject, montage, experiment, session, files, primary_log='session_log',
                 allow_unparsed_events=False, include_stim_params=False):
        """
        constructor
        :param protocol: Protocol for this subject/session
        :param subject: Subject that ran in this session (no montage code)
        :param montage: Montage for this subject/session
        :param experiment: Experiment name and number for this session (e.g. FR3, catFR1)
        :param session: Session for this subject/session
        :param files: A dictionary of files (as returned from an instance of Transferer). Must include an entry with
                      the key as primary_log, and may include a list of files under the key 'annotations' pointing to
                      .ann files
        :param primary_log: The key to the 'files' dictionary in which the log to be parsed resides
        :param allow_unparsed_events: If false, parser will throw an exception when a line cannot be parsed
        :param include_stim_params: If true, events are initialized with an empty stim_params sub rec-array
        :return:
        """

        super(BaseSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files, primary_log,
                                                   allow_unparsed_events, include_stim_params)

        # These events are common in pyepl, and can be skipped by default
        self._add_type_to_new_event(
            B=self._event_skip,
            E=self._event_skip
        )

    def _read_primary_log(self):
        """
        Reads the lines from the primary log file, splitting each line based on the delimiter defined for the class
        :return: A list containing each line in the log file, split on the appropriate delimiter
        """
        return [line.strip().split(self._SPLIT_DELIMITER)
                for line in codecs.open(self._primary_log, encoding='utf-8').readlines()]

    def _add_type_to_new_event(self, **kwargs):
        """
        Defines the types which should correspond to the creation of a new event
        Adds type-> creation function mapping
        :param kwargs: TYPE = function
        """
        for (key, value) in kwargs.items():
            self._type_to_new_event[key] = value

    def _add_type_to_modify_events(self, **kwargs):
        """
        Defines the types which should correspond to the modification of existing events
        Adds type-> modifying function mapping
        :param kwargs:  TYPE = function
        """
        for (key, value) in kwargs.items():
            self._type_to_modify_events[key] = value

    def event_default(self, split_line):
        """
        Returns a default event with mstime, msoffset, and type filled in
        :param split_line: A single split line from the primary log file
        :return: new event
        """
        event = self._empty_event
        event.mstime = int(split_line[self._MSTIME_INDEX])
        event.msoffset = int(split_line[self._MSOFFSET_INDEX])
        event.type = split_line[self._TYPE_INDEX]
        return event

    def _get_raw_event_type(self, split_line):
        return split_line[self._TYPE_INDEX]


class BaseSys3LogParser(BaseLogParser):

    _STIME_FIELD = 't_event'
    _TYPE_FIELD = 'event_label'
    _EEG_OFFSET_FIELD = 'offset'

    def _read_primary_log(self):
        contents = []
        for log in self._primary_log:
            contents += json.load(codecs.open(log,  encoding='utf-8'))['events']
        return contents

    def event_default(self, event_json):
        """
        Returns a default event with mstime, msoffset, and type filled in
        :param split_line: A single split line from the primary log file
        :return: new event
        """
        event = self._empty_event
        event.mstime = event_json[self._STIME_FIELD] * 1000
        event.type = event_json[self._TYPE_FIELD]
        event.eegoffset = event_json[self._EEG_OFFSET_FIELD]
        return event

    def _get_raw_event_type(self, event_json):
        return event_json[self._TYPE_FIELD]


class BaseSys3_1LogParser(BaseSessionLogParser):
    _MSTIME_FIELD = 'timestamp'
    _TYPE_FIELD = 'event'
    _PHASE_TYPE_FIELD = 'phase_type'

    def __init__(self, protocol, subject, montage, experiment, session, files, primary_log='session_log',
                 allow_unparsed_events=False, include_stim_params=False):
        if primary_log not in files:
            primary_log = 'session_log_txt'
        self.LOG_READERS = {
            '.sql': self._read_sql_log,
            '.sqlite': self._read_sql_log,
            '.json': self._read_unityepl_log,
            '.log': self._read_session_log,
        }

        BaseSessionLogParser.__init__(self, protocol, subject, montage,
                                      experiment, session, files,
                                      primary_log=primary_log,
                                      allow_unparsed_events=allow_unparsed_events,
                                      include_stim_params=include_stim_params)
        self._files = files
        self._phase = ''

    def _get_raw_event_type(self, event_json):
        return event_json[self._TYPE_FIELD]

    # TODO: ADD LOGIC TO CONTROL WHETHER SESSION.SQLITE OR SESSION.LOG IS USED?
    # TODO: DECIDE WHETHER SESSION.LOG IS USABLE -- NOT ALWAYS CONSISTENT WITH SESSION.SQLITE
    #    AND THE INCONSISTENCIES ARE PROBLEMATIC
    def parse(self):
        try:
            return super(BaseSys3_1LogParser, self).parse()
        except Exception as exc:
            logger.warn('Encountered error in parsing session.sqlite: \n %s: %s' % (str(type(exc)), exc.message))
            if self._files.get('session_log_txt'):
                logger.warn('Parsing session.log instead')

                self._contents = self._read_session_log(self._files['session_log_txt'])
                return super(BaseSys3_1LogParser, self).parse()
            else:
                raise exc

    @staticmethod
    def _read_sql_log(log):
        conn = sqlite3.connect(log)
        query = 'SELECT msg FROM logs WHERE name = "events"'
        msgs = [json.loads(msg) for msg in pd.read_sql_query(query, conn).msg.values]
        conn.close()
        return msgs

    def _read_unityepl_log(self, filename):
        """Read events from the UnityEPL format (JSON strings separated by
        newline characters).

        :param str filename:

        """
        df = pd.read_json(filename, lines=True)
        messages = df[df.type == 'network'].data
        network = pd.DataFrame([msg['message']['data']
                                for msg in messages if msg['sent']])
        times = [msg['message']['time'] for msg in messages if msg['sent']]
        network[self._MSTIME_FIELD] = times
        self._TYPE_FIELD = self._UNITY_TYPE_FIELD
        events = network[network.name.notnull()].dropna(1, 'all').reset_index(drop=True)
        return [e.to_dict() for _, e in events.iterrows()]

    def _read_session_log(self, log):
        def load_json(s, *args, **kwargs):
            try:
                return json.loads(s, *args, **kwargs)
            except ValueError:
                return {}
        with open(log) as logfile:
            lines = [x.strip().split('\t') for x in logfile.readlines() if len(x.split('\t')) > 1]
        event_jsons = [load_json(x[-1].partition(' ')[-1]) for x in lines]
        mstimes = [int(x[0]) for x in lines]
        types = [x[-1].partition(' ')[0] for x in lines]
        for i in range(len(event_jsons)):
            event_jsons[i][self._MSTIME_FIELD] = mstimes[i]
            event_jsons[i][self._TYPE_FIELD] = types[i]
        return event_jsons

    def _read_primary_log(self):
        msgs = []
        if isinstance(self._primary_log, basestring):
            logs = [self._primary_log]
        else:
            logs = self._primary_log
        for log in logs:
            log_ext = os.path.splitext(log)[-1]
            msgs += self.LOG_READERS[log_ext](self._primary_log)
        return msgs

    def event_default(self, event_json):
        event = self._empty_event
        event.mstime = event_json[self._MSTIME_FIELD]
        event.type = event_json[self._TYPE_FIELD]
        event.phase = self._phase

        return event

    def clean_events(self, events):
        # Add in experiment version
        events = super(BaseSys3_1LogParser, self).clean_events(events)
        with open(self._files['event_log'][0], 'r') as event_log:
            version_info = json.load(event_log)['versions']
        events.exp_version = version_info['task']['version']
        return events


class BaseUnityLTPLogParser(BaseLogParser):

    def __init__(self, protocol, subject, montage, experiment, session, files, primary_log='session_log'):
        if primary_log not in files:
            primary_log = 'session_log_txt'

        BaseLogParser.__init__(self, protocol, subject, montage, experiment, session, files, primary_log=primary_log,
                               allow_unparsed_events=True)
        self._files = files
        self._trial = -999

    def _get_raw_event_type(self, event_json):
        return event_json['type']

    def parse(self):
        try:
            return super(BaseUnityLTPLogParser, self).parse()
        except Exception as exc:
            logger.warn('Encountered error in parsing %s session %s: \n %s: %s' % (self._subject, self._session,
                                                                                   str(type(exc)), exc.message))
            raise exc

    def _read_unityepl_log(self, filename):
        """
        Read events from the UnityEPL format (JSON strings separated by
        newline characters).

        :param str filename: The path to the session log you wish to parse.
        """
        # Read session log
        df = pd.read_json(filename, lines=True)
        # Filter out sync pulse events (these will be parsed later on during alignment)
        df = df[df.type != 'Sync pulse begin']
        # Create a list of dictionaries, where each dictionary is the information about one event
        events = [e.to_dict() for _, e in df.iterrows()]
        # Replace spaces in event type names with underscores
        for i, e in enumerate(events):
            events[i]['type'] = e['type'].replace(' ', '_')
            if e['type'] == 'stimulus' and 'displayed text' in e['data']:
                events[i]['type'] = 'stimulus_display'
        return events

    def _read_primary_log(self):
        evdata = self._read_unityepl_log(self._primary_log)
        return evdata

    def event_default(self, event_json):
        event = self._empty_event
        event.mstime = event_json['time']
        event.type = event_json['type']
        event.trial = self._trial
        return event


class RecogParser(BaseSessionLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(RecogParser, self).__init__(protocol, subject, montage, experiment, session, files,
                                          allow_unparsed_events=True)
        self._add_type_to_new_event(
            RECOG_WORD=self.event_recog
            # other event types?
        )
        self._add_fields(
            ('item_name', '', 'S12'),
            ('recognized', -1, 'int'),
            ('rejected', -1, 'int')
        )

    def event_recog(self, split_line):
        raise NotImplementedError


class EventComparator(object):
    """
    Compares two sets of np.recarray events, comparing events with matching types and mstimes and producing a list of
    discrepancies
    """
    # FIXME: Combine with StimComparator??

    def __init__(self, events1, events2, field_switch=None, field_ignore=None, exceptions=None, type_ignore=None,
                 type_switch=None, match_field='mstime', same_fields=True,verbose=True):
        """
        :param events1:
        :param events2:
        :param field_switch: {'event1_field' : 'matching_event2_field', ...}
        :param field_ignore: ('fields1_to_ignore', 'field2_to_ignore') (ignores the field in both sets of events)
        :param exceptions: function that allows you to ignore a discrepancy
        :param type_ignore: ('type1_to_ignore', 'type2_to_ignore' ...) (ignores events with events.type in type_ignore)
        :param type_switch: {'event1_type': 'event2_type', ...}
        :param match_field: Along with 'type', this field is used to decide which events to compare
        :param same_fields: Whether to expect that the fields in events1 and events2 are identical. If false,
                            only compare those fields that are present in both.
        """
        ev1_names = events1.dtype.names
        ev2_names = events2.dtype.names

        self.field_switch = field_switch if field_switch else {}
        self.type_switch = type_switch if type_switch else {}
        self.field_ignore = field_ignore if field_ignore else []
        self.type_ignore = type_ignore if type_ignore else []
        self.exceptions = exceptions if exceptions else lambda *_: False
        self.match_field = match_field
        self.verbose = verbose

        if not same_fields:
            names = [n for n in ev1_names if n in ev2_names]
            events1 = events1[names]
            events2 = events2[names]
        else:

            # Make sure we can compare the events
            for name in ev1_names:
                if name not in ev2_names and name not in self.field_switch and name not in self.field_ignore:
                    raise EventFieldError(name)

            for name in ev2_names:
                if name not in ev1_names and name not in self.field_switch.values() and name not in self.field_ignore:
                    raise EventFieldError(name)

        self.events1 = events1
        self.events2 = events2

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
        event1, event2 = (ev if not ev.shape else ev[0] for ev in (event1, event2))
        if subfield:
            ev1 = event1[subfield]
            ev2 = event2[subfield]
        else:
            ev1 = event1
            ev2 = event2

        names = set(ev1.dtype.names).intersection(set(ev2.dtype.names))
        for field in names:
            if isinstance(ev1[field], np.void) and ev2[field].dtype.names:  # Why is this typing as void?
                mismatch.extend(self._get_field_mismatch(event1, event2, field))
            elif ev2[field] != ev1[field] and not self.exceptions(event1, event2, field, subfield):
                mismatch.append('%s: %s v. %s' % (field, ev1[field], ev2[field]))
        return mismatch

    def compare(self):
        """
        Compares the provided events structures
        :return : found mismatch (boolean), string containing mismatches
        """
        found_bad = False
        err_msg = ''

        # Which of the events2 structure have been used
        mask2 = np.ones(len(self.events2), dtype=bool)

        # Ignore all of the events with bad types
        for this_ignore in self.type_ignore:
            mask2[self.events2['type'] == this_ignore] = False

        # Collect the events that failed comparison. Have to initialize with an event, which will be removed later.
        bad_events1 = self.events1[0]

        for i, event1 in enumerate(self.events1):
            # Get events that occurred close in time, and with the same type
            this_mask2 = np.logical_and(
                    np.abs(event1[self.match_field] - self.events2[self.match_field]) <= 4,
                    event1['type'] == self.events2['type'])

            # Get any other events with equivalent types
            if event1['type'] in self.type_switch:
                for type in self.type_switch[event1['type']]:
                    this_mask2 = np.logical_or(this_mask2, np.logical_and(
                        np.abs(event1[self.match_field] - self.events2[self.match_field]) <= 4,
                        type == self.events2['type']
                    ))

            # If we couldn't find a match, record this event
            if not this_mask2.any() and not event1['type'] in self.type_ignore:
                bad_events1 = np.append(bad_events1, event1)
            elif event1['type'] not in self.type_ignore:  # Otherwise, compare the events
                mismatches = self._get_field_mismatch(event1, self.events2[this_mask2])
                if len(mismatches) > 0:
                    found_bad = True
                    bad_events1 = np.append(bad_events1, event1)
                    for mismatch in mismatches:
                        err_msg += 'mismatch: %d %s\n' % (i, mismatch)

            # Mark that these events have been seen
            mask2[this_mask2] = False

        if self.verbose:
            # Gather any bad events from events1
            if bad_events1.size > 1:
                for bad_event1 in bad_events1[1:]:
                    if not self.exceptions(bad_event1, None, None):
                        found_bad = True
                        err_msg += '\n--1--\n' + pformat_rec(bad_event1)

            # Gather any bad events from events2
            if mask2.any():
                for bad_event2 in self.events2[mask2]:
                    if not self.exceptions(None, bad_event2, None):
                        found_bad = True
                        err_msg += '\n--2--\n' + pformat_rec(bad_event2)

        return found_bad, err_msg


class StimComparator(object):
    """
    Similar to EventComparator, but specifically for stimulation events, as it requires field/subfield comparison
    """

    def __init__(self, events1, events2, fields_to_compare, exceptions, match_field='mstime'):
        """
        :param events1:
        :param events2:
        :param fields_to_compare: {'field1.subfield1' -> 'field2.subfield2'}
        :param exceptions: function that defines okay mismatches
        :param match_field: which field to match events based upon
        """
        self.events1 = events1
        self.events2 = events2
        self.fields_to_compare = {}
        self.match_field = match_field
        for field_name1, field_name2 in fields_to_compare.items():
            try:
                # Check that both fields can be retrieved from the events structures
                _ = self.get_subfield(self.events1[0], field_name1)
                _ = self.get_subfield(self.events2[0], field_name2)
                self.fields_to_compare[field_name1] = field_name2
            except ValueError:
                logger.warn('Could not access fields {}/{} for comparison'.format(field_name1, field_name2))

        self.exceptions = exceptions

    @classmethod
    def get_subfield(cls, event, field_whole):
        """
        Gets a subfield from a recarray based on a string 'a.b.c'
        :param event: the event from which to retrieve the subfield
        :param field_whole: the string representing the subfield to retrieve
        :return: The retrieved subfield
        """
        field_split = field_whole.split('.')
        event_field = event
        for field in field_split:
            if len(event_field) > 0:
                event_field = event_field[field]
            else:
                return None
        return event_field

    def _get_field_mismatch(self, event1, event2):
        """
        Gets the fields that do not match between two events
        :param event1:
        :param event2:
        :return: a string of any mismatches
        """
        mismatches = ''
        bad_events_1 = []
        bad_events_2 = []
        for field_name1, field_name2 in self.fields_to_compare.items():
            field1 = self.get_subfield(event1, field_name1)
            field2 = self.get_subfield(event2, field_name2)
            try:
                field2_is_nan = np.isnan(field2)
            except Exception:
                field2_is_nan = False
            if not (field1 is None and field2_is_nan) and field1 != field2:
                if not self.exceptions(event1, event2, field_name1, field_name2):
                    mismatches += '{}/{}: {} vs {}\n'.format(field_name1, field_name2, field1, field2)
                    bad_events_1.append(event1)
                    bad_events_2.append(event2)

        for (ev1, ev2) in zip(bad_events_1, bad_events_2):
            mismatches += '--1--\n{}\n--2--\n{}'.format(pformat_rec(ev1), pformat_rec(ev2))

        return mismatches

    def compare(self):
        """
        Compares all events in event1 and events2
        :return:
        """
        mismatches = ''

        for i, event1 in enumerate(self.events1):
            this_mask2 = np.logical_and(
                    np.abs(event1[self.match_field] - self.events2[self.match_field]) <= 4,
                    event1['type'] == self.events2['type'])
            if not this_mask2.any():
                continue
            this_mismatch = self._get_field_mismatch(event1, self.events2[this_mask2])
            if this_mismatch:
                mismatches += this_mismatch + '\n'
        return mismatches


class EventCombiner(object):
    """
    Merges separate events into a single structure
    """

    def __init__(self, events, sort_field='mstime'):
        """
        constructor
        :param events: A list of the events to be combined
        :param sort_field: The field that determines the order of the events
        """
        self.events = events
        self.sort_field = sort_field

    @staticmethod
    def get_default(instance):
        """
        Gets the default value for a field based on a single value
        :param instance: the model value
        :return: the default value
        """
        if isinstance(instance, basestring):
            return ''
        elif isinstance(instance, (int, float)):
            return -999
        elif isinstance(instance, (list, tuple, np.ndarray)):
            return []
        elif isinstance(instance, dict):
            return {}

    def combine(self):
        """
        Combines the events that were passed into the constructor
        :return: combined events, sorted by the specified sort_field
        """
        # Convert the events to a dictionary
        all_dict_events = []
        for events in self.events:
            if len(events) == 0:
                continue
            dict_events = to_dict(events)
            # Add fields to each dictionary that don't appear in the other
            if len(all_dict_events) > 0:
                keys = [k for k in dict_events[0].keys() if k not in all_dict_events[0].keys()]
                for key in keys:
                    default = self.get_default(dict_events[0][key])
                    for event in all_dict_events:
                        event[key] = default
                keys = [k for k in all_dict_events[0].keys() if k not in dict_events[0].keys()]
                for key in keys:
                    default = self.get_default(all_dict_events[0][key])
                    for event in dict_events:
                        event[key] = default
            all_dict_events += dict_events

        # Sort them, and return them
        all_dict_events = sorted(all_dict_events, key=lambda d:d[self.sort_field])
        dtypes = self.combine_dtypes([e.dtype for e in self.events])
        return from_dict(all_dict_events, dtypes=dtypes)

    def combine_dtypes(self, dtypes):
        assert len(dtypes) > 0

        dtype_0 = dtypes[0]
        if len(dtypes) == 1:
            return dtype_0
        for dtype in dtypes[1:]:
            if (dtype.names is None) ^ (dtype_0.names is None):
                # pick the named one
                dtype_0 = dtype_0 if dtype_0.names else dtype
                continue
            if dtype.names is None:
                dtype_0 = max(dtype_0, dtype)
                continue
            else:

                type_dict = {}
                nested_type_names = set(n for dt in [dtype_0, dtype] for n in dt.names if dt.fields[n] is not None)
                for name in nested_type_names:
                    if name in dtype_0.names and name in dtype.names:
                        type_dict[name] = self.combine_dtypes([dtype_0[name], dtype[name]])
                    else:
                        type_dict[name] = dtype[name] if name in dtype.names else dtype_0[name]
                flat_names = set(n for dt in [dtype_0,dtype] for n in dt.names if n not in nested_type_names)
                for name in flat_names:
                    if name in dtype_0.names and name in dtype.names:
                        type_dict[name] = max(dtype_0[name], dtype[name])
                    else:
                        type_dict[name] = dtype_0[name] if name in dtype_0.names else dtype[name]
                dtype_0 = np.dtype([x for x in type_dict.iteritems()])
        return dtype_0


def get_version_num(session_log_file):
    """
    Convenience function to get the version number of an experiment
    :param session_log_file: the file in which to look for the session number
    :return: the version number as a float
    """
    with open(session_log_file, 'r') as f:
        for line in f:
            split_line = line.split()
            if split_line[2] == 'SESS_START':
                version = float(split_line[-1].replace('v_', ''))
                return version
