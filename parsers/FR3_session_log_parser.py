import numpy as np

class BaseSessionLogParser():

    # Index in split line corresponding to these fields
    _MSTIME_INDEX = 0
    _MSOFFSET_INDEX = 1
    _TYPE_INDEX = 2

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

    def __init__(self, filename):
        """
        :param format: ('mstime', 'msoffset', 'type')
        """
        self._file = filename
        self._contents = [line.strip().split() for line in file(filename, 'r').readlines()]
        self._fields = self._BASE_FIELDS
        self._type_to_fn = {
            'B' : self.type_default,
            'E' : self.type_default
        }
        pass

    def add_fields(self, *args):
        """
        Adds fields to events structure
        :param *args: args of format (name, default_value, dtype)
        """
        init_fields = list(self._fields)
        init_fields.extend(args)
        self._fields = tuple(init_fields)

    def add_type_functions(self, **kwargs):
        """
        Adds type->function mapping
        :param kwargs: TYPE = function
        """
        for (key, value) in kwargs.items():
            self._type_to_fn[key] = value

    @staticmethod
    def _event_from_template(template):
        """
        Creates events out of template of type ( (name1, default1, dtype1), (name2, ...), ...)
        :param template:
        :return: recarray of these names, defaults, and types
        """
        defaults = tuple(field[1] for field in template)
        dtypes = [(field[0], field[2]) for field in template]
        return np.rec.array(defaults, dtype=dtypes)

    @property
    def _empty_event(self):
        """
        Returns an event with fieldnames and defaults from self._fields
        :return:
        """
        return self._event_from_template(self._fields)

    def _default_event(self, split_line):
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

    def type_default(self, split_line):
        """
        type->function mapped for events with just 'type' field
        :param split_line:
        :return: event, mod_fn
        """
        event = self._default_event(split_line)
        return event, None

    def make_all_events(self):
        """
        Makes all events for the passed in file
        :return:
        """
        events = self._empty_event
        for split_line in self._contents:
            type = split_line[self._TYPE_INDEX]
            if type in self._type_to_fn:
                new_event, mod_fn = self._type_to_fn[type](split_line)
                events = np.append(events, new_event)
                if mod_fn:
                    # TODO: How to keep pycharm from complaining about this?
                    events = mod_fn(events)
        # Remove first (empty) event
        return events[1:]

class FRSessionLogParser(BaseSessionLogParser):

    _STIM_PARAM_FIELDS = (
        ('hostTime', -1, 'int32'),
        ('elec1', -1, 'int16'),
        ('elec2', -1, 'int16'),
        ('amplitude', -1, 'int16'),
        ('burstFreq', -1, 'int16'),
        ('nBursts', -1, 'int16'),
        ('pulseFreq', -1, 'int16'),
        ('nPulses', -1, 'int16'),
        ('pulseWidth', -1, 'int16')
    )

    @classmethod
    def empty_stim_params(cls):
        """
        Makes a recarray for empty stim params (no stimulation)
        :return:
        """
        return cls._event_from_template(cls._STIM_PARAM_FIELDS)

    @classmethod
    def _fr_fields(cls):
        """
        Returns the template for a new FR field
        Has to be a method because of call to empty_stim_params, unfortunately
        :return:
        """
        return (
            ('list', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('item', 'X', 'S16'),
            ('itemno', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('rectime', -999, 'int16'),
            ('isStim', False, 'b1'),
            ('expVersion', '', 'S16'),
            ('stimList', False, 'b1'),
            ('stimParams', cls.empty_stim_params(), np.object),
        )

    def __init__(self, filename):
        BaseSessionLogParser.__init__(self, filename)
        self._session = -999
        self._list = -999
        self._serialpos = -999
        self._stimList = False
        self._item = ''
        self.add_fields(*self._fr_fields())
        self.add_type_functions(
            INSTRUCT_VIDEO=self.type_instruct_video,
            SESS_START=self.type_sess_start,
            MIC_TEST=self.type_default,
            PRACTICE_TRIAL=self.type_default,
            COUNTDOWN_START=self.type_default,
            COUNTDOWN_END=self.type_default,
            PRACTICE_ORIENT=self.type_default,
            PRACTICE_ORIENT_OFF=self.type_default,
            PRACTICE_WORD=self.type_practice_word,
            PRACTICE_WORD_OFF=self.type_practice_word_off
        )

    def _default_event(self, split_line):
        """
        Override base class's default event to include list, serial position, and stimList
        :param split_line:
        :return:
        """
        event = BaseSessionLogParser._default_event(self, split_line)
        event.list = self._list
        event.session = self._session
        event.serialpos = self._serialpos
        event.stimList = self._stimList
        return event

    def type_instruct_video(self, split_line):
        """
        type->fn mapping for INSTRUCT_VIDEO
        :param split_line:
        :return: event, mod_fn
        """
        event= self._default_event(split_line)
        if split_line[3] == 'ON':
            event.type = 'INSTRUCT_ON'
        else:
            event.type = 'INSTRUCT_OFF'
        return event, None

    def type_sess_start(self, split_line):
        """
        type->fn mapping for SESS_START
        :param split_line:
        :return: event, mod_fn
        """
        self._session = int(split_line[3])
        return self._default_event(split_line), self.apply_session

    def apply_session(self, events):
        """
        mod_fn which applies session to all previous events
        :param events:
        :return:
        """
        for event in events:
            event.session = self._session
        return events

    def type_practice_word(self, split_line):
        """
        type->fn mapping for PRACTICE_WORD
        :param split_line:
        :return: event, mod_fn
        """
        event = self._default_event(split_line)
        self._item = split_line[3]
        event.item = self._item
        return event, None

    def type_practice_word_off(self, split_line):
        """
        type->fn mapping for PRACTICE_WORD_OFF
        :param split_line:
        :return: event, mod_fn
        """
        event = self._default_event(split_line)
        event.item = self._item
        return event, None

def test_fr_parser():
    parser = FRSessionLogParser('../tests/session.log')
    import viewers.view_recarray as disp
    parsed = parser.make_all_events()
    disp.pprint_rec(parsed)
    for i in range(len(parsed)):
        print('*'*10 + str(i) + '*'*10)
        disp.pprint_rec(parsed[i])