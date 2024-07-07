import pandas as pd
from .base_log_parser import BaseSessionLogParser
from .dtypes import keystroke_fields

# MacOS key code map taken from https://gist.github.com/jfortin42/68a1fcbf7738a1819eb4b2eef298f4f8
KEY_MAP = {
    53: "ESCAPE",
    122: "F1",
    120: "F2",
    99: "F3",
    118: "F4",
    96: "F5",
    97: "F6",
    98: "F7",
    100: "F8",
    101: "F9",
    109: "F10",
    103: "F11",
    111: "F12",
    105: "F13",
    107: "F14",
    113: "F15",
    106: "F16",
    64: "F17",
    79: "F18",
    80: "F19",
    50: "TILDE",
    18: "1",
    19: "2",
    20: "3",
    21: "4",
    23: "5",
    22: "6",
    26: "7",
    28: "8",
    25: "9",
    29: "0",
    27: "MINUS",
    24: "EQUAL",
    51: "BACKSPACE",
    48: "TAB",
    12: "Q",
    13: "W",
    14: "E",
    15: "R",
    17: "T",
    16: "Y",
    32: "U",
    34: "I",
    31: "O",
    35: "P",
    33: "OPEN_BRACKET",  # or OPEN_BRACE
    30: "CLOSE_BRACKET",  # or CLOSE_BRACE
    42: "BACKSLASH",  # or PIPE
    272: "CAPSLOCK",
    0: "A",
    1: "S",
    2: "D",
    3: "F",
    5: "G",
    4: "H",
    38: "J",
    40: "K",
    37: "L",
    41: "COLON",  # or SEMI_COLON
    39: "SIMPLE_QUOTE",  # or DOUBLE_QUOTES
    36: "ENTER",
    257: "SHIFT_LEFT",
    6: "Z",
    7: "X",
    8: "C",
    9: "V",
    11: "B",
    45: "N",
    46: "M",
    43: "LESS_THAN",  # or COMMA
    47: "GREATER_THAN",  # or DOT
    44: "SLASH",  # or QUESTION_MARK
    258: "SHIFT_RIGHT",
    256: "CTRL_LEFT",
    259: "COMMAND_LEFT",
    261: "OPTION_LEFT",  # or ALT
    49: "SPACEBAR",
    260: "COMMAND_RIGHT",
    262: "ALT_GR",
    279: "FN",
    269: "CTRL_RIGHT",
    123: "LEFT",
    125: "DOWN",
    124: "RIGHT",
    126: "UP",
    117: "DEL",
    115: "HOME",
    119: "END",
    116: "PAGE_UP",
    121: "PAGE_DOWN",
    71: "CLEAR",
    83: "PAD_1",
    84: "PAD_2",
    85: "PAD_3",
    86: "PAD_4",
    87: "PAD_5",
    88: "PAD_6",
    89: "PAD_7",
    91: "PAD_8",
    92: "PAD_9",
    82: "PAD_0",
    81: "PAD_EQUAL",
    75: "PAD_DIVIDE",
    67: "PAD_MULTIPLY",
    78: "PAD_SUB",
    69: "PAD_ADD",
    76: "PAD_ENTER",
    65: "PAD_DOT"
}


class UnityKeystrokeParser(BaseSessionLogParser):
    _MSTIME_FIELD = 'timestamp'
    _TYPE_FIELD = 'type'
    _PHASE_TYPE_FIELD = 'phase_type'
    _RAW_KEYSTROKE_TYPES = ['key press/release', 'key/mouse press/release']
    
    def __init__(self, protocol, subject, montage, experiment, session, files, primary_log='session_json',
                 include_stim_params=False):
        if primary_log not in files:
            raise ValueError(f'Primary log "{primary_log}" not in <files>')
        
        self._primary_log = primary_log
        
        BaseSessionLogParser.__init__(self, protocol, subject, montage,
                                      experiment, session, files,
                                      primary_log=primary_log,
                                      allow_unparsed_events=True,
                                      include_stim_params=include_stim_params)
        self._files = files
        self._phase = ''
        self._trial = -999
        
        self._add_fields(*keystroke_fields)
        self._add_type_to_new_event(keystroke=self._parse_keystroke)

    def _get_raw_event_type(self, event_json):
        raw_type = event_json[self._TYPE_FIELD]
        if raw_type in self._RAW_KEYSTROKE_TYPES:
            raw_type = 'keystroke'
        return raw_type
    
    def _read_primary_log(self):
        return self._read_unityepl_log(self._primary_log)
    
    def parse(self):
        return super().parse()

    def _read_unityepl_log(self, filename):
        """Read events from the UnityEPL format (JSON strings separated by
        newline characters).

        :param str filename:

        """
        df = pd.read_json(filename, lines=True)
        df = df.query('type in @self._RAW_KEYSTROKE_TYPES')
        df = pd.concat([df.drop(columns=['data']).reset_index(),
                        pd.json_normalize(df['data']).reset_index()],
                    axis=1)
        del df['index']
        df.rename(columns={'time': self._MSTIME_FIELD}, inplace=True)
        return [e.to_dict() for _, e in df.iterrows()]
    
    def _parse_keystroke(self, event_json):
        event = self._empty_event
        event.mstime = event_json[self._MSTIME_FIELD]
        event.type = 'KEYSTROKE_PRESS' if event_json['is pressed'] else 'KEYSTROKE_RELEASE'
        event.key = KEY_MAP[event_json['key code']]
        event.list = self._trial
        event.trial = self._trial
        event.session = self._session
        event.phase = self._phase
        return event


class PyEPLKeystrokeParser(BaseSessionLogParser):
    _MSTIME_FIELD = 'mstime'
    _TYPE_FIELD = 'type'
    # enforce consistency between experiment versions
    _KEY_REMAP = {'RETURN': 'ENTER',
                  'Logging Begins': 'Key Logging Begins',  # pyEPL keylogs cut off pressed keys with 'Logging Begins' events
                  'Logging Ends': 'Key Logging Ends',
    }
    
    def __init__(self, protocol, subject, montage, experiment, session, files, primary_log='keyboard_keylog',
                 include_stim_params=False):
        if primary_log not in files:
            raise ValueError(f'Primary log "{primary_log}" not in <files>')
        
        self._primary_log = primary_log
        
        BaseSessionLogParser.__init__(self, protocol, subject, montage,
                                      experiment, session, files,
                                      primary_log=primary_log,
                                      allow_unparsed_events=True,
                                      include_stim_params=include_stim_params)
        self._files = files
        self._phase = ''
        self._trial = -999
        
        self._add_fields(*keystroke_fields)
        self._add_type_to_new_event(keystroke=self._parse_keystroke)

    def _get_raw_event_type(self, event_json):
        return 'keystroke'
    
    def _read_primary_log(self):
        return self._read_keylog(self._primary_log)
    
    def parse(self):
        return super().parse()

    def _read_keylog(self, filename):
        """Read events from the keyboard.keylog TSV format from pyEPL.

        :param str filename:

        """
        df = pd.read_csv(filename, delimiter='\t', header=0,
                         names=[self._MSTIME_FIELD, '???', self._TYPE_FIELD, 'key'])
        return [e.to_dict() for _, e in df.iterrows()]
    
    def _parse_keystroke(self, event_json):
        event = self._empty_event
        event.mstime = event_json[self._MSTIME_FIELD]
        if event_json['type'] == 'P':
            event.type = 'KEYSTROKE_PRESS'
        elif event_json['type'] == 'R': 
            event.type = 'KEYSTROKE_RELEASE'
        elif event_json['type'] == 'B': 
            event.type = 'KEYSTROKE_LOG_BEGIN'
        elif event_json['type'] == 'E':
            event.type = 'KEYSTROKE_LOG_END'
        else:
            raise ValueError(f'Unrecognized keystroke keylog event type found: {event_json["type"]}')
        
        event.key = self._KEY_REMAP[event_json['key']] if event_json['key'] in self._KEY_REMAP else event_json['key']
        event.list = self._trial
        event.trial = self._trial
        event.session = self._session
        event.phase = self._phase
        return event
