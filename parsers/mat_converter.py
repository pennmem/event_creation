from base_log_parser import BaseSessionLogParser
from fr_log_parser import FRSessionLogParser
from catfr_log_parser import CatFRSessionLogParser
from pal_log_parser import PALSessionLogParser
from math_parser import MathLogParser
from ps_log_parser import PSSessionLogParser
import numpy as np
import re
import os
import json
import glob
import datetime
from readers import eeg_reader
from readers.eeg_reader import read_jacksheet
from ptsa.data.readers.BaseEventReader import BaseEventReader
from submission.transferer import DATA_ROOT, RHINO_ROOT, EVENTS_ROOT, DB_ROOT
from viewers.view_recarray import strip_accents
from scipy.io import loadmat

class BaseMatConverter(object):
    """
    Base class to convert .mat files to record arrays
    """

    # These fields get converted regardless of the type of event
    _BASE_FIELD_CONVERSION = {
        'eegfile': 'eegfile',
        'eegoffset': 'eegoffset',
        'mstime': 'mstime',
        'msoffset': 'msoffset',
        'type': 'type'
    }

    # No values get converted automatically
    _BASE_VALUE_CONVERSION = {}

    # The base fields and stim fields are inherited from the BaseLogParser
    _BASE_FIELDS = BaseSessionLogParser._BASE_FIELDS
    _STIM_FIELDS = BaseSessionLogParser._STIM_FIELDS

    @classmethod
    def stim_params_template(cls):
        return BaseSessionLogParser.stim_params_template()

    def __init__(self, protocol, subject, montage, experiment, session, original_session, files,
                 events_type='matlab_events', include_stim_params=False):
        """
        Constructor
        :param protocol: New protocol for the events
        :param subject: New subject...
        :param montage: New montage number...
        :param experiment: New experiment...
        :param session: New session...
        :param original_session: Original session (used to parse old events to specific session)
        :param files: Output of Transferer instance
        :param events_type: key from which to grab events file from transferer output
        :param include_stim_params: Whether to include stimulation parameters in the final record array
        """
        self._protocol = protocol
        self._subject = subject
        self._montage = montage
        self._experiment = experiment
        self._session = session
        self._include_stim_params = include_stim_params
        self._fields = self._BASE_FIELDS

        # Get the matlab events for this specific session
        event_reader = BaseEventReader(filename=str(files[events_type]), common_root=DB_ROOT)
        mat_events = event_reader.read()
        sess_events = mat_events[mat_events.session == original_session]
        self._mat_events = sess_events

        if include_stim_params:
            self._fields += (self.stim_params_template(),)

        self._field_conversion = self._BASE_FIELD_CONVERSION.copy()
        self._value_converion = self._BASE_VALUE_CONVERSION.copy()

        # B, E, and empty event types can be skipped
        self._type_conversion = {
            'B': self._skip_event,
            'E': self._skip_event,
            '': self._skip_event
        }

        # Try to read the jacksheet if it is present
        if 'contacts' in files:
            self._jacksheet = read_jacksheet(files['contacts'])
        elif 'jacksheet' in files:
            self._jacksheet = read_jacksheet(files['jacksheet'])
        else:
            self._jacksheet = None

    def _add_fields(self, *args):
        """
        Adds these fields to the default record array output
        :param args: Fields, like listed in _BASE_FIELDS
        """
        init_fields = list(self._fields)
        init_fields.extend(args)
        self._fields = tuple(init_fields)

    def _add_field_conversion(self, **kwargs):
        """
        :param kwargs: fields of the mat structure with name 'key' will be converted to name 'value'
        """
        self._field_conversion.update(kwargs)

    @property
    def _reverse_field_conversion(self):
        """
        Convenience to reverse the field conversion dictionary
        """
        return {v:k for k,v in self._field_conversion.items()}

    def _add_type_conversion(self, **kwargs):
        """
        :param kwargs Fields with type 'key' are converted to have type 'value'
        """
        self._type_conversion.update(kwargs)

    def _add_value_conversion(self, **kwargs):
        """
        :param kwargs: Fields with type 'key' that have value 'v' are converted to have value 'value'['v']
        """
        self._value_converion.update(kwargs)

    @property
    def event_template(self):
        """
        The template for each event
        :return: The template
        """
        return self._fields

    @property
    def _empty_event(self):
        """
        Outputs the event with fields that will be the same for every event in the session
        :return: the empty event
        """
        event = BaseSessionLogParser.event_from_template(self.event_template)
        event.protocol = self._protocol
        event.subject = self._subject
        event.montage = self._montage
        event.experiment = self._experiment
        event.session = self._session
        return event

    def convert_fields(self, mat_event):
        """
        Converts the fields for a given matlab event to the record array, ignoring special rules
        :param mat_event: the matlab event to convert
        :return: the record array event
        """
        py_event = self._empty_event
        for mat_field, py_field in self._field_conversion.items():
            if mat_field in mat_event.dtype.names:
                if isinstance(mat_event[mat_field], basestring):
                    py_event[py_field] = strip_accents(mat_event[mat_field])
                else:
                    py_event[py_field] = mat_event[mat_field]
        return py_event


    def convert_single_event(self, mat_event):
        """
        Converts a single matlab event to a python event, applying the specified conversions
        :param mat_event: The matlab event
        :return: The record array event
        """
        # Can implement a specific conversion based on type
        if mat_event.type in self._type_conversion:
            py_event = self._type_conversion[mat_event.type](mat_event)
        else:
            # Otherwise it undergoes the default conversion
            py_event = self.convert_fields(mat_event)
            for key, value in self._value_converion.items():
                mat_field = self._reverse_field_conversion[key]
                if isinstance(mat_event[mat_field], np.ndarray):
                    mat_item = mat_event[mat_field].item()
                else:
                    mat_item = mat_event[mat_field]
                if mat_item in value:
                    py_event[key] = value[mat_item]
        if self._include_stim_params and py_event:
            self.stim_field_conversion(mat_event, py_event)
        return py_event

    def convert(self):
        """
        Converts all matlab events to record-array events
        :return: The record array events
        """
        py_events = self._empty_event
        for mat_event in self._mat_events:
            new_py_event = self.convert_single_event(mat_event)
            if new_py_event:
                py_events = np.append(py_events, new_py_event)
        py_events = self.clean_events(py_events.view(np.recarray))
        return py_events[1:]

    def _skip_event(self, mat_event):
        return


    def clean_events(self, events):
        """
        Wound up being the meat of the conversion.
        :param events: Events with type and values converted
        :return: Events modified in whatever custom way. In this case, fixing experiment version and eegfile
        """
        if 'exp_version' in events.dtype.names:
            events.exp_version = re.sub(r'[^\d.]', '', events[10].exp_version)
        for event in events:
            event.eegfile = os.path.basename(event.eegfile)
        return events

    def stim_field_conversion(self, mat_event, py_event):
        """
        Subclasses must implement their own stim field conversion
        """
        raise NotImplementedError


class MatlabEEGExtractor(object):
    """
    Class that reads in EEG files referenced in a matlab file and applies the appropriate gain,
    then saves them out with a gain of 1 and the new parameters file
    """

    # Matches anything that looks like an EEG file
    EEG_FILE_REGEX = re.compile(r'.*\.[0-9]+$')

    def __init__(self, original_session, files):
        """
        Constructor
        :param original_session: The session to reference in the matlab events
        :param files: output of transferer, must include 'matlab_events'
        """
        event_reader = BaseEventReader(filename=str(files['matlab_events']),
                                       common_root=DB_ROOT)
        mat_events = event_reader.read()
        sess_events = mat_events[mat_events.session == original_session]
        self._mat_events = sess_events


    def copy_ephys(self, destination):
        """
        Copies the eeg recordings to the specified destination
        :param destination: The folder in which to place the noreref dir containing the eeg recordings
        """

        # Get the locations of the eeg files
        eeg_locations = np.unique(self._mat_events.eegfile)
        info = {}
        noreref = os.path.join(destination, 'noreref')
        if not os.path.exists(noreref):
            os.makedirs(noreref)

        # For each unique eeg location:
        for eeg_location in eeg_locations:
            eeg_location = os.path.join(RHINO_ROOT, eeg_location)
            # Get the original parameters
            params = self.get_params(eeg_location)
            n_samples = np.nan

            # For each channel file in the folder, copy them to the output
            eeg_filenames = glob.glob('{}.*'.format(eeg_location))
            for eeg_filename in eeg_filenames:
                if re.match(self.EEG_FILE_REGEX, eeg_filename):
                    with open(eeg_filename) as eeg_file:
                        data = np.fromfile(eeg_file, params['data_format'])
                        data = data * params['gain']
                        data = data.astype(params['data_format'])
                        out_file = os.path.join(noreref, os.path.basename(eeg_filename))
                        data.tofile(out_file)
                        n_samples = len(data)

            # Fill out the new parameters
            name = os.path.basename(eeg_location)
            date_str = '_'.join(name.split('_')[-2:])
            date_time = datetime.datetime.strptime(date_str, eeg_reader.EEG_reader.STRFTIME)
            date_ms = int((date_time - eeg_reader.EEG_reader.EPOCH).total_seconds() * 1000)
            info[name] = {
                'data_format': params['data_format'],
                'n_samples': n_samples,
                'name': name,
                'sample_rate': params['sample_rate'],
                'source_file': 'N/A',
                'start_time_ms': date_ms,
                'start_time_str': date_str
            }

        # Output the parameters file
        with open(os.path.join(destination, 'sources.json'), 'w') as source_file:
            json.dump(info, source_file, indent=2, sort_keys=True)


    @staticmethod
    def get_params(eeg_location):
        """
        Gets the parameters for an old eeg file
        :param eeg_location: location in which to find the params.txt file
        :return: dictionary of parameters
        """
        params = {}
        params_filename = '{}.params.txt'.format(eeg_location)
        if not os.path.exists(params_filename):
            params_filename = os.path.join(os.path.dirname(eeg_location), 'params.txt')

        with open(params_filename) as params_file:
            for line in params_file:
                split_line = line.split()
                if split_line[0] == 'samplerate':
                    params['sample_rate'] = float(split_line[1])
                elif split_line[0] == 'dataformat':
                    params['data_format'] = split_line[1].strip("'")
                elif split_line[0] == 'gain':
                    params['gain'] = float(split_line[1])

        return params


class FRMatConverter(BaseMatConverter):
    """
    Specifics for converting FR events
    """

    # These fields appear in the final record array
    _FR_FIELD_CONVERSION = {
        'serialpos': 'serialpos',
        'item': 'word',
        'itemno': 'wordno',
        'recalled': 'recalled',
        'rectime': 'rectime',
        'intrusion': 'intrusion',
        'stimList': 'stim_list',
        'isStim': 'is_stim',
        'expVersion': 'exp_version',
        'list': 'list'
    }

    # These values are converted in the record array (if not done, -999 defaults to True)
    _FR_VALUE_CONVERSION = {
        'is_stim': { -999: False },
        'recalled': { -999: False },
        'stim_list': { -999: False },
    }

    _FR_FIELDS = FRSessionLogParser._fr_fields()

    # FR2 specific fields
    FR2_STIM_DURATION = FRSessionLogParser.FR2_STIM_DURATION
    FR2_STIM_PULSE_FREQUENCY = FRSessionLogParser.FR2_STIM_PULSE_FREQUENCY
    FR2_STIM_N_PULSES = FRSessionLogParser.FR2_STIM_N_PULSES
    FR2_STIM_BURST_FREQUENCY = FRSessionLogParser.FR2_STIM_BURST_FREQUENCY
    FR2_STIM_N_BURSTS = FRSessionLogParser.FR2_STIM_N_BURSTS
    FR2_STIM_PULSE_WIDTH = FRSessionLogParser.FR2_STIM_PULSE_WIDTH

    def __init__(self, protocol, subject, montage, experiment, session, original_session, files):
        """
        Constructor
        :param protocol: New protocol for the events
        :param subject: New subject...
        :param montage: New montage number...
        :param experiment: New experiment...
        :param session: New session...
        :param original_session: Original session (used to parse old events to specific session)
        :param files: Output of Transferer instance
        """
        super(FRMatConverter, self).__init__(protocol, subject, montage, experiment, session, original_session, files,
                                             include_stim_params=True)
        if experiment == 'FR3':
            raise NotImplementedError('FR3 conversion not implemented')

        self._add_fields(*self._FR_FIELDS)
        self._add_field_conversion(**self._FR_FIELD_CONVERSION)
        self._add_value_conversion(**self._FR_VALUE_CONVERSION)

        self._fr2_stim_params = {
            'pulse_freq': self.FR2_STIM_PULSE_FREQUENCY,
            'n_pulses': self.FR2_STIM_N_PULSES,
            'burst_freq': self.FR2_STIM_BURST_FREQUENCY,
            'n_bursts': self.FR2_STIM_N_BURSTS,
            'pulse_width': self.FR2_STIM_PULSE_WIDTH,
            'stim_duration': self.FR2_STIM_DURATION
        }

    def clean_events(self, events):
        """
        For the provided events structure, "cleans" the events, fixing any errors and discrepancies between
        the matlab and the record-array events
        :param events: Events to modify
        :return: modified events
        """
        super(FRMatConverter, self).clean_events(events)

        serialpos = 0
        current_list = -999
        current_word = 'X'
        current_serialpos = -999
        last_stim_time = 0
        last_stim_duration = 0
        last_event = None
        for event in events:
            if 'PRACTICE' in event.type:
                event.list = -1
            if event.type == 'PRACTICE_WORD':
                event.serialpos = serialpos
                serialpos += 1
                event.wordno = -1
            if event.type == 'REC_WORD_VV':
                event.intrusion = -1
                event.list = current_list
                event.msoffset = 20
                event.wordno = -1
            if event.type == 'REC_WORD':
                event.msoffset = 20
            if event.type == 'REC_START':
                event.list = current_list
                event.msoffset = 1
            if event.type == 'WORD_OFF':
                event.word = current_word
            if event.type == 'REC_WORD' and event.intrusion == 0:
                event.recalled = True

            if (event.type in ('REC_WORD', 'WORD_OFF') and (event.wordno in (-1, -999))):
                this_word_events = events[events.word == event.word]
                wordnos = np.unique(this_word_events.wordno)
                wordnos = wordnos[np.logical_and(wordnos != -1, wordnos != -999)]
                if len(wordnos) > 0:
                    event.wordno = wordnos[0]

            if event.type == 'PRACTICE_WORD_OFF':
                event.word = current_word

            if event.type == 'WORD_OFF':
                event.serialpos = current_serialpos


            if event.list != -999:
                current_list = event.list

            if event.word != 'X':
                current_word = event.word

            if event.serialpos > 0:
                current_serialpos = event.serialpos

            if event.type == 'STIM_ON':
                event.stim_params['stim_on'] = True
                last_stim_time = event.mstime
                last_stim_duration = self.FR2_STIM_DURATION
                events.stim_list[events.list == event.list] = True
              #  if last_event:
              #      last_event.stim_params = event.stim_params
              #      last_event.stim_params['stim_on'] = 0

            if event.mstime <= last_stim_time + last_stim_duration:
                event.is_stim = 1
                event.stim_params['stim_on'] = True
            else:
                event.stim_params = BaseSessionLogParser.empty_stim_params()

            last_event = event
        return events

    def stim_field_conversion(self, mat_event, py_event):
        if self._experiment == 'FR1':
            return

        if self._experiment == 'FR2':
            if np.isnan(mat_event.stimAnode) or mat_event.stimAnode == -999:
                return

            params = {
                'anode_number': mat_event.stimAnode,
                'cathode_number': mat_event.stimCathode,
                'amplitude': mat_event.stimAmp
            }
            params.update(self._fr2_stim_params)
            BaseSessionLogParser.set_event_stim_params(py_event, self._jacksheet, **params)

        if self._experiment == 'FR3':
            raise NotImplementedError


class YCMatConverter(BaseMatConverter):

    YC_FIELD_CONVERSION = {
        'item': 'stimulus',
        'itemno': 'stimulus_num',
        'recalled': 'recalled',
        'isStim': 'is_stim',
        'expVersion': 'exp_version',
        'env_size': 'env_size',
        'block': 'block',
        'blocknum': 'block_num',
        'startLocs': 'start_locs',
        'paired_block': 'paired_block',
        'objLocs': 'obj_locs',
        'respLocs': 'resp_locs',
        'respDistErr': 'resp_dist_err',
        'respPerformanceFactor': 'resp_performance_factor',
        'respReactionTime': 'resp_reaction_time',
        'respPathLength': 'resp_path_length',
        'respTravelTime': 'resp_travel_time',
        'Path': 'path',
    }

    PATH_FIELDS = (
        ('x', -999, 'f8'),
        ('y', -999, 'f8'),
        ('direction', -999, 'f8'),
        ('time', -999, 'f8'),
        ('_remove', True, 'b')
    )

    @classmethod
    def empty_path_field(cls):
        return BaseSessionLogParser.event_from_template(cls.PATH_FIELDS)

    def get_yc_fields(self):
        return (
            ('stimulus', '', 'S64'),
            ('stimulus_num', -999, 'int16'),
            ('recalled', False, 'int16'),
            ('is_stim', False, 'int16'),
            ('exp_version', '', 'S64'),
            ('env_size', -999, 'f8', 4),
            ('block', -999, 'int16'),
            ('block_num', -999, 'int16'),
            ('start_locs', -999, 'f8', 2),
            ('paired_block', -999, 'int16'),
            ('obj_locs', -999., 'f8', 2),
            ('resp_locs', -999, 'f8', 2),
            ('resp_dist_err', -999, 'f8'),
            ('resp_performance_factor', -999, 'f8'),
            ('resp_reaction_time', -999, 'f8'),
            ('resp_path_length', -999, 'f8'),
            ('resp_travel_time', -999, 'f8'),
            ('path', self.empty_path_field(), BaseSessionLogParser.dtype_from_template(self.PATH_FIELDS), self.max_path_entries),
        )


    YC2_STIM_DURATION = 5000
    YC2_STIM_PULSE_FREQUENCY = 50
    YC2_STIM_N_PULSES = 250
    YC2_STIM_BURST_FREQUENCY = 1
    YC2_STIM_N_BURSTS = 1
    YC2_STIM_PULSE_WIDTH = 300

    def __init__(self, protocol, subject, montage, experiment, session, original_session, files):
        """
        Constructor
        :param protocol: New protocol for the events
        :param subject: New subject...
        :param montage: New montage number...
        :param experiment: New experiment...
        :param session: New session...
        :param original_session: Original session (used to parse old events to specific session)
        :param files: Output of Transferer instance
        """
        super(YCMatConverter, self).__init__(protocol, subject, montage, experiment, session, original_session, files,
                                             include_stim_params=True)
        filename = str(files['matlab_events'])
        mat_events = loadmat(filename, squeeze_me=True)['events']
        sess_events = mat_events[mat_events['session'] == original_session]
        self.max_path_entries = max([e['Path'].item()[0].shape[0] for e in sess_events])
        self.events_for_path = sess_events

        self._add_fields(*self.get_yc_fields())
        self._add_field_conversion(**self.YC_FIELD_CONVERSION)
        self._add_type_conversion(Path=self.convert_path)

    def stim_field_conversion(self, mat_event, py_event):
        if not mat_event['stimAmp'] > 0:
            return py_event

        stim_params = {
            'pulse_freq': self.YC2_STIM_PULSE_FREQUENCY,
            'n_pulses': self.YC2_STIM_N_PULSES,
            'burst_freq': self.YC2_STIM_BURST_FREQUENCY,
            'n_bursts': self.YC2_STIM_N_BURSTS,
            'pulse_width': self.YC2_STIM_PULSE_WIDTH,
            'stim_duration': self.YC2_STIM_DURATION,
        }

        if isinstance(mat_event['stimAnode'], (int, float)):
            stim_params['anode_number'] = mat_event['stimAnode']
            stim_params['cathode_number'] = mat_event['stimCathode']
        else:
            stim_params['anode_label'] = mat_event['stimAnode']
            stim_params['cathode_label'] = mat_event['stimCathode']

        stim_params['amplitude'] = mat_event['stimAmp']

        stim_params['stim_on'] = mat_event['isStim']

        BaseSessionLogParser.set_event_stim_params(py_event, self._jacksheet, **stim_params)
        return py_event

    def convert_fields(self, mat_event, i):
        """
        Converts the fields for a given matlab event to the record array, ignoring special rules
        :param mat_event: the matlab event to convert
        :return: the record array event
        """
        py_event = self._empty_event
        for mat_field, py_field in self._field_conversion.items():
            if mat_field in mat_event.dtype.names:
                if mat_field == 'Path':
                    self.convert_path(py_event, self.events_for_path[i])
                elif isinstance(mat_event[mat_field], basestring):
                    py_event[py_field] = strip_accents(mat_event[mat_field])
                else:
                    py_event[py_field] = mat_event[mat_field]
        return py_event

    def convert_path(self, py_event, mat_event):
        for name in ('x', 'y', 'direction', 'time'):
            for i, entry in enumerate(mat_event['Path'][name].item()):
                py_event.path[name][i] = entry
                py_event.path._remove[i] = False


    def convert_single_event(self, mat_event, i):
        """
        Converts a single matlab event to a python event, applying the specified conversions
        :param mat_event: The matlab event
        :return: The record array event
        """
        # Can implement a specific conversion based on type
        if mat_event.type in self._type_conversion:
            py_event = self._type_conversion[mat_event.type](mat_event)
        else:
            # Otherwise it undergoes the default conversion
            py_event = self.convert_fields(mat_event, i)
            for key, value in self._value_converion.items():
                mat_field = self._reverse_field_conversion[key]
                if isinstance(mat_event[mat_field], np.ndarray):
                    mat_item = mat_event[mat_field].item()
                else:
                    mat_item = mat_event[mat_field]
                if mat_item in value:
                    py_event[key] = value[mat_item]
        if self._include_stim_params and py_event:
            self.stim_field_conversion(mat_event, py_event)
        return py_event


    def convert(self):
        """
        Converts all matlab events to record-array events
        :return: The record array events
        """
        py_events = self._empty_event
        for i, mat_event in enumerate(self._mat_events):
            new_py_event = self.convert_single_event(mat_event, i)
            if new_py_event:
                py_events = np.append(py_events, new_py_event)
        py_events = self.clean_events(py_events.view(np.recarray))
        return py_events[1:]

class MathMatConverter(BaseMatConverter):

    _MATH_FIELD_CONVERSION = {
        'list': 'list',
        'test': 'test',
        'answer': 'answer',
        'iscorrect': 'iscorrect',
        'rectime': 'rectime',
    }

    _MATH_FIELDS = MathLogParser._math_fields()

    def __init__(self, protocol, subject, montage, experiment, session, original_session, files):
        super(MathMatConverter, self).__init__(protocol, subject, montage, experiment, session, original_session, files,
                                               events_type='math_events', include_stim_params=False)
        self._add_fields(*self._MATH_FIELDS)
        self._add_field_conversion(**self._MATH_FIELD_CONVERSION)

class CatFRMatConverter(BaseMatConverter):

    _CATFR_FIELD_CONVERSION = {
        'list': 'list',
        'serialpos': 'serialpos',
        'item': 'word',
        'itemno': 'wordno',
        'recalled': 'recalled',
        'rectime': 'rectime',
        'intrusion': 'intrusion',
        'isStim': 'is_stim',
        'category': 'category',
        'categoryNum': 'category_num',
        'expVersion': 'exp_version',
        'stimList': 'stim_list'
    }

    _CATFR_FIELDS = CatFRSessionLogParser._catfr_fields()

    _CATFR_VALUE_CONVERSION = {
        'is_stim': {-999: False},
        'recalled': {-999: False},
        'stim_list': {-999: False},
    }

    CATFR2_STIM_DURATION = CatFRSessionLogParser.CATFR2_STIM_DURATION
    CATFR2_STIM_PULSE_FREQUENCY = CatFRSessionLogParser.CATFR2_STIM_PULSE_FREQUENCY
    CATFR2_STIM_N_PULSES = CatFRSessionLogParser.CATFR2_STIM_N_PULSES
    CATFR2_STIM_BURST_FREQUENCY = CatFRSessionLogParser.CATFR2_STIM_BURST_FREQUENCY
    CATFR2_STIM_N_BURSTS = CatFRSessionLogParser.CATFR2_STIM_N_BURSTS
    CATFR2_STIM_PULSE_WIDTH = CatFRSessionLogParser.CATFR2_STIM_PULSE_WIDTH

    def __init__(self, protocol, subject, montage, experiment, session, original_session, files):
        super(CatFRMatConverter, self).__init__(protocol, subject, montage, experiment, session, original_session, files,
                                                include_stim_params=True)
        self._add_fields(*self._CATFR_FIELDS)
        self._add_field_conversion(**self._CATFR_FIELD_CONVERSION)
        self._add_value_conversion(**self._CATFR_VALUE_CONVERSION)

        self._catfr2_stim_params = {
            'pulse_freq': self.CATFR2_STIM_PULSE_FREQUENCY,
            'n_pulses': self.CATFR2_STIM_N_PULSES,
            'burst_freq': self.CATFR2_STIM_BURST_FREQUENCY,
            'n_bursts': self.CATFR2_STIM_N_BURSTS,
            'pulse_width': self.CATFR2_STIM_PULSE_WIDTH,
            'stim_duration': self.CATFR2_STIM_DURATION
        }

    def clean_events(self, events):
        super(CatFRMatConverter, self).clean_events(events)

        last_stim_time = 0
        last_stim_duration = 0
        last_event = None
        for event in events:
            event.word = event.word.upper()

            if event.type == 'REC_WORD':
                pres_mask = np.logical_and(events.word == event.word, events.type == 'WORD')
                if np.any(pres_mask) and np.any(events[pres_mask].list == event.list):
                    pres_event = events[pres_mask]
                    event.serialpos = pres_event.serialpos
                    events.recalled[pres_mask] = True
                    event.recalled = True
                    event.wordno = events[pres_mask].wordno
                elif np.any(pres_mask) and event.list > events[pres_mask].list:
                    event.intrusion = event.list - events[pres_mask].list
                    event.wordno = events[pres_mask].wordno

            if event.type == 'REC_WORD_VV':
                event.wordno = -1
                event.intrusion = -1

            if event.type  == 'STIM_ON':
                last_stim_time = event.mstime
                last_stim_duration = self.CATFR2_STIM_DURATION
                events.stim_list[events.list == event.list] = True

            if event.mstime <= last_stim_time + last_stim_duration:
                event.is_stim = 1
                event.stim_params['stim_on'] = True
            else:
                event.stim_params = BaseSessionLogParser.empty_stim_params()

            last_event = event
        return events

    def stim_field_conversion(self, mat_event, py_event):
        if self._experiment == 'catFR1':
            return

        if self._experiment == 'catFR2':
            if np.isnan(mat_event.stimAnode):
                return

            params = {
                'anode_number': mat_event.stimAnode,
                'cathode_number': mat_event.stimCathode,
                'amplitude': mat_event.stimAmp
            }
            params.update(self._catfr2_stim_params)

            BaseSessionLogParser.set_event_stim_params(py_event, self._jacksheet, **params)


class PALMatConverter(BaseMatConverter):

    _PAL_FIELD_CONVERSION = {
        'stimType': 'stim_type',
        'stimTrial': 'stim_list',
        'stimList': 'stim_list',
        'list': 'list',
        'serialpos': 'serialpos',
        'probepos': 'probepos',
        'study_1': 'study_1',
        'study_2': 'study_2',
        'cue_direction': 'cue_direction',
        'probe_word': 'probe_word',
        'expecting_word': 'expecting_word',
        'resp_word': 'resp_word',
        'correct': 'correct',
        'intrusion': 'intrusion',
        'pass': 'resp_pass',
        'vocalization': 'vocalization',
        'RT': 'RT',
        'isStim': 'is_stim',
        'expVersion': 'exp_version',
    }

    _PAL_VALUE_CONVERSION = {
        'expecting_word': {-999: ''},
        'probe_word': {-999: ''},
        'resp_word': {-999: ''},
        'study_1': {-999: ''},
        'study_2': {-999: ''},
        'is_stim': {-999: 0},
        'stim_type': {'[]': 'NONE',
                      '': 'NONE'},
        'stim_list': {np.nan: 0}
    }

    _PAL_FIELDS = PALSessionLogParser._pal_fields()


    PAL2_STIM_DURATION = PALSessionLogParser.PAL2_STIM_DURATION
    PAL2_STIM_PULSE_FREQUENCY = PALSessionLogParser.PAL2_STIM_PULSE_FREQUENCY
    PAL2_STIM_N_PULSES = PALSessionLogParser.PAL2_STIM_N_PULSES
    PAL2_STIM_BURST_FREQUENCY = PALSessionLogParser.PAL2_STIM_BURST_FREQUENCY
    PAL2_STIM_N_BURSTS = PALSessionLogParser.PAL2_STIM_N_BURSTS
    PAL2_STIM_PULSE_WIDTH = PALSessionLogParser.PAL2_STIM_PULSE_WIDTH

    def __init__(self, protocol, subject, montage, experiment, session, original_session, files):
        super(PALMatConverter, self).__init__(protocol, subject, montage, experiment, session, original_session, files,
                                              include_stim_params=True)
        self._add_fields(*self._PAL_FIELDS)
        self._add_field_conversion(**self._PAL_FIELD_CONVERSION)
        self._add_value_conversion(**self._PAL_VALUE_CONVERSION)

        self._fr2_stim_params = {
            'pulse_freq': self.PAL2_STIM_PULSE_FREQUENCY,
            'n_pulses': self.PAL2_STIM_N_PULSES,
            'burst_freq': self.PAL2_STIM_BURST_FREQUENCY,
            'n_bursts': self.PAL2_STIM_N_BURSTS,
            'pulse_width': self.PAL2_STIM_PULSE_WIDTH,
            'stim_duration': self.PAL2_STIM_DURATION
        }


    def clean_events(self, events):
        super(PALMatConverter, self).clean_events(events)

        last_event = None
        last_stim_time = 0
        last_stim_duration = 0
        for i, event in enumerate(events):
            word_mask = np.logical_and(events.study_1 == event.study_1, events.list == event.list)

            if event.type == 'SESS_START':
                event.resp_pass = 0
                event.stim_list = 0

            if event.type == 'STIM_ON':
                event.list = events[i-1].list

            if event.type == 'REC_EVENT':
                if event.resp_word == 'PASS':
                    event.resp_pass = True

                if event.resp_word == '<>':
                    event.vocalization = True

                study_pair = events[np.logical_and(
                    np.logical_or(events.study_1 == event.resp_word, events.study_2 == event.resp_word),
                    events.type == 'STUDY_PAIR')]

                if len(study_pair) > 0:
                    study_pair = study_pair[0]
                    if event.list - study_pair.list >= 0:
                        event.intrusion = event.list - study_pair.list
                    else:
                        event.intrusion = -1

                this_study_mask = np.logical_and(
                    events.probepos == event.probepos, events.list == event.list
                )
                events.intrusion[this_study_mask] = event.intrusion

                if not event.vocalization:
                    events.vocalization[this_study_mask] = 0




            if event.type in ('STUDY_ORIENT', 'STUDY_PAIR'):
                cue_directions = np.unique(events[word_mask].cue_direction)
                if np.any(cue_directions!=-999):
                    event.cue_direction = cue_directions[cue_directions != -999]

                test_probe_mask = np.logical_and(
                    np.logical_or(events.probe_word == event.study_1, events.probe_word == event.study_2),
                    events.type == 'TEST_PROBE')

                test_probe = events[test_probe_mask]

                event.probe_word = test_probe.probe_word[0]
                event.expecting_word = test_probe.expecting_word[0]

                rec_event_mask = np.logical_and(
                    np.logical_or(events.probe_word == event.study_1, events.probe_word == event.study_2),
                    events.type == 'REC_EVENT')
                rec_events = events[rec_event_mask]

                if np.any(rec_events.vocalization != -999):
                    if np.any(rec_events.vocalization == 1):
                        event.vocalization = 1
                    else:
                        event.vocalization = 0
                else:
                    event.vocalization = 0

                if np.any(rec_events.intrusion != -999):
                    if np.any(rec_events.intrusion != 1):
                        event.intrusion = 0
                    else:
                        event.intrusion = 1
                else:
                    event.vocalization = 0

                if np.any(rec_events.resp_word == 'PASS'):
                    event.resp_pass = 1




            if event.type == 'STUDY_ORIENT':
                study_pair = np.logical_and(word_mask, events.type == 'STUDY_PAIR')
                event.vocalization = events[study_pair].vocalization
                event.intrusion = events[study_pair].intrusion

            if event.type in ('REC_END', 'REC_START'):
                event.list = last_event.list

            if event.type in ('TEST_PROBE', 'TEST_ORIENT'):
                study_mask = np.logical_and(events.type == 'STUDY_PAIR',
                                            np.logical_or(events.study_1 == event.expecting_word,
                                                          events.study_2 == event.expecting_word))
                if np.any(study_mask):
                    study_pair = events[study_mask]
                    event.study_1 = study_pair[0].study_1
                    event.study_2 = study_pair[0].study_2
                    event.vocalization = study_pair.vocalization
                    event.intrusion = study_pair.intrusion

                rec_event = events[np.logical_and(events.type == 'REC_EVENT', events.expecting_word == event.expecting_word)]

                if len(rec_event)>0:
                    event.vocalization = rec_event[0].vocalization
                    event.intrusion = rec_event[0].intrusion

                if np.any(rec_event.resp_word == 'PASS'):
                    event.resp_pass = 1

            if event.type == 'REC_START':
                next_event = events[i+1]
                if next_event.type == 'REC_EVENT':
                    event.resp_word = next_event.resp_word
                    event.RT = next_event.RT
                    event.serialpos = next_event.serialpos
                    event.correct = next_event.correct
                    event.vocalization = next_event.vocalization
                    event.intrusion = next_event.intrusion

            if event.type == 'REC_END':
                last_test_probe = events[:i][events[:i].type == 'TEST_PROBE'][-1]
                event.serialpos = last_test_probe.serialpos


            if event.type in ('MATH_START', 'MATH_END'):
                event.list = events[i-1].list

            if event.type in ('TEST_START', 'MATH_START', 'MATH_END'):
                event.resp_pass = 0


            last_event = event

        for i, event in enumerate(events):

            if event.type == 'STIM_ON':
                last_stim_time = event.mstime
                last_stim_duration = self.PAL2_STIM_DURATION
                events.stim_list[events.list == event.list] = True
                for name in ('resp_word', 'probe_word', 'probepos', 'cue_direction', 'resp_pass', 'RT', 'correct',
                             'study_1', 'study_2', 'vocalization', 'intrusion', 'list', 'expecting_word', 'serialpos'):
                    event[name] = events[i+1][name]

                if 'TEST' in events[i-1].type:
                    test_mask = np.logical_and(events.probepos == events[i-1].probepos, events.list == events[i-1].list)
                    for type in ('TEST_ORIENT', 'TEST_PROBE', 'REC_EVENT'):
                        type_mask = np.logical_and(events.type == type, test_mask)
                        events.is_stim[type_mask] = True
                        events.stim_params[type_mask] = event.stim_params
                        events.stim_params[type_mask]['stim_on'] = False

                if 'STUDY' in events[i-1].type:
                    test_mask = np.logical_and(events.probepos == events[i-1].probepos, events.list == events[i-1].list)
                    for type in ('STUDY_ORIENT', 'STUDY_PAIR'):
                        type_mask = np.logical_and(events.type == type, test_mask)
                        events.is_stim[type_mask] = True
                        events.stim_params[type_mask] = event.stim_params
                        events.stim_params[type_mask]['stim_on'] = False

            if event.mstime <= last_stim_time + last_stim_duration:
                event.is_stim = 1
                event.stim_params['stim_on'] = True

            if event.is_stim == 0:
                event.stim_params['_remove'] = True

        return events

    def stim_field_conversion(self, mat_event, py_event):
        if self._experiment == 'PAL1':
            return

        if self._experiment == 'PAL2':


            if 'stimLoc' in mat_event.dtype.names:
                if np.isnan(mat_event.stimLoc[0]):
                    return
                params = {
                    'anode_number': mat_event.stimLoc[0],
                    'cathode_number': mat_event.stimLoc[1],
                    'amplitude': mat_event.stimAmp
                }
            else:
                if np.isnan(mat_event.stimAnode):
                    return
                params = {
                    'anode_number': mat_event.stimAnode,
                    'cathode_number': mat_event.stimCathode,
                    'amplitude': mat_event.stimAmp
                }

            params.update(self._fr2_stim_params)
            BaseSessionLogParser.set_event_stim_params(py_event, self._jacksheet, **params)

        if self._experiment == 'PAL3':
            raise NotImplementedError


class PSMatConverter(BaseMatConverter):

    _PS_FIELD_CONVERSION = {
        'ADs_present': 'ad_observed',
        'expVersion': 'exp_version',
    }

    _PS_VALUE_CONVERSION = {
        'type': {'STIMULATING': 'STIM_ON'},
    }

    _PS_FIELDS = PSSessionLogParser._ps_fields()

    _PULSE_WIDTH = 300

    def __init__(self, protocol, subject, montage, experiment, session, original_session, files):
        super(PSMatConverter, self).__init__(protocol, subject, montage, experiment, session, original_session, files,
                                             include_stim_params = True)
        self._add_fields(*self._PS_FIELDS)
        self._add_field_conversion(**self._PS_FIELD_CONVERSION)
        self._add_value_conversion(**self._PS_VALUE_CONVERSION)

    def clean_events(self, events):
        super(PSMatConverter, self).clean_events(events)
        exp_version = np.unique(events.exp_version)
        exp_version = [e for e in exp_version if e != '']
        if len(exp_version) == 0:
            exp_version = '1.0'

        events.exp_version = exp_version
        events.protocol = self._protocol
        events.montage = self._montage
        events.experiment = self._experiment
        events.exp_version = exp_version

        mstime_diff = events[10].mstime - events[1].mstime
        eegoffset_diff = events[10].eegoffset - events[1].eegoffset
        eeg_rate = float(eegoffset_diff)/float(mstime_diff)

        stim_mask = events.type == 'STIM_ON'
        stim_off_events = events[stim_mask]
        stim_off_events.type = 'STIM_OFF'
        durations = stim_off_events.stim_params.stim_duration[:,0]
        burst_freqs = stim_off_events.stim_params.burst_freq[:,0]
        pulse_freqs = stim_off_events.stim_params.pulse_freq[:,0]
        n_pulses = stim_off_events.stim_params.n_pulses[:,0]
        durations = np.array([d if b == 1 else (d+n*p) for b, d, p, n in zip(burst_freqs, durations, pulse_freqs, n_pulses)])


        stim_off_events.mstime = stim_off_events.mstime + durations
        stim_off_events.eegoffset += np.array(durations * eeg_rate, dtype=int)
        stim_off_events.is_stim = 0
        stim_off_events.stim_params.stim_on = 0

        events = np.insert(events, np.where(stim_mask)[0], stim_off_events)
        return events

    def stim_field_conversion(self, mat_event, py_event):


        if py_event.type in ('STIM_ON', 'STIM_SINGLE_PULSE', 'BEGIN_BURST'):
            py_event.is_stim = True
        else:
            py_event.is_stim = False



        if 'stimDuration' in mat_event.dtype.names:
            py_event.exp_version = '2.0'
            stim_duration = mat_event.stimDuration
            n_bursts = mat_event.nBursts if mat_event.nBursts > 0 else 1
            pulse_freq = mat_event.pulse_frequency if mat_event.pulse_frequency > 0 else 1
            amplitude = mat_event.amplitude * 1000
            n_pulses = mat_event.nPulses
            stim_duration = (float(n_bursts - 1) / (mat_event.burst_frequency if mat_event.burst_frequency > 0 else 1)) + \
                             (float(n_pulses - 1) / pulse_freq) + \
                            (float(self._PULSE_WIDTH/1000000 & 2))
            stim_duration = round(stim_duration * 1000, -1)

        elif py_event.type == 'STIM_SINGLE_PULSE':
            n_pulses = 1
            stim_duration = 1
            pulse_freq = -1
            n_bursts = 1
            amplitude = mat_event.amplitude
        elif py_event.type == 'BEGIN_BURST':
            n_pulses = mat_event.pulse_duration * mat_event.pulse_frequency / 1000
            n_bursts = mat_event.nBursts
            stim_duration = int(float(mat_event.nBursts) / mat_event.burst_frequency * 1000)
            pulse_freq = mat_event.pulse_frequency
            py_event.type = 'STIM_ON'
            amplitude = mat_event.amplitude
        else:
            n_pulses = (mat_event.pulse_frequency * mat_event.pulse_duration) / 1000
            n_bursts = mat_event.nBursts if mat_event.nBursts != -999 else 1
            stim_duration = mat_event.pulse_duration * n_bursts
            pulse_freq = mat_event.pulse_frequency
            amplitude = mat_event.amplitude

        if not py_event.type in ('STIM_ON', 'STIM_OFF', 'STIM_SINGLE_PULSE', 'BEGIN_BURST'):
            return

        params = {
            'anode_label': mat_event.stimAnodeTag,
            'cathode_label': mat_event.stimCathodeTag,
            'amplitude': amplitude,
            'pulse_freq': pulse_freq,
            'burst_freq': mat_event.burst_frequency if mat_event.burst_frequency > 0 else 1,
            'n_bursts': n_bursts,
            'n_pulses': n_pulses,
            'stim_duration': stim_duration,
            'pulse_width': self._PULSE_WIDTH,
            'stim_on': py_event.is_stim
        }
        BaseSessionLogParser.set_event_stim_params(py_event, self._jacksheet, **params)


CONVERTERS = {
    'FR': FRMatConverter,
    'catFR': CatFRMatConverter,
    'PAL': PALMatConverter,
    'PS': PSMatConverter,
    'YC': YCMatConverter
}

test_sessions = (
    #('R1001P', ('FR1',), (0,)),
    # ('R1082N', ('PAL2',), (0,)),
    # ('R1050M', ('PAL2',), (0,)),
    # ('R1100D', ('PAL1',), (1,)),
    # ('R1001P', ('PAL1',), (0,)),
    # ('R1016M', ('catFR2',), (0,)),
    # ('R1050M', ('catFR2',), (0,)),
    # ('R1004D', ('catFR1',), (0,)),
    # ('R1041M', ('catFR1',), (0,)),
    # ('R1060M', ('catFR1',), (0,)),
    # ('R1101T', ('FR2',), (0,)),
    # ('R1001P', ('FR2',), (1,)),
    # ('R1060M', ('FR1',), (0,)),
    ('R1001P', ('YC1',), (0, )),
)

def test_fr_mat_converter():

    global DB_ROOT
    old_db_root = DB_ROOT


    for subject, exp, session in test_sessions:
        orig_exp = exp[0]
        orig_sess = session[0]
        new_exp = exp[1] if len(exp)>1 else exp[0]
        new_sess = session[1] if len(session)>1 else session[0]

        DB_ROOT = EVENTS_ROOT
        print subject, exp, session

        mat_file = os.path.join(EVENTS_ROOT, 'RAM_{}'.format(orig_exp[0].upper()+orig_exp[1:]), '{}_events.mat'.format(subject))

        files = {'jacksheet': os.path.join(DATA_ROOT, subject, 'docs', 'jacksheet.txt'),
                 'matlab_events': mat_file}

        converter_type = CONVERTERS[re.sub(r'[\d]','', new_exp)]
        converter = converter_type('r1', subject, 0.0, new_exp, new_sess, orig_sess, files)
        py_events = converter.convert()

        from viewers.view_recarray import to_json, from_json
        to_json(py_events, open('test_{}_events.json'.format(subject), 'w'))

        new_events = from_json('test_{}_events.json'.format(subject))
        DB_ROOT = old_db_root

        old_events = from_json(os.path.join(DB_ROOT, 'protocols', 'r1', 'subjects', subject, 'experiments', new_exp,
                                            'sessions', str(new_sess), 'behavioral', 'current_processed', 'task_events.json'))

        compare_converted_events(old_events, new_events)

def compare_converted_events(new_events, old_events):
    for new_event in new_events:
        close_old_events = old_events[np.abs(old_events.eegoffset - new_event.eegoffset) < 15]
        matching_old_event = close_old_events[close_old_events.type == new_event.type]
        if len(matching_old_event) == 0:
            print 'MISSING EVENT! {}'.format(new_event.type)
            continue
        matching_old_event = matching_old_event[0]
        if compare_single_event(new_event, matching_old_event):
            print '******1*****'
            ppr(new_event)
            print '******2*****'
            ppr(matching_old_event)
            print '------------'
        else:
            print '.',


from viewers.view_recarray import pprint_rec as ppr
def compare_single_event(new_event, old_event):
    names = new_event.dtype.names
    is_bad = False
    for name in names:

        if name not in ('eegfile', 'eegoffset', 'mstime', 'rectime', 'msoffset', 'session'):
            if isinstance(new_event[name], (np.ndarray, np.record)) and new_event[name].dtype.names:
                is_bad = is_bad or compare_single_event(new_event[name], old_event[name])
            elif new_event[name] != old_event[name]:
                if not exceptions(new_event, old_event, name):
                    print '{}: {} vs {}'.format(name, new_event[name], old_event[name])
                    is_bad = True
    return is_bad


def exceptions(new_event, old_event, field):
    if isinstance(new_event[field], basestring):
        if new_event[field].upper() == old_event[field].upper():
            return True
    if not 'type' in new_event.dtype.names:
        return False

    if new_event.type in ('REC_START', 'REC_END'):
        return True

    if new_event.type in ('TEST_PROBE', 'TEST_ORIENT') and field in ('vocalization', 'intrusion'):
        return True

    if field == 'RT' and abs(new_event.RT - old_event.RT) < 5:
        return True

    if 'amplitude' in new_event.dtype.names:
        if new_event[field] == 0 or new_event[field] == '':
            return True



    if field == 'is_stim' and new_event.type == 'DISTRACT_START':
        return True

    if new_event.type == 'PRACTICE_WORD' and field == 'word':
        return True

    if field == 'exp_version' and new_event['experiment'] == 'PAL1':
        return True

    if field in ('vocalization', 'intrusion') and new_event.type in ('STIM_ON', 'STUDY_PAIR', 'STUDY_ORIENT'):
        return True

    if field == 'stim_type' and new_event['stim_type'] == '' and old_event['stim_type'] == 'NONE':
        return True

    return False


if __name__ == '__main__':
    test_fr_mat_converter()