from base_log_parser import BaseSessionLogParser, EventComparator
from fr_log_parser import FRSessionLogParser
from catfr_log_parser import CatFRSessionLogParser
from pal_log_parser import PALSessionLogParser
from math_parser import MathLogParser
import numpy as np
import re
import os
import shutil
import json
import glob
import datetime
from readers import eeg_reader
from ptsa.data.readers.BaseEventReader import BaseEventReader
from submission.transferer import DATA_ROOT, DB_ROOT, RHINO_ROOT, EVENTS_ROOT
from viewers.view_recarray import pformat_rec
from parsers.system2_log_parser import System2LogParser


class BaseMatConverter(object):

    _BASE_FIELD_CONVERSION = {
        'eegfile': 'eegfile',
        'eegoffset': 'eegoffset',
        'mstime': 'mstime',
        'msoffset': 'msoffset',
        'type': 'type'
    }

    _BASE_VALUE_CONVERSION = {}

    _BASE_FIELDS = BaseSessionLogParser._BASE_FIELDS
    _STIM_FIELDS = BaseSessionLogParser._STIM_FIELDS

    @classmethod
    def stim_params_template(cls):
        return BaseSessionLogParser.stim_params_template()

    def __init__(self, protocol, subject, montage, experiment, session, original_session, files,
                 events_type='matlab_events', include_stim_params=False):
        self._protocol = protocol
        self._subject = subject
        self._montage = montage
        self._experiment = experiment
        self._session = session
        self._include_stim_params = include_stim_params
        self._fields = self._BASE_FIELDS

        event_reader = BaseEventReader(filename=str(files[events_type]),
                                       common_root=DB_ROOT)
        mat_events = event_reader.read()
        sess_events = mat_events[mat_events.session == original_session]
        self._mat_events = sess_events

        if include_stim_params:
            self._fields += (self.stim_params_template(),)

        self._field_conversion = self._BASE_FIELD_CONVERSION.copy()
        self._value_converion = self._BASE_VALUE_CONVERSION.copy()
        self._type_conversion = {
            'B': self._skip_event,
            'E': self._skip_event,
            '': self._skip_event
        }

        # Try to read the jacksheet if it is present
        if 'jacksheet' in files:
            jacksheet_contents = [x.strip().split() for x in open(files['jacksheet']).readlines()]
            self._jacksheet = {int(x[0]): x[1].upper() for x in jacksheet_contents}
        else:
            self._jacksheet = None

    def _add_fields(self, *args):
        init_fields = list(self._fields)
        init_fields.extend(args)
        self._fields = tuple(init_fields)

    def _add_field_conversion(self, **kwargs):
        self._field_conversion.update(kwargs)

    @property
    def _reverse_field_conversion(self):
        return {v:k for k,v in self._field_conversion.items()}

    def _add_type_conversion(self, **kwargs):
        self._type_conversion.update(kwargs)

    def _add_value_conversion(self, **kwargs):
        self._value_converion.update(kwargs)

    @property
    def event_template(self):
        return self._fields

    @property
    def _empty_event(self):
        event = BaseSessionLogParser.event_from_template(self.event_template)
        event.protocol = self._protocol
        event.subject = self._subject
        event.montage = self._montage
        event.experiment = self._experiment
        event.session = self._session
        return event

    def convert_fields(self, mat_event):
        py_event = self._empty_event
        for mat_field, py_field in self._field_conversion.items():
            if py_field in mat_event.dtype.names:
                py_event[py_field] = mat_event[mat_field]
        return py_event


    def convert_single_event(self, mat_event):
        if mat_event.type in self._type_conversion:
            py_event = self._type_conversion[mat_event.type](mat_event)
        else:
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
        py_events = self._empty_event
        for mat_event in self._mat_events:
            new_py_event = self.convert_single_event(mat_event)
            if new_py_event:
                py_events = np.append(py_events, new_py_event)
        self.clean_events(py_events.view(np.recarray))
        return py_events[1:]

    def _skip_event(self, mat_event):
        return


    def clean_events(self, events):
        if 'exp_version' in events.dtype.names:
            events.exp_version = re.sub(r'[^\d.]', '', events[10].exp_version)
        for event in events:
            event.eegfile = os.path.basename(event.eegfile)
        return events

    def stim_field_conversion(self, mat_event, py_event):
        raise NotImplementedError


class MatlabEEGExtractor(object):
    EEG_FILE_REGEX = re.compile(r'.*\.[0-9]+$')

    def __init__(self, original_session, files):
        event_reader = BaseEventReader(filename=str(files['matlab_events']),
                                       common_root=DB_ROOT)
        mat_events = event_reader.read()
        sess_events = mat_events[mat_events.session == original_session]
        self._mat_events = sess_events


    def copy_ephys(self, destination):
        eeg_locations = np.unique(self._mat_events.eegfile)
        info = {}
        os.makedirs(os.path.join(destination, 'noreref'))
        for eeg_location in eeg_locations:
            eeg_location = os.path.join(RHINO_ROOT, eeg_location)
            params = self.get_params(eeg_location)
            n_samples = np.nan
            eeg_filenames = glob.glob('{}.*'.format(eeg_location))
            for eeg_filename in eeg_filenames:
                if re.match(self.EEG_FILE_REGEX, eeg_filename):
                    with open(eeg_filename) as eeg_file:
                        data = np.fromfile(eeg_file, params['data_format'])
                        data = data * params['gain']
                        data = data.astype(params['data_format'])
                        out_file = os.path.join(destination, 'noreref', os.path.basename(eeg_filename))
                        data.tofile(out_file)
                        n_samples = len(data)

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
        with open(os.path.join(destination, 'sources.json'), 'w') as source_file:
            json.dump(info, source_file, indent=2, sort_keys=True)


    @staticmethod
    def get_params(eeg_location):
        params = {}
        params_filename = '{}.params.txt'.format(eeg_location)

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

    _FR_VALUE_CONVERSION = {
        'is_stim': { -999: False },
        'recalled': { -999: False },
        'stim_list': { -999: False },
        'type': {'STIM_ON': 'STIM'}
    }

    _FR_FIELDS = FRSessionLogParser._fr_fields()

    FR2_STIM_DURATION = FRSessionLogParser.FR2_STIM_DURATION
    FR2_STIM_PULSE_FREQUENCY = FRSessionLogParser.FR2_STIM_PULSE_FREQUENCY
    FR2_STIM_N_PULSES = FRSessionLogParser.FR2_STIM_N_PULSES
    FR2_STIM_BURST_FREQUENCY = FRSessionLogParser.FR2_STIM_BURST_FREQUENCY
    FR2_STIM_N_BURSTS = FRSessionLogParser.FR2_STIM_N_BURSTS
    FR2_STIM_PULSE_WIDTH = FRSessionLogParser.FR2_STIM_PULSE_WIDTH

    def __init__(self, protocol, subject, montage, experiment, session, original_session, files):
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

            if event.type in ('STIM', 'STIM_ON'):
                last_stim_time = event.mstime
                last_stim_duration = self.FR2_STIM_DURATION
                events.stim_list[events.list == event.list] = True

            if event.mstime <= last_stim_time + last_stim_duration:
                event.is_stim = 1
                event.stim_params['stim_on'] = True
                if last_event:
                    last_event.stim_params = event.stim_params
            else:
                event.stim_params = BaseSessionLogParser.empty_stim_params()

            last_event = event

    def stim_field_conversion(self, mat_event, py_event):
        if self._experiment == 'FR1':
            return

        if self._experiment == 'FR2':
            if np.isnan(mat_event.stimAnode):
                return

            params = {
                'anode_number': mat_event.stimAnode,
                'cathode_number': mat_event.stimCathode,
                'amplitude': mat_event.stimAmp
            }
            params.update(self._fr2_stim_params)

        if self._experiment == 'FR3':
            if np.isnan(mat_event.stimParams.elec1):
                return

            params = {
                'anode_number': mat_event.stimParams.elec1,
                'cathode_number': mat_event.stimParams.elec2,
                'amplitude': mat_event.stimParams.amplitude,
                'burst_freq': mat_event.stimParams.burstFreq,
                'n_bursts': mat_event.stimParams.nBursts,
                'pulse_freq': mat_event.stimParams.pulseFreq,
                'n_pulses': mat_event.stimParams.nPulses,
                'pulse_width': mat_event.stimParams.pulseWidth
            }

            params['stim_duration'] = System2LogParser.get_duration(params)

        BaseSessionLogParser.set_event_stim_params(py_event, self._jacksheet, **params)


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


    def clean_events(self, events):
        super(MathMatConverter, self).clean_events(events)

        for event in events:
            event.version = re.sub(r'[^\d]', '', event.version)

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

            if event.type in ('STIM', 'STIM_ON'):
                last_stim_time = event.mstime
                last_stim_duration = self.CATFR2_STIM_DURATION
                events.stim_list[events.list == event.list] = True

            if event.mstime <= last_stim_time + last_stim_duration:
                event.is_stim = 1
                event.stim_params['stim_on'] = True
                if last_event and last_event.type == 'WORD':
                    last_event.stim_params = event.stim_params
            else:
                event.stim_params = BaseSessionLogParser.empty_stim_params()

            last_event = event

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
        'stimTrial': 'stim_trial',
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
        'study_2': {-999: ''}
    }

    _PAL_FIELDS = PALSessionLogParser._pal_fields()

    def __init__(self, protocol, subject, montage, experiment, session, original_session, files):
        super(PALMatConverter, self).__init__(protocol, subject, montage, experiment, session, original_session, files,
                                              include_stim_params=True)
        self._add_fields(*self._PAL_FIELDS)
        self._add_field_conversion(**self._PAL_FIELD_CONVERSION)
        self._add_value_conversion(**self._PAL_VALUE_CONVERSION)

    def clean_events(self, events):
        super(PALMatConverter, self).clean_events(events)

        last_event = None
        for i, event in enumerate(events):
            word_mask = np.logical_and(events.study_1 == event.study_1, events.list == event.list)

            if event.type in ('STUDY_ORIENT', 'STUDY_PAIR'):
                cue_directions = np.unique(events[word_mask].cue_direction)
                if np.any(cue_directions!=-999):
                    event.cue_direction = cue_directions[cue_directions != -999]

                test_probe = events[np.logical_and(
                    np.logical_or(events.probe_word == event.study_1, events.probe_word == event.study_2),
                    events.type == 'TEST_PROBE')]

                event.probe_word = test_probe.probe_word


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
                study_pair = events[study_mask]
                event.study_1 = study_pair[0].study_1
                event.study_2 = study_pair[0].study_2
                event.vocalization = study_pair.vocalization
                event.intrusion = study_pair.intrusion

                rec_event = events[np.logical_and(events.type == 'REC_EVENT', events.expecting_word == event.expecting_word)]

                if len(rec_event)>0:
                    event.vocalization = rec_event[0].vocalization
                    event.intrusion = rec_event[0].intrusion

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
                event.list = events[i-2].list

            last_event = event

    def stim_field_conversion(self, mat_event, py_event):
        pass

CONVERTERS = {
    'FR': FRMatConverter,
    'catFR': CatFRMatConverter,
    'PAL': PALMatConverter
}

test_sessions = (
    ('R1001P', 'PAL1', 0,),
)

def test_fr_mat_converter():

    global DB_ROOT
    old_db_root = DB_ROOT


    for subject, exp, session in test_sessions:
        DB_ROOT = EVENTS_ROOT
        print subject, exp, session

        mat_file = os.path.join(EVENTS_ROOT, 'RAM_{}'.format(exp[0].upper()+exp[1:]), '{}_events.mat'.format(subject))

        files = {'jacksheet': os.path.join(DATA_ROOT, subject, 'docs', 'jacksheet.txt'),
                 'matlab_events': mat_file}

        converter_type = CONVERTERS[re.sub(r'[\d]','', exp)]
        converter = converter_type('r1', subject, 0.0, exp, session, session, files)
        py_events = converter.convert()

        from viewers.view_recarray import to_json, from_json
        to_json(py_events, open('test_{}_events.json'.format(subject), 'w'))

        new_events = from_json('test_{}_events.json'.format(subject))
        DB_ROOT = old_db_root

        old_events = from_json(os.path.join(DB_ROOT, 'protocols', 'r1', 'subjects', subject, 'experiments', exp,
                                            'sessions', str(session), 'behavioral', 'current_processed', 'task_events.json'))

        compare_converted_events(old_events, new_events)

def compare_converted_events(new_events, old_events):
    for new_event in new_events:
        close_old_events = old_events[np.abs(old_events.mstime - new_event.mstime) < 5]
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


from viewers.view_recarray import pprint_rec as ppr
def compare_single_event(new_event, old_event):
    names = new_event.dtype.names
    is_bad = False
    for name in names:

        if name not in ('eegfile', 'eegoffset', 'mstime', 'rectime', 'msoffset'):
            if isinstance(new_event[name], (np.ndarray, np.record)) and new_event[name].dtype.names:
                compare_single_event(new_event[name], old_event[name])
            elif new_event[name] != old_event[name]:
                if not exceptions(new_event, old_event, name):
                    print '{}: {} vs {}'.format(name, new_event[name], old_event[name])
                    is_bad = True
    return is_bad


def exceptions(new_event, old_event, field):
    if isinstance(new_event[field], basestring):
        if new_event[field].upper() == old_event[field].upper():
            return True

    if new_event.type == 'REC_START':
        return True

    if new_event.type in ('TEST_PROBE', 'TEST_ORIENT') and field in ('vocalization', 'intrusion'):
        return True

    if field == 'RT' and abs(new_event.RT - old_event.RT) < 5:
        return True

    if 'amplitude' in new_event.dtype.names:
        if new_event[field] == 0 or new_event[field] == '':
            return True

    if not 'type' in new_event.dtype.names:
        return False

    if field == 'is_stim' and new_event.type == 'DISTRACT_START':
        return True

    if new_event.type == 'PRACTICE_WORD' and field == 'word':
        return True

    if field == 'exp_version' and new_event['experiment'] == 'PAL1':
        return True

    return False