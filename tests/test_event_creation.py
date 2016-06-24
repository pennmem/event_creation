
from ptsa.data.readers.BaseEventReader import BaseEventReader
from parsers.fr_log_parser import fr_log_parser_wrapper
from parsers.base_log_parser import EventComparator
from alignment.system2 import System2Aligner
import pandas as pd
import numpy as np
import os
from viewers.view_recarray import pprint_rec, to_json

# {'subject': session_#'}
FR3_SESSIONS = (('R1163T', 0),
                ('R1154D', 4),
                ('R1154D', 2),
                ('R1124J_1',0),
                ('R1124J_1',1),
                ('R1145J_1',0),
                ('R1154D', 0),
                ('R1154D', 1),
                ('R1154D', 2),
                ('R1154D', 3),
                ('R1161E', 0),
                ('R1161E', 1),
                ('R1163T', 1),
                ('R1166D', 0),
                ('R1166D', 1),
                ('R1170J_1', 0),
                ('R1170J_1', 1)
                )
DATA_ROOT = '/Volumes/RHINO2/data/eeg/'


def test_to_json():

    def persistent_field_fn(event):
        if event['type']=='WORD':
            return ('list', 'serialpos', 'word', 'wordno', 'recalled',
                    'intrusion', 'stimList', 'subject', 'session', 'eegfile',
                     'rectime')
        else:
            return ('list', 'serialpos', 'stimList', 'subject', 'session', 'eegfile', 'rectime')

    subject = 'R1163T'
    session = 0;
    FR_parser = fr_log_parser_wrapper(subject, session, 'FR3', base_dir=DATA_ROOT)
    aligner = System2Aligner(subject, 'FR3', session, FR_parser.parse(),  False, DATA_ROOT)
    aligner.add_stim_events(FR_parser.event_template, persistent_field_fn)
    py_events = aligner.align('SESS_START')
    print to_json(py_events)


def test_from_json():
    def persistent_field_fn(event):
        if event['type']=='WORD':
            return ('list', 'serialpos', 'word', 'wordno', 'recalled',
                    'intrusion', 'stimList', 'subject', 'session', 'eegfile',
                     'rectime')
        else:
            return ('list', 'serialpos', 'stimList', 'subject', 'session', 'eegfile', 'rectime')

    subject = 'R1163T'
    session = 0;
    FR_parser = fr_log_parser_wrapper(subject, session, 'FR3', base_dir=DATA_ROOT)
    aligner = System2Aligner(subject, 'FR3', session, FR_parser.parse(),  False, DATA_ROOT)
    aligner.add_stim_events(FR_parser.event_template, persistent_field_fn)
    py_events = aligner.align('SESS_START')
    to_json(py_events, open('test_file.json', 'w'))

    df = pd.read_json(open('test_file.json'))

    arr = df.to_records(False)

    #for key in df:
    #    if type(df[key][0] == dict):
    #        arr[key] = pd.DataFrame(arr[key].tolist()).to_records(False)

    pprint_rec(arr)
    for name in arr.dtype.names:
        print name, (arr[name] == py_events[name]).all()


def xtest_fr3_event_creation():

    def exceptions(event1, event2, field, parent_field=None):
        if field == 'recalled' and event1['recalled'] == False and event2['recalled'] == -999:
            return True
        if field == 'list' and event1['list'] == -1 and event2['list'] == -999:
            return True
        if field == 'wordno' and event1['list'] == -1:
            return True
        if field == 'type' and event1['field'] == 'INSTRUCT_START':
            return True
        if field == 'type' and event1['type'] == 'PRACTICE_WORD_OFF':
            return True
        if field == 'serialpos' and event1['list'] == -1:
            return True
        if field == 'word' and event1['list'] == -1:
            return True
        if (field in ['word', 'wordno', 'serialpos', 'rectime', 'recalled']) and event1['type'] == 'WORD_OFF':
            return True
        if field == 'msoffset' and event1['type'] == 'REC_START':
            return True
        if field in ('expVersion', 'serialPos', 'recalled', 'msoffset') and event1['type'] == 'REC_WORD':
            return True
        if field == 'serialpos' and event1['type'] == 'REC_WORD':
            return True
        if field in ('wordno', 'intrusion', 'msoffset', 'mstime') and event1['type'] == 'REC_WORD_VV':
            return True
        if field == 'rectime' and np.abs(event1['rectime'] - event2['rectime']) <= 1:
            return True
        if field == 'mstime' and np.abs(event1['mstime'] - event2['mstime']) <= 1.5:
            return True
        if field == 'wordno' and event1['type'] == 'REC_WORD' and event1['intrusion'] == -1:
            return True
        if field == 'eegoffset' and abs(event1['eegoffset'] - event2['eegoffset']) <= 5:
            return True
        if field == 'msoffset' and event1['type'] == 'STIM':
            return True
        if field == 'stimList' and event1['stimList'] == False and event2['stimList'] == -999:
            return True
        if field == 'stimList' and event1['type'] in ('REC_START', 'REC_WORD', 'REC_WORD_VV') and event1['stimList'] == True:
            return True
        if field in ('wordno', 'word', 'serialpos') and event1['type'] == 'STIM' and event2['wordno'] == -999:
            return True
        if field is None and event2 is None and event1['type'] == 'STIM' and event1['eegoffset'] == -1:
            return True
        if parent_field == 'stimParams' and event1['type'] != 'STIM' and event1['type'] != 'STIM_OFF':
            return True
        return False

    def persistent_field_fn(event):
        if event['type']=='WORD':
            return ('list', 'serialpos', 'word', 'wordno', 'recalled',
                    'intrusion', 'stimList', 'subject', 'session', 'eegfile',
                     'rectime')
        else:
            return ('list', 'serialpos', 'stimList', 'subject', 'session', 'eegfile', 'rectime')

    for subject, session in FR3_SESSIONS:
        print 'Processing %s:%d' % (subject, session)
        FR_parser = fr_log_parser_wrapper(subject, session, 'FR3', base_dir=DATA_ROOT)
        aligner = System2Aligner(subject, 'FR3', session, FR_parser.parse(),  False, DATA_ROOT)
        aligner.add_stim_events(FR_parser.event_template, persistent_field_fn)
        py_events = aligner.align('SESS_START')

        mat_fr3_events_reader = \
            BaseEventReader(
                    filename=os.path.join(DATA_ROOT, subject, 'behavioral', 'FR3', 'session_%d' % session, 'events.mat'),
                    common_root=DATA_ROOT)

        print 'Loading mat_events for %s:%d' % (subject, session)
        mat_events = mat_fr3_events_reader.read()

        comparator = EventComparator(py_events, mat_events,
                                     field_switch={'word': 'item', 'wordno': 'itemno'},
                                     field_ignore=('eegfile', 'expVersion'),
                                     exceptions=exceptions,
                                     type_ignore=('INSTRUCT_START', 'INSTRUCT_END',
                                      'PRACTICE_REC_START', 'PRACTICE_REC_END',
                                      'SESSION_SKIPPED', 'INSTRUCT_VIDEO','STIM_OFF')
                                     )
        print 'Verifying %s:%d' % (subject, session)
        found_bad, err_msg = comparator.compare()

        assert found_bad is False, err_msg