
from ptsa.data.readers.BaseEventReader import BaseEventReader
from parsers.fr_log_parser import parse_fr3_session_log
from parsers.base_log_parser import EventComparator
import numpy as np
import os

# {'subject': session_#'}
FR3_SESSIONS = {'R1170J_1': 0}
TEST_DIR = './test_data'


def test_fr3_event_creation():
    def exceptions(event1, event2, field):
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
        if field == 'mstime' and np.abs(event1['mstime'] - event2['mstime']) <= 1:
            return True
        if field == 'wordno' and event1['type'] == 'REC_WORD' and event1['intrusion'] == -1:
            return True
        return False

    for subject, session in FR3_SESSIONS.items():
        py_events = parse_fr3_session_log(subject, session, 'FR3', base_dir=TEST_DIR)

        mat_fr3_events_reader = \
            BaseEventReader(
                    filename=os.path.join(TEST_DIR, subject, 'behavioral', 'FR3', 'session_%d' % session, 'events.mat'),
                    common_root=TEST_DIR)

        mat_events = mat_fr3_events_reader.read()

        comparator = EventComparator(py_events, mat_events,
                                     {'word': 'item', 'wordno': 'itemno'},
                                     ['eegfile', 'eegoffset', 'stimList', 'stimParams'],
                                     exceptions,
                                     ('INSTRUCT_START', 'INSTRUCT_END',
                                      'PRACTICE_REC_START', 'PRACTICE_REC_END',
                                      'SESSION_SKIPPED', 'INSTRUCT_VIDEO')
                                     )
        found_bad, err_msg = comparator.compare()

        assert found_bad is False, err_msg
