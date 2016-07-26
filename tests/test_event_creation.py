
from ptsa.data.readers.BaseEventReader import BaseEventReader
from parsers.fr_log_parser import fr_log_parser_wrapper
from parsers.pal_log_parser import pal_log_parser_wrapper
from parsers.math_parser import math_parser_wrapper
from parsers.catfr_log_parser import catfr_log_parser_wrapper
from parsers.ltpfr_log_parser import ltpfr_log_parser_wrapper
from parsers.base_log_parser import EventComparator
from alignment.system2 import System2Aligner
from alignment.system1 import System1Aligner
import numpy as np
import os
import re
from viewers.view_recarray import pprint_rec, to_json, from_json

# {'subject': session_#'}
FR3_SESSIONS = (('R1163T', 0),
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

DATA_ROOT = '/Volumes/rhino_mount/data/eeg/'

PAL_SESSIONS = (  # ('R1175N', 0),
    ('R1028M', 0),
    ('R1003P', 0)
)
CATFR_SYS1_SESSIONS = (('R1016M', 0, 'R1016M_20Jan15_1032.128.129.catFR1_0.sync.txt'),
                       ('R1028M', 0, 'R1028M_27Feb15_1109.066.067.catFR1_0.sync.txt')
                      )

LTPFR_SESSIONS = (('LTP107', 0),
                  ('LTP108', 0)
                  )
FR1_SYS1_SESSIONS = (#('R1001P', 1, 'R1001P_13Oct14_1457.062.063.FR1_1.sync.txt'),
                     #('R1006P', 0, 'R1006P_11Feb15_1503.072.129.FR1_0.sync.txt'),
                     #('R1026D', 0, 'R1026D_22Feb15_0908.074.075.FR1_0.sync.txt'),
                     ('R1083J', 0, 'R1083J_13Sep15_1129.087.088.FR1_0.sync.txt'),
                     ('R1083J', 1, 'R1083J_14Sep15_1429.087.088.sync.txt'),
                     ('R1083J', 1, 'R1083J_14Sep15_1429.087.088.FR1_1.sync.txt'),
                     #('R1083J', 2, 'R1083J_15Sep15_1130.087.088.FR1_2.sync.txt')
                    )
EEG_FILE_REGEX = r'R1[0-9]{3}[A-Z]_\d{2}[A-z]{3}\d{2}_\d{4}'

def xtest_to_json():

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


def xtest_from_json():
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
    arr = from_json('test_file.json')

    #for key in df:
    #    if type(df[key][0] == dict):
    #        arr[key] = pd.DataFrame(arr[key].tolist()).to_records(False)

    pprint_rec(arr)

    comparator = EventComparator(py_events, arr)
    found_bad, error_msg = comparator.compare()
    assert found_bad is False, error_msg

def event_exceptions(event1, event2, field, parent_field=None):
    if field == 'eegfile':
        # SESS_END might come just after the recording has stopped, so it shouldn't have an eeg file
        if event1['type'] == 'SESS_END': #
            return True

        # Comparing only eegfile basename because it may be stored in a different location
        try:
            basename1 = os.path.basename(event1['eegfile'])
        except AttributeError:
            basename1 = os.path.basename(event1['eegfile'][0])

        try:
            basename2 = os.path.basename(event2['eegfile'])
        except AttributeError:
            basename2 = os.path.basename(event2['eegfile'][0])
        return basename1 == basename2

    # there is an allowable difference of up to 5 ms for eeg offset between new and old events
    if field == 'eegoffset' and abs(event1['eegoffset'] - event2['eegoffset']) <= 5:
        return True

    # There is an allowable difference of up to 1.5 ms for mstime between new and old events
    if field == 'mstime' and np.abs(event1['mstime'] - event2['mstime']) <= 1.5:
        return True

    # There is an allowable difference of up to 1ms for recall time between old and new events
    if field == 'rectime' and np.abs(event1['rectime'] - event2['rectime']) <= 1:
        return True

    # stimParams do not matter if it is not a STIM or STIM_OFF event
    # stimParams are propagated through events during stimulation in the new events
    if parent_field == 'stimParams' and event1['type'] != 'STIM' and event1['type'] != 'STIM_OFF':
        return True

    # Recalled is limited to True and False in new events, and is set to False where old events were -999
    if field == 'recalled' and event1['recalled'] == False and event2['recalled'] == -999:
        return True

    # stimList is limited to True and False in new events, and is set to False where old events were -999
    if field == 'stimList' and event1['stimList'] == False and event2['stimList'] == -999:
        return True

    # For practice list events, old events had list improperly as -999 instead of -1
    if field == 'list' and event1['list'] == -1 and event2['list'] == -999:
        return True

    # Word Number is still not filled in for practice events, but is now -1 (like intrusions) rather than -999
    if field == 'wordno' and event1['list'] == -1 and event1['wordno'] == -1:
        return True

    # Serial position was not properly recorded for practice list in old events
    if field == 'serialpos' and event1['list'] == -1 and event2['serialpos'] == -999:
        return True

    # Word during word_off was not recorded for practice lists in old events
    if field == 'word' and event1['type'] == 'PRACTICE_WORD_OFF' and event1['list'] == -1:
        return True

    # Word, word number, serial position, rectime, and recalled were not marked for old word_off events
    if (field in ['word', 'wordno', 'serialpos', 'rectime', 'recalled']) and event1['type'] == 'WORD_OFF':
        return True

    # msoffset was recorded as -999 in old events for REC_START
    if field == 'msoffset' and event1['type'] == 'REC_START' and event2['msoffset'] == -999:
        return True

    # In old vocalization events, wordno, intrusion, msoffset, and mstime were -999
    # In new vocalization events, wordno and intrusion are -1, and msoffset and mstime are accurate
    if field in ('wordno', 'intrusion', 'msoffset', 'mstime') and event1['type'] == 'REC_WORD_VV':
        return True

    # In old Recall events, expVersion, serial position, recalled, and msoffset were -999 for REC_word events
    # SP is now set to true serial position, and recalled is set to true
    if field in ('expVersion', 'serialpos', 'recalled', 'msoffset') and event1['type'] == 'REC_WORD':
        return True

    # In old recall events, wordNo was not set for recalls that were XLIs but still in the wordpool
    if field == 'wordno' and event1['type'] == 'REC_WORD' and event1['intrusion'] == -1 and event2['intrusion'] == -1:
        return True

    # STIM events had an msoffset of 0, now have an msoffset of -1 to signify that offset is unknown
    if field == 'msoffset' and event1['type'] == 'STIM':
        return True

    # stimList was not set properly on stim lists for old REC events
    if field == 'stimList' and event1['type'] in ('REC_START', 'REC_WORD', 'REC_WORD_VV') \
            and event1['stimList'] == True and event2['stimList'] == -999:
        return True

    # During old stim events, word, word number, and serial position did not propagate through stim events
    if field in ('wordno', 'word', 'serialpos') and event1['type'] == 'STIM' and event2['wordno'] == -999:
        return True

    # Stimulation events that occurred before the start of the session were not recorded in old events
    if field is None and event2 is None and event1['type'] == 'STIM' and event1['eegoffset'] == -1:
        return True

    return False

parsers = {
    'FR': fr_log_parser_wrapper,
    'PAL': pal_log_parser_wrapper,
    'catFR': catfr_log_parser_wrapper,
    'math': math_parser_wrapper
}

def check_event_creation(subject, experiment, session, is_sys2, comparator_inputs,
                         task_pulse_file=None, eeg_pulse_file=None,
                         eeg_file_stem=None):
    print 'Testing: %s, %s: %d' % (subject, experiment, session)

    parser_type = parsers[re.sub(r'\d', '', experiment)]
    parser = parser_type(subject, session, experiment, base_dir=DATA_ROOT)

    print 'Parsing events...'
    unaligned_events = parser.parse()

    print 'Aligning events...'
    if is_sys2:
        aligner = System2Aligner(subject, experiment, session, unaligned_events, False, DATA_ROOT)
        aligner.add_stim_events(parser.event_template, parser.persist_fields_during_stim)
        py_events = aligner.align(parser.START_EVENT)
    else:
        aligner = System1Aligner(unaligned_events, task_pulse_file, eeg_pulse_file, eeg_file_stem, DATA_ROOT)
        py_events = aligner.align()

    print 'Loading mat events...'
    mat_events_reader = \
        BaseEventReader(
            filename=os.path.join(DATA_ROOT, subject, 'behavioral', experiment, 'session_%d' % session, 'events.mat'),
            common_root=DATA_ROOT
        )
    mat_events = mat_events_reader.read()

    print 'Comparing...'
    comparator = EventComparator(py_events, mat_events, **comparator_inputs)
    found_bad, error_message = comparator.compare()

    if found_bad is True:
        assert False, error_message
    else:
        print 'Comparison Success!'

FR3_COMPARATOR_INPUTS = dict(
    field_switch={'word': 'item', 'wordno': 'itemno'},
    field_ignore=('expVersion'),
    exceptions = event_exceptions,
    type_ignore=('INSTRUCT_START','INSTRUCT_END',
                 'PRACTICE_REC_START', 'PRACTICE_REC_END',
                 'SESSION_SKIPPED', 'INSTRUCT_VIDEO', 'STIM_OFF')
)

def test_fr3_event_creation():
    for subject, session in FR3_SESSIONS:
        yield check_event_creation, subject, 'FR3', session, True, FR3_COMPARATOR_INPUTS

def xtest_fr3_event_creation():

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
                                     exceptions=fr3_exceptions,
                                     type_ignore=('INSTRUCT_START', 'INSTRUCT_END',
                                      'PRACTICE_REC_START', 'PRACTICE_REC_END',
                                      'SESSION_SKIPPED', 'INSTRUCT_VIDEO','STIM_OFF')
                                     )
        print 'Verifying %s:%d' % (subject, session)
        found_bad, err_msg = comparator.compare()

        assert found_bad is False, err_msg

def xtest_read_json():
    import json
    from viewers.view_recarray import mkdtype, copy_values
    d = json.load(open('test_file.json'))
    dt = mkdtype(d[0])
    arr = np.zeros(len(d), dt)
    copy_values(d, arr)
    pprint_rec(arr[0])
    pprint_rec(arr)


def fr1_exceptions(event1, event2, field, parent_field=None):
    if field == 'serialpos' and event1['type'] == 'PRACTICE_WORD' and event2['serialpos'] == -999:
        return True
    if field == 'wordno' and event1['type'] == 'PRACTICE_WORD' and event2['serialpos'] == -999:
        return True
    if field == 'word' and event1['type'] == 'PRACTICE_WORD' and event2['word'] == 'X':
        return True
    if field == 'isStim' and event1['isStim'] == False and event2['isStim'] == -999:
        return True
    if field == 'subject' and event1['subject'] == 'R1001P' and event2['subject'] == 'REO001P':
        return True
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
    if field in ('wordno', 'word', 'serialpos') and event1['type'] == 'STIM' and event2['wordno'] == -999:
        return True
    if field == 'eegfile' and event1['eegfile'] == os.path.basename(event2['eegfile'][0].replace('REO001P', 'R1001P')):
        return True
    if field is None and event2 is None and event1['type'] == 'STIM' and event1['eegoffset'] == -1:
        return True
    return False

def xtest_align_sys1_events():
    exp = 'FR1'

    for subject, session, file in FR1_SYS1_SESSIONS:
        print 'Creating py_events for %s:%d' % (subject, session)

        task_file = os.path.join('/Volumes','rhino_mount','data','eeg',subject,'behavioral',exp,'session_%d'%session, 'eeg.eeglog')
        eeg_file = os.path.join('/Volumes', 'rhino_mount','data','eeg',subject,'eeg.noreref',file)
        FR_parser = fr_log_parser_wrapper(subject, session, exp, base_dir='/Volumes/rhino_mount/data/eeg')
        aligner = System1Aligner(FR_parser.parse(), task_file, eeg_file, re.findall(EEG_FILE_REGEX, eeg_file)[0])
        py_events = aligner.align()

        mat_fr3_events_reader = \
            BaseEventReader(
                    filename=os.path.join(DATA_ROOT, subject, 'behavioral', 'FR1', 'session_%d' % session, 'events.mat'),
                    common_root=DATA_ROOT)

        print 'Loading mat_events for %s:%d' % (subject, session)
        mat_events = mat_fr3_events_reader.read()

        comparator = EventComparator(py_events, mat_events,
                                     field_switch={'word': 'item', 'wordno': 'itemno'},
                                     field_ignore=('expVersion', 'stimList',
                                                   'stimAnode', 'stimAnodeTag',
                                                   'stimCathode', 'stimCathodeTag',
                                                   'stimAmp', 'stimParams', 'stimLoc'),
                                     exceptions=fr1_exceptions,
                                     type_ignore=('INSTRUCT_START', 'INSTRUCT_END',
                                      'PRACTICE_REC_START', 'PRACTICE_REC_END',
                                      'SESSION_SKIPPED', 'INSTRUCT_VIDEO','STIM_OFF',
                                      'B', 'E', 'SESS_END')
                                     )
        print 'Verifying %s:%d' % (subject, session)
        found_bad, err_msg = comparator.compare()

        assert found_bad is False, err_msg

def xtest_pal_event_creation():

    def exceptions(event1, event2, field, parent_field=None):
        if field == 'subject' and event2['subject'] == 'REO001P':
            return True
        if field == 'list' and event1['list'] == -1 and event2['list'] == -999:
            return True
        if field == 'serialpos' and event1['list'] == -1:
            return True
        if field == 'serialpos' and event2['serialpos'] == 0:
            return True
        if field == 'msoffset' and event1['type'] == 'REC_START':
            return True
        if field in ('expVersion', 'serialPos', 'recalled', 'msoffset') and event1['type'] == 'REC_WORD':
            return True
        if field == 'serialpos' and event1['type'] == 'REC_WORD':
            return True
        if field == 'RT' and np.abs(event1['RT'] - event2['RT']) <= 1:
            return True
        if field == 'mstime' and np.abs(event1['mstime'] - event2['mstime']) <= 1.5:
            return True
        if field == 'eegoffset' and abs(event1['eegoffset'] - event2['eegoffset']) <= 5:
            return True
        if field == 'msoffset' and event1['type'] == 'STIM':
            return True
        if field == 'stimList' and event1['type'] in ('REC_START', 'REC_WORD', 'REC_WORD_VV') and event1[
            'stimList'] == True:
            return True
        if field in ('wordno', 'word', 'serialpos') and event1['type'] == 'STIM' and event2['wordno'] == -999:
            return True
        if field is None and event2 is None and event1['type'] == 'STIM' and event1['eegoffset'] == -1:
            return True
        if field == 'study_1' and event1['study_1'] == '' and event2['study_1'] == -999:
            return True
        if field == 'study_2' and event1['study_2'] == '' and event2['study_2'] == -999:
            return True
        if field == 'probe_word' and event1['probe_word'] == '' and event2['probe_word'] == -999:
            return True
        if field == 'expecting_word' and event1['expecting_word'] == '' and event2['expecting_word'] == -999:
            return True
        if field == 'resp_word' and event1['resp_word'] == '' and event2['resp_word'] == -999:
            return True
        if field == 'isStim' and event1['isStim'] == 'False' and event2['isStim'] == -999:
            return True
        if field == 'stimType' and event1['stimType'] == '' and event2['stimType'] == '[]':
            return True
        if field == 'stimType' and event1['stimType'] == '' and event2['stimType'] == 'NONE':
            return True
        if field == 'stimType' and event1['stimType'] == '' and event2['stimType'] == []:
            return True
        if field == 'resp_word' and event1['resp_word'] == '' and event2['resp_word'] == -999:
            return True
        if field == 'intrusion' and event2['intrusion'] != 0:
            return True
        if field == 'stimTrial' and event1['stimTrial'] == False and event2['stimTrial'] == -999:
            return True
        if parent_field == 'stimParams' and event1['type'] != 'STIM' and event1['type'] != 'STIM_OFF':
            return True
        return False

    def persistent_field_fn(event):
        if event['type'] == 'WORD':
            return ('list', 'serialpos', 'word', 'wordno', 'recalled',
                    'intrusion', 'stimList', 'subject', 'session', 'eegfile',
                    'rectime')
        else:
            return ('list', 'serialpos', 'stimList', 'subject', 'session', 'eegfile', 'rectime')

    for subject, session in PAL_SESSIONS:
        print 'Processing %s:%d' % (subject, session)
        PAL_parser = pal_log_parser_wrapper(subject, session, 'PAL1', base_dir=DATA_ROOT)
        py_events = PAL_parser.parse()

        mat_fr3_events_reader = \
            BaseEventReader(
                filename=os.path.join(DATA_ROOT, subject, 'behavioral', 'PAL1', 'session_%d' % session,
                                      'events.mat'),
                common_root=DATA_ROOT)

        print 'Loading mat_events for %s:%d' % (subject, session)
        mat_events = mat_fr3_events_reader.read()

        comparator = EventComparator(py_events, mat_events,
                                     field_switch={'resp_pass': 'pass'},
                                     field_ignore=('eegfile', 'expVersion', 'eegoffset', 'stimLoc', 'stimAmp',
                                                   'stimParams', 'stimAnode', 'stimAnodeTag', 'stimCathode',
                                                   'stimCathodeTag'),
                                     exceptions=exceptions,
                                     type_ignore=('COUNTDOWN_START', 'COUNTDOWN_END', 'B', 'E', 'STIM_PARAMS')
                                     )
        print 'Verifying %s:%d' % (subject, session)
        found_bad, err_msg = comparator.compare()

        assert found_bad is False, err_msg



def xtest_math_event_creation():

    def exceptions(event1, event2, field, parent_field=None):
        if field == 'test' and event1['test'].split(';') == event2['test']:
            return True
        if field == 'test':
            return True
        return False

    def persistent_field_fn(event):
        if event['type']=='WORD':
            return ('list', 'serialpos', 'word', 'wordno', 'recalled',
                    'intrusion', 'stimList', 'subject', 'session', 'eegfile',
                     'rectime')
        else:
            return ('list', 'serialpos', 'stimList', 'subject', 'session', 'eegfile', 'rectime')

    for subject, session in PAL_SESSIONS:
        print 'Processing %s:%d' % (subject, session)
        math_parser = math_parser_wrapper(subject, session, 'PAL2', base_dir=DATA_ROOT)
        py_events = math_parser.parse()


        mat_fr3_events_reader = \
            BaseEventReader(
                    filename=os.path.join(DATA_ROOT, subject, 'behavioral', 'PAL2', 'session_%d' % session,
                                          'MATH_events.mat'),
                    common_root=DATA_ROOT)

        print 'Loading mat_events for %s:%d' % (subject, session)
        mat_events = mat_fr3_events_reader.read()

        comparator = EventComparator(py_events, mat_events,
                                     field_ignore=('eegfile', 'eegoffset'),
                                     type_ignore=('COUNTDOWN_START', 'COUNTDOWN_END', 'B', 'E', 'STIM_PARAMS'),
                                     exceptions=exceptions
                                     )
        print 'Verifying %s:%d' % (subject, session)
        found_bad, err_msg = comparator.compare()

        assert found_bad is False, err_msg


def xtest_catfr_event_creation():

    def exceptions(event1, event2, field, parent_field=None):
        if field == 'rectime' and event1['rectime'] - event2['rectime'] == 1000:
            return True
        if field == 'recalled' and event1['recalled'] == False and event2['recalled'] == -999:
            return True
        if field == 'list' and event1['list'] == -1 and event2['list'] == -999:
            return True
        if field == 'wordno' and event1['list'] == -1:
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
        if field == 'stimList' and event2['stimList'] == -999:
            return True
        if field == 'stimList' and event1['type'] in ('REC_START', 'REC_WORD', 'REC_WORD_VV') and event1['stimList'] == True:
            return True
        if field in ('wordno', 'word', 'serialpos') and event1['type'] == 'STIM' and event2['wordno'] == -999:
            return True
        if field is None and event2 is None and event1['type'] == 'STIM' and event1['eegoffset'] == -1:
            return True
        if field == 'isStim' and event1['isStim'] == False and event2['isStim'] == -999:
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

    for subject, session, sync in CATFR_SYS1_SESSIONS:
        print 'Processing %s:%d' % (subject, session)
        catFR_parser = catfr_log_parser_wrapper(subject, session, 'catFR1', base_dir=DATA_ROOT)
        aligner = System1Aligner(catFR_parser.parse(), task_file, eeg_file, re.findall(EEG_FILE_REGEX, eeg_file)[0])
        py_events = aligner.align()

        mat_fr3_events_reader = \
            BaseEventReader(
                    filename=os.path.join(DATA_ROOT, subject, 'behavioral', 'catFR1', 'session_%d' % session,
                                          'events.mat'),
                    common_root=DATA_ROOT)

        print 'Loading mat_events for %s:%d' % (subject, session)
        mat_events = mat_fr3_events_reader.read()

        comparator = EventComparator(py_events, mat_events,
                                     field_switch={'word': 'item', 'wordno': 'itemno'},
                                     field_ignore=('eegfile', 'eegoffset', 'expVersion', 'stimAnode', 'stimAnodeTag',
                                                   'stimCathode', 'stimCathodeTag', 'stimAmp'),
                                     exceptions=exceptions,
                                     type_ignore=('INSTRUCT_START', 'INSTRUCT_END',
                                                  'SESSION_SKIPPED', 'INSTRUCT_VIDEO','STIM_OFF', 'STIM_PARAMS',
                                                  'B', 'E', 'COUNTDOWN_START')
                                     )
        print 'Verifying %s:%d' % (subject, session)
        found_bad, err_msg = comparator.compare()

        assert found_bad is False, err_msg

def xtest_ltpfr_event_creation():

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
        if field == 'isStim' and event1['isStim'] == False and event2['isStim'] == -999:
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

    for subject, session in LTPFR_SESSIONS:
        print 'Processing %s:%d' % (subject, session)
        ltpFR_parser = ltpfr_log_parser_wrapper(subject, session, '', base_dir=DATA_ROOT)
        py_events = ltpFR_parser.parse()

        mat_ltpfr_events_reader = \
            BaseEventReader(
                    filename=os.path.join(DATA_ROOT, subject, 'session_%d' % session,
                                          'events.mat'),
                    common_root=DATA_ROOT)

        print 'Loading mat_events for %s:%d' % (subject, session)
        mat_events = mat_ltpfr_events_reader.read()

        comparator = EventComparator(py_events, mat_events,
                                     field_switch={'word': 'item', 'wordno': 'itemno'},
                                     field_ignore=('eegfile', 'eegoffset', 'expVersion', 'stimAnode', 'stimAnodeTag',
                                                   'stimCathode', 'stimCathodeTag', 'stimAmp'),
                                     exceptions=exceptions,
                                     type_ignore=('INSTRUCT_START', 'INSTRUCT_END',
                                      'PRACTICE_REC_START', 'PRACTICE_REC_END',
                                      'SESSION_SKIPPED', 'INSTRUCT_VIDEO','STIM_OFF')
                                     )
        print 'Verifying %s:%d' % (subject, session)
        found_bad, err_msg = comparator.compare()

        assert found_bad is False, err_msg
