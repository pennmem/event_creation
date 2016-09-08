from readers.eeg_reader import get_eeg_reader
from submission.transferer import DATA_ROOT
from ptsa.data.readers import BaseEventReader
import glob
import json
import os
from pprint import pprint

def get_matching_start_strings(subject, start_string):

    filenames = glob.glob(os.path.join(DATA_ROOT, subject, 'raw', '*', '*.edf')) + \
                glob.glob(os.path.join(DATA_ROOT, subject, 'raw', '*', '*.EEG')) + \
                glob.glob(os.path.join(DATA_ROOT, subject, 'raw', '*', '*.EDF'))

    start_strings = []
    for filename in filenames:
        reader = get_eeg_reader(filename)
        start_strings.append((filename, reader.get_start_time_string()))
        if reader.get_start_time_string() == start_string:
            return os.path.dirname(filename), reader.get_start_time_string()
    return start_string, start_strings

def find_matching_behavioral_dir(subject, experiment, mstime_start):

    behavioral_dirs = glob.glob(os.path.join(DATA_ROOT, subject, 'behavioral', experiment, 'session_*'))
    diffs = []
    for behavioral_dir in behavioral_dirs:
        session_log = os.path.join(behavioral_dir, 'session.log')
        if os.path.exists(session_log):
            for line in open(session_log):
                this_mstime = int(line.split()[0])
                diffs.append((session_log, abs(this_mstime - mstime_start)))
                break

    best_diff = sorted(diffs, key=lambda d: d[1])[0] if diffs else []
    return best_diff

def find_matching_raw_dir(subject, experiment, mstime_start):

    raw_dir = os.path.join(DATA_ROOT, subject, 'raw', '*')
    filenames = glob.glob(os.path.join(raw_dir, '*.edf')) + glob.glob(os.path.join(raw_dir, '*.EEG')) \
                        + glob.glob(os.path.join(raw_dir, '*.EDF'))
    diffs = []
    for filename in filenames:
        reader = get_eeg_reader(filename)
        start_time = reader.get_start_time_ms()
        diffs.append((os.path.dirname(filename), start_time - mstime_start, reader.get_start_time_string()))
    sorted_diffs = sorted(diffs, key=lambda d:abs(d[1]))
    best_diff = sorted_diffs[0]
    return best_diff

def match_dirs(subject, experiment, session):
    if 'PS' in experiment:
        sessions = json.load(open('submission/ps_sessions.json'))
    else:
        sessions = json.load(open('submission/verbal_sessions.json'))
    code = subject
    try:
        sess_dict = sessions[subject][experiment][str(session)]
        code = sess_dict.get('code', subject)
        experiment = sess_dict.get('original_experiment', experiment)
        session = sess_dict.get('original_session', session)
        print 'orig session: {}'.format(session)
    except:
        pass

    reader = BaseEventReader(filename=os.path.join(DATA_ROOT, '..', 'events',
                                                   'RAM_{}'.format(experiment),
                                                   '{}_events.mat'.format(code)),
                             common_root=DATA_ROOT)
    mat_events = reader.read()
    session_events = mat_events[mat_events.session == session]
    mstime_start = session_events[0].mstime
    eeg_file = os.path.basename(session_events[10].eegfile)
    eeg_file_time = '_'.join(eeg_file.split('_')[-2:])


    #sample_rate = float(session_events[20].mstime - session_events[10].mstime) / float(session_events[20].eegoffset - session_events[10].eegoffset)
    #raw_time_start = session_events[20].mstime - (sample_rate * session_events[20].eegoffset)

    print find_matching_behavioral_dir(code, experiment, mstime_start)
    #print find_matching_raw_dir(code, experiment, raw_time_start)
    print get_matching_start_strings(code, eeg_file_time)


if __name__ == '__main__':
    #get_start_strings(raw_input('subject :'))
    subject = raw_input('Subject: ')
    exp = raw_input('experiment: ')
    session = int(raw_input('Session: '))
    match_dirs(subject, exp, session)