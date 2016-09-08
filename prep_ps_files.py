from submission.transferer import DATA_ROOT
import os
import glob
from ptsa.data.readers import BaseEventReader
import numpy as np
import json
from readers.eeg_reader import get_eeg_reader
from readers.nsx_utility.brpylib import NsxFile

def get_suffixes(subject):
    events_file= os.path.join(DATA_ROOT, '..', 'events', 'RAM_PS', '{}_events.mat'.format(subject))
    mat_events_reader = BaseEventReader(filename=events_file, common_root=DATA_ROOT)
    mat_events = mat_events_reader.read()
    sessions = np.unique(mat_events.session)
    suffixes = {}
    for session in sessions:
        sess_events = mat_events[mat_events.session == session]
        eegfile = sess_events[10].eegfile
        suffixes[session] =  '_'.join(eegfile.split('_')[-2:])
    return suffixes

extensions = '.edf', '.EDF', '.eeg', '.EEG', '/*.ns2'

def get_raw_start_strings(subject):
    filenames = []
    for extension in extensions:
        filenames += glob.glob(os.path.join(DATA_ROOT, subject, 'raw', '*', '*{}'.format(extension)))

    raw_dirnames = {}
    for filename in filenames:
        try:
            if os.path.splitext(filename)[-1] == '.ns2':
                start_string = NsxFile(filename).basic_header['TimeOrigin'].strftime('%d%b%y_%H%M')
            else:
                reader = get_eeg_reader(filename)
                start_string = reader.get_start_time_string()
            if os.path.splitext(filename)[-1] == '.ns2':
                raw_dirnames[start_string] = os.path.dirname(os.path.dirname(filename))
            else:
                raw_dirnames[start_string] = os.path.dirname(filename)
        except:
            print 'Couldn\'t read {}'.format(filename)
    return raw_dirnames

def get_sync_file(subject, suffix):
    filenames = glob.glob(os.path.join(DATA_ROOT, subject, 'eeg.noreref', '*{}*.sync.txt'.format(suffix)))

    real_filenames = []
    for filename in filenames:
        real_filename = os.path.realpath(filename)
        if real_filename not in real_filenames:
            real_filenames.append(real_filename)

    if len(real_filenames) != 1:
        ps_filenames = []
        for filename in real_filenames:
            if 'PS' in filename:
                ps_filenames.append(filename)
        real_filenames = ps_filenames
    if len(real_filenames) != 1:
        raise Exception('too many or not enough filenames with suffix {}\n{}'.format(suffix, filenames))
    return real_filenames[0]


def make_raw_symlink(subject, orig_raw_dir, session):
    new_raw_dir = os.path.join(DATA_ROOT, subject, 'raw', 'PS_{}'.format(session))
    if os.path.realpath(orig_raw_dir) == os.path.realpath(new_raw_dir):
        print 'Raw symlink already in place'
        return
    if os.path.islink(new_raw_dir):
        os.unlink(new_raw_dir)
    if os.path.exists(new_raw_dir):
        raise Exception('{} already exists'.format(new_raw_dir))
    try:
        os.symlink(os.path.basename(orig_raw_dir), new_raw_dir)
    except OSError:
        print 'Weird OS error...'
        pass
    except:
        print '{} -> {} unsuccessful!'.format(new_raw_dir, os.path.basename(orig_raw_dir))
        raise
    print('{} {} RAW LINK MADE SUCCESSFULLY'.format(subject, session))


def make_sync_symlink(subject, orig_sync, session):
    sync_basename = '.'.join(orig_sync.split('.')[:-3])
    new_sync_name = '{}.PS_{}.sync.txt'.format(sync_basename, session)
    if os.path.realpath(orig_sync) == os.path.realpath(new_sync_name):
        print 'Sync symlink already in place'
        return
    if os.path.exists(new_sync_name):
        raise Exception('{} already exists'.format(new_sync_name))
    os.symlink(os.path.basename(orig_sync), new_sync_name)
    print('{} {} SYNC LINK MADE SUCCESSFULLY'.format(subject, session))

def test_prep_ps():
    ps_subjects = json.load(open('submission/ps_sessions.json'))
    subjects = sorted(ps_subjects.keys())

    for subject in subjects:
        raw_dirnames = False
        experiments = ps_subjects[subject]
        for experiment in experiments:
            sessions = experiments[experiment]
            for new_session in sessions:
                code = sessions[new_session].get('code', subject)
                #if not sessions[new_session].get('system_2', True):
                #    print 'skipping {} {} {}'.format(subject, experiment, new_session)
                #    continue
                if not raw_dirnames:
                    raw_dirnames = get_raw_start_strings(code)
                suffixes = get_suffixes(code)

                original_session = sessions[new_session]['original_session']
                suffix = suffixes[original_session]
                try:
                    raw_dir = raw_dirnames.get(suffix, None)
                    if not raw_dir:
                        continue
                    yield make_raw_symlink, code, raw_dir, original_session
                except Exception as e:
                    print 'subject {} raw skipped: {}'.format(code, e)
                if sessions[new_session].get('system_1', False):
                    try:
                        yield get_sync_file, code, suffix
                        sync_file = get_sync_file(code, suffix)
                        yield make_sync_symlink, code, sync_file, original_session
                    except Exception as e:
                        print 'subject {} sync skipped: {}'.format(subject, e)
