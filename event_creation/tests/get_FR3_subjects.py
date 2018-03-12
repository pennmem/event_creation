import subprocess
import os
import glob

RHINO_DATA_LOC = '/Volumes/RHINO2/data/eeg'
RHINO_EVENTS_LOC = '/Volumes/RHINO2/data/events/'
TEST_DATA_LOC = '../tests/test_data'


def do_rsync(src, dst, exclude=tuple() , shell=False):
    exclude_cmd = []
    for exc in exclude:
        exclude_cmd += ['--exclude', exc]
    cmd = ['rsync', '-av'] + exclude_cmd + [src, dst]

    if shell:
        cmd = ' '.join(cmd)

    subprocess.call(cmd, shell=shell)

def copy_behavioral(subject, exp):
    exp_loc = os.path.join(RHINO_DATA_LOC, subject, 'behavioral', exp, '')
    dst_loc = os.path.join(TEST_DATA_LOC, subject, 'behavioral', exp)
    subprocess.call(['mkdir', '-p', dst_loc])
    do_rsync(exp_loc, dst_loc, ('*.wav',))

def copy_raw(subject, exp):
    raw_loc = os.path.join(RHINO_DATA_LOC, subject, 'raw', '%s_*' % exp)
    dst_loc = os.path.join(TEST_DATA_LOC, subject, 'raw', '')
    subprocess.call(['mkdir', '-p', dst_loc])
    do_rsync(raw_loc, dst_loc, ('*.ns2', '*.mat'), shell=True)


def get_subject_exp(subject, exp):
    copy_behavioral(subject, exp)
    copy_raw(subject, exp)

def get_exp_subjects(exp):
    ev_path = os.path.join(RHINO_EVENTS_LOC, exp, '*_events.mat')
    ev_files = glob.glob(ev_path)
    subs = [os.path.basename(ev_file)[:-11] for ev_file in ev_files]
    return subs

for sub in get_exp_subjects('RAM_FR3'):
    get_subject_exp(sub, 'FR3')

