import re
import os
import json
import glob
import numpy as np

from transferer import generate_ephys_transferer, generate_session_transferer, TransferPipeline
from tasks import SplitEEGTask, EventCreationTask, AggregatorTask, CompareEventsTask
from transferer import DB_ROOT, DATA_ROOT
from loggers import log, logger

from parsers.math_parser import MathSessionLogParser

try:
    from ptsa.data.readers.BaseEventReader import BaseEventReader
    PTSA_LOADED=True
except:
    log('PTSA NOT LOADED')
    PTSA_LOADED=False

GROUPS = {
    'FR': ('verbal',),
    'PAL': ('verbal',),
    'catFR': ('verbal',)
}

class UnknownMontageException(Exception):
    pass

def build_split_pipeline(subject, montage, experiment, session, protocol='R1', groups=tuple(), code=None, original_session=None, **kwargs):
    transferer = generate_ephys_transferer(subject, experiment, session, protocol, groups,
                                           code=code, original_session=original_session,**kwargs)
    task = SplitEEGTask(subject, montage, experiment, session, **kwargs)
    return TransferPipeline(transferer, task)

def build_events_pipeline(subject, montage, experiment, session, do_math=True, protocol='R1', code=None,
                          groups=tuple(), do_compare=False, **kwargs):
    exp_type = re.sub('\d','', experiment)
    if exp_type in GROUPS:
        groups += GROUPS[exp_type]

    transferer, groups = generate_session_transferer(subject, experiment, session, protocol, groups, code=code, **kwargs)
    tasks = [EventCreationTask(subject, montage, experiment, session, 'system_2' in groups)]
    if do_math:
        tasks.append(EventCreationTask(subject, montage, experiment, session, 'system_2' in groups, 'math', MathSessionLogParser))
    if do_compare:
        tasks.append(CompareEventsTask(subject, montage, experiment, session, protocol, code,
                                       kwargs['original_session'] if 'original_session' in kwargs else None))

    tasks.append(AggregatorTask(subject, montage, experiment, session, protocol, code))
    return TransferPipeline(transferer, *tasks)

def determine_montage_from_code(code, protocol='R1', allow_new=False, allow_skip=False):
    montage_file = os.path.join(DB_ROOT, 'protocols', protocol, 'montages', code, 'info.json')
    if os.path.exists(montage_file):
        info = json.load(open(montage_file))
        return info['montage']
    elif '_' not in code:
        return '0.0'
    elif not allow_new:
        raise UnknownMontageException('Could not determine montage for {}'.format(code))
    else:
        montage_code = int(code.split('_')[1])
        ref_localized_file = os.path.join(DATA_ROOT, code, 'ref_localized.txt')
        if os.path.exists(ref_localized_file):
            ref_localized_subj = open(ref_localized_file).read()
            ref_montage = determine_montage_from_code(ref_localized_subj, protocol, allow_new, False)
            new_montage_num = float(ref_montage) + .1
        else:
            subj_code = code.split('_')[0]
            if montage_code == 0:
                return '0.0'
            ref_montage = determine_montage_from_code('{}_{}'.format(subj_code, montage_code-1), protocol, True, False)
            new_montage_num = (float(ref_montage) + 1.1)

        if new_montage_num % 1 > montage_code:
            raise Exception('Could not allocate new montage number with existing montage {} and code {}'.format(
                ref_montage, code))
        elif round((new_montage_num % 1) * 10) < montage_code:
            if allow_skip:
                log('Warning: Skipping montages from {} to {}'.format(new_montage_num, montage_code))
                return '%d.%d' % (int(new_montage_num), montage_code)
            else:
                raise Exception('Montage creation error: montages {} to {} do not exist'.format((new_montage_num%1)*10,
                                                                                                montage_code))

        return '%1.1f' % (new_montage_num)


def get_subject_sessions_by_experiment(experiment, protocol='R1'):
    events_dir = os.path.join(DATA_ROOT, '..', 'events', 'RAM_{}'.format(experiment))
    events_files = sorted(glob.glob(os.path.join(events_dir, '{}*_events.mat'.format(protocol))))
    for events_file in events_files:
        subject = '_'.join(os.path.basename(events_file).split('_')[:-1])
        mat_events_reader = BaseEventReader(filename=events_file, common_root=DATA_ROOT)
        log('Loading matlab events {exp}: {subj}'.format(exp=experiment, subj=subject))
        try:
            mat_events = mat_events_reader.read()
            yield (subject, np.unique(mat_events['session']))
        except AttributeError:
            log('Failed.')

def get_first_unused_session(subject, experiment, protocol='R1'):
    sessions_dir = os.path.join(DB_ROOT,
                               'protocols', protocol,
                               'subjects', subject,
                               'experiments', experiment,
                               'sessions')
    if not os.path.exists(sessions_dir):
        return 0
    session_folders = os.listdir(sessions_dir)
    session_folders_with_behavioral = [folder for folder in session_folders
                                       if os.path.isdir(os.path.join(sessions_dir, folder)) and
                                       'behavioral' in os.listdir(os.path.join(sessions_dir, folder))]
    session_folders_with_current = [folder for folder in session_folders_with_behavioral
                                    if 'current_processed' in os.listdir(os.path.join(sessions_dir, folder, 'behavioral'))]
    session_numbers = [int(sess) for sess in session_folders_with_current if sess.isdigit()]
    if session_numbers:
        return max(session_numbers) + 1
    else:
        return 0


TEST_SESSIONS = (
        dict(subject='R1002P', montage='0.0', experiment='FR1', session=0, force=True, do_compare=True),
        dict(subject='R1002P', montage='0.0', experiment='FR1', session=1, force=True, do_compare=True),
        dict(subject='R1001P', montage='0.0', experiment='FR1', session=1, force=True, do_compare=True),
        dict(subject='R1003P', montage='0.0', experiment='FR2', session=0, force=True, do_compare=True),
        dict(subject='R1006P', montage='0.1', experiment='FR2', session=1, code='R1006P_1', original_session=0, force=True, do_compare=True),
    )


def run_individual_pipline(pipeline_fn, kwargs, force_run=False):
    pipeline = pipeline_fn(**kwargs)
    pipeline.run(force_run)

def xtest_split_pipeline():
    for test_session in TEST_SESSIONS:
        yield run_individual_pipline, build_split_pipeline, test_session

def xtest_events_pipeline():
    for test_session in TEST_SESSIONS:
        yield run_individual_pipline, build_events_pipeline, test_session, 'force' in test_session and test_session['force']

def xtest_determine_montage():
    subjects = ('R1059J', 'R1059J_1', 'R1059J_2')
    for subject in subjects:
        print 'Montage number for {} is : '.format(subject), \
            determine_montage_from_code(subject, allow_new=True, allow_skip=True)

def xtest_get_subject_sessions():
    from pprint import pprint
    pprint(get_subject_sessions_by_experiment('FR1'))
    pprint(get_subject_sessions_by_experiment('FR2'))

def check_subject_sessions(code, experiment, sessions, protocol='R1'):
    subject = re.sub(r'_.*', '', code)
    montage = determine_montage_from_code(code, allow_new=True, allow_skip=True)
    test_inputs = dict(
        subject = subject,
        montage = montage,
        experiment = experiment,
        force = False,
        do_compare = True,
        code = code
    )
    sessions = sorted(sessions)
    for session in sessions:
        test_inputs['original_session'] = session
        if '_' in code:
            test_inputs['session'] = get_first_unused_session(subject, experiment, protocol)
        else:
            test_inputs['session'] = session

        if test_inputs['session'] != test_inputs['original_session']:
            log('Warning: {subj} {exp}_{orig_sess} -> {code} {exp}_{sess}'.format(subj=subject,
                                                                                  code=code,
                                                                                  sess=session,
                                                                                  exp=experiment,
                                                                                  orig_sess=test_inputs['session']))
        yield run_individual_pipline, build_split_pipeline, test_inputs
        yield run_individual_pipline, build_events_pipeline, test_inputs, False


def check_experiment(experiment):
    experiment_sessions = get_subject_sessions_by_experiment(experiment)
    for subject, sessions in experiment_sessions:
        for test in check_subject_sessions(subject, experiment, sessions):
            yield test

EXPERIMENTS = ('FR1', 'FR2', 'FR3', 'PAL1', 'PAL2', 'PAL3', 'catFR1', 'catFR2', 'catFR3')

def test_all_experiments():
    print 'STARTING'
    logger.add_log_files(os.path.join(DB_ROOT, 'protocols', 'log.txt'))
    for experiment in EXPERIMENTS:
        for test in check_experiment(experiment):
            yield test

if __name__ == '__main__':
    subject = raw_input('Enter subject code: ')
    experiment = raw_input('Enter experiment: ')
    session = int(raw_input('Enter new session number: '))
    code = raw_input('Enter code or blank: ')
    montage = raw_input('Enter montage or blank: ')
    original_session = raw_input('Enter old session number or blank: ')
    if not montage:
        montage = '0.0'

    if not code:
        code = subject

    if not original_session:
        original_session = session
    else:
        original_session = int(original_session)

    inputs = dict(
        subject=subject,
        montage=montage,
        experiment=experiment,
        force=False,
        do_compare=True,
        code=code,
        session=session,
        original_session=original_session
    )
    run_individual_pipline(build_split_pipeline, inputs)
    run_individual_pipline(build_events_pipeline, inputs)