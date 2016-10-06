import re
import os
import json
import glob
import numpy as np
import sys
import argparse
import collections
import traceback


from pipelines import build_events_pipeline, build_split_pipeline,\
                      build_convert_eeg_pipeline, build_convert_events_pipeline,\
                      build_import_montage_pipeline, MATLAB_CONVERSION_TYPE, SOURCE_IMPORT_TYPE
from transferer import DB_ROOT, DATA_ROOT, UnTransferrableException
from loggers import log, logger

try:
    from ptsa.data.readers.BaseEventReader import BaseEventReader
    PTSA_LOADED=True
except:
    log('PTSA NOT LOADED')
    PTSA_LOADED=False



class UnknownMontageException(Exception):
    pass



def determine_montage_from_code(code, protocol='r1', allow_new=False, allow_skip=False):
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


def get_all_codes():
    subjects = []
    for experiment in EXPERIMENTS:
        if re.match(r'catFR[0-4]', experiment):
            ram_exp = 'RAM_{}'.format(experiment[0].capitalize() + experiment[1:])
        else:
            ram_exp = 'RAM_{}'.format(experiment)
        events_dir = os.path.join(DATA_ROOT, '..', 'events', ram_exp)
        events_files = sorted(glob.glob(os.path.join(events_dir, '*_events.mat')),
                              key=lambda f: f.split('_')[:-1])
        subjects.extend([os.path.basename(f).replace('_events.mat', '') for f in events_files])
    return sorted(list(set(subjects)))

def get_previous_subjects(subject):
    prev_subjects = []
    while '_' in subject:
        split_subj = subject.split('_')
        num = int(split_subj[1])
        if num == 1:
            subject = split_subj[0]
        else:
            subject = '{}_{}'.format(split_subj[0], num-1)
        prev_subjects.append(subject)
    return prev_subjects

def build_data_export_database():
    nested_dict = lambda: collections.defaultdict(nested_dict)
    subjects_for_export = [x.strip() for x in open('subjects_for_export.txt').readlines() if len(x.strip()) > 0 ]
    subjects = nested_dict()
    for experiment in ('FR1', 'FR2', 'YC1', 'YC2', 'PAL1', 'PAL2', 'catFR1', 'catFR2'):
        subject_sessions = get_subject_sessions_by_experiment(experiment, include_montage_changes=True)
        for subject, sessions in subject_sessions:
            if not subject[:6] in subjects_for_export:
                continue
            previous_subjects = get_previous_subjects(subject)
            max_previous_sessions = 0
            for previous_subject in previous_subjects:
                if previous_subject in subjects and experiment in subjects[previous_subject]:
                    previous_sessions = [int(s) for s in subjects[previous_subject][experiment].keys()]
                    max_session = max(previous_sessions)
                    max_previous_sessions = max(max_session, max_previous_sessions) + 1
            for i, session in enumerate(sessions):
                session_dict = {
                    'original_session': session,
                    'montage': determine_montage_from_code(subject, allow_new=True, allow_skip=True),
                    'code': subject
                }
                subject_without_montage = subject.split('_')[0]
                print 'adding', subject, experiment, i+max_previous_sessions
                subjects[subject_without_montage][experiment][i + max_previous_sessions] = session_dict
            #print(json.dumps(subjects, indent=2, sort_keys=True))
    json.dump(subjects, open('export_sessions.json', 'w'), indent=2, sort_keys=True)

def build_verbal_import_database():
    nested_dict = lambda: collections.defaultdict(nested_dict)
    subjects = nested_dict()
    for experiment in EXPERIMENTS:
        subject_sessions = get_subject_sessions_by_experiment(experiment, include_montage_changes=True)
        for subject, sessions in subject_sessions:
            previous_subjects = get_previous_subjects(subject)
            max_previous_sessions = 0
            for previous_subject in previous_subjects:
                if previous_subject in subjects and experiment in subjects[previous_subject]:
                    previous_sessions = [int(s) for s in subjects[previous_subject][experiment].keys()]
                    max_session = max(previous_sessions)
                    max_previous_sessions = max(max_session, max_previous_sessions) + 1
            for session in sessions:
                session_dict = {
                    'original_session': session,
                    'montage': determine_montage_from_code(subject, allow_new=True, allow_skip=True),
                    'code': subject
                }
                subject_without_montage = subject.split('_')[0]
                subjects[subject_without_montage][experiment][session + max_previous_sessions] = session_dict
            print(json.dumps(subjects, indent=2, sort_keys=True))
    json.dump(subjects, open('verbal_sessions.json', 'w'), indent=2, sort_keys=True)

def build_YC_import_database():
    nested_dict = lambda: collections.defaultdict(nested_dict)
    subjects = nested_dict()
    for experiment in ('YC1', 'YC2'):
        subject_sessions = get_subject_sessions_by_experiment(experiment, include_montage_changes=True)
        for subject, sessions in subject_sessions:
            previous_subjects = get_previous_subjects(subject)
            max_previous_sessions = 0
            for previous_subject in previous_subjects:
                if previous_subject in subjects and experiment in subjects[previous_subject]:
                    previous_sessions = [int(s) for s in subjects[previous_subject][experiment].keys()]
                    max_session = max(previous_sessions)
                    max_previous_sessions = max(max_session, max_previous_sessions) + 1
            for session in sessions:
                session_dict = {
                    'original_session': session,
                    'montage': determine_montage_from_code(subject, allow_new=True, allow_skip=True),
                    'code': subject
                }
                subject_without_montage = subject.split('_')[0]
                subjects[subject_without_montage][experiment][session + max_previous_sessions] = session_dict
            print(json.dumps(subjects, indent=2, sort_keys=True))
    json.dump(subjects, open('yc_sessions.json', 'w'), indent=2, sort_keys=True)

def run_montage_import_pipeline(kwargs, force_run=False):
    pipeline = build_import_montage_pipeline(**kwargs)
    pipeline.run(force_run)

def xtest_import_existing_montages():
    codes = get_all_codes()
    for code in codes:
        try:
            subject = code.split('_')[0]
            montage = determine_montage_from_code(code, 'r1', allow_new=True, allow_skip=True)
            kwargs = dict(
                subject=subject,
                montage=montage,
                code=code,
                protocol='r1'
            )
        except:
            continue
        yield run_montage_import_pipeline, kwargs, False



def get_subject_sessions_by_experiment(experiment, protocol='r1', include_montage_changes=False):
    if re.match('catFR[0-4]', experiment):
        ram_exp = 'RAM_{}'.format(experiment[0].capitalize() + experiment[1:])
    else:
        ram_exp = 'RAM_{}'.format(experiment)
    events_dir = os.path.join(DATA_ROOT, '..', 'events', ram_exp)
    events_files = sorted(glob.glob(os.path.join(events_dir, '{}*_events.mat'.format(protocol.upper()))),
                          key=lambda f: f.split('_')[:-1])
    for events_file in events_files:
        subject = '_'.join(os.path.basename(events_file).split('_')[:-1])
        if '_' in subject and not include_montage_changes:
            continue
        mat_events_reader = BaseEventReader(filename=events_file, common_root=DATA_ROOT)
        log('Loading matlab events {exp}: {subj}'.format(exp=experiment, subj=subject))
        try:
            mat_events = mat_events_reader.read()
            yield (subject, np.unique(mat_events['session']))
        except AttributeError:
            log('Failed.')


def get_first_unused_session(subject, experiment, protocol='r1'):
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


def run_individual_pipline(pipeline_fn, kwargs, force_run=False):
    pipeline = pipeline_fn(**kwargs)
    pipeline.run(force_run)

def run_convert_import_pipeline(kwargs, force_run_ephys, force_run_beh):
    convert_eeg_pipeline = build_convert_eeg_pipeline(**kwargs)
    convert_events_pipeline = build_convert_events_pipeline(**kwargs)

    convert_eeg_pipeline.run(force_run_ephys)
    convert_events_pipeline.run(force_run_beh)

def run_full_import_pipeline(kwargs, force_run=False):
    try:
        split_pipeline = build_split_pipeline(**kwargs)
        events_pipeline = build_events_pipeline(**kwargs)
        build = True
    except Exception as e:
        log('Could not make build pipeline: {}'.format(e))
        build = False

    try:
        convert_eeg_pipeline = build_convert_eeg_pipeline(**kwargs)
        convert_events_pipeline = build_convert_events_pipeline(**kwargs)
        convert = True
    except Exception as e:
        log('Could not make convert pipeline: {}'.format(e))
        if not build:
            raise
        convert = False

    if build and convert:
        if split_pipeline.previous_transfer_type() == MATLAB_CONVERSION_TYPE:
            pipeline_order = ((convert_eeg_pipeline, convert_events_pipeline),
                              (split_pipeline, events_pipeline))
        else:
            pipeline_order = ((split_pipeline, events_pipeline),
                              (convert_eeg_pipeline, convert_events_pipeline))
    elif build:
        pipeline_order = ((split_pipeline, events_pipeline),)
    else:
        pipeline_order = ((convert_eeg_pipeline, convert_events_pipeline),)

    for pipelines in pipeline_order:
        log("Attempting pipeline {}".format(pipelines[0].current_transfer_type()))
        try:
            for pipeline in pipelines:
                pipeline.run(force_run)
            return
        except Exception as e:
            log("Exception {} encountered while running pipeline {}".format(e, pipelines[0].current_transfer_type()))
            traceback.print_exc()

    raise UnTransferrableException("All pipelines failed!")

def check_subject_sessions(code, experiment, sessions, protocol='r1'):
    bad_sessions = json.load(open(BAD_EXPERIMENTS_FILE))
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
        if subject in bad_sessions and \
                        experiment in bad_sessions[subject] and \
                        str(session) in bad_sessions[subject][experiment]:
            log('SKIPPING {} {}_{}: {}'.format(subject, experiment, session,
                                               bad_sessions[subject][experiment][str(session)]))
            continue

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

        yield run_full_import_pipeline, test_inputs, test_inputs['force']


def check_experiment(experiment):
    experiment_sessions = get_subject_sessions_by_experiment(experiment)
    for subject, sessions in experiment_sessions:
        for test in check_subject_sessions(subject, experiment, sessions):
            yield test

BAD_EXPERIMENTS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unrecoverable_sessions.json')

EXPERIMENTS = ('FR1', 'FR2', 'FR3', 'PAL1', 'PAL2', 'PAL3', 'catFR1', 'catFR2', 'catFR3')


def clean_db_dir():
    for root, dirs, files in os.walk(DB_ROOT, False):
        if len(dirs) == 0 and len(files) == 1 and 'log.txt' in files:
            os.remove(os.path.join(root, 'log.txt'))
            log('Removing %s'%root)
            os.rmdir(root)
        elif len(os.listdir(root)) == 0:
            log('Removing %s'%root)
            os.rmdir(root)
this_dir = os.path.realpath(os.path.dirname(__file__))

def xtest_import_all_ps_sessions():
    for test in run_from_json_file(os.path.join(this_dir, 'ps_sessions.json')):
        yield test

def xtest_import_all_verbal_sessions():
    for test in run_from_json_file(os.path.join(this_dir, 'verbal_sessions.json')):
        yield test

def test_import_all_yc_sessions():
    for test in convert_from_json_file(os.path.join(this_dir, 'yc_sessions.json')):
        yield test


def xtest_import_sharing_database():
    for test in run_from_json_file(os.path.join(this_dir, 'export_sessions.json')):
        yield test

def xtest_all_experiments():
    logger.add_log_files(os.path.join(DB_ROOT, 'protocols', 'log.txt'))
    for experiment in EXPERIMENTS:
        for test in check_experiment(experiment):
            yield test


def convert_from_json_file(filename):
    subjects = json.load(open(filename))
    sorted_subjects = sorted(subjects.keys())
    for subject in sorted_subjects:
        experiments = subjects[subject]
        for new_experiment in experiments:
            sessions = experiments[new_experiment]
            for session in sessions:
                info = sessions[session]
                experiment = info.get('original_experiment', new_experiment)
                original_session = info.get('original_session', session)
                is_sys1 = info.get('system_1', False)
                is_sys2 = info.get('system_2', False)
                montage = info.get('montage', '0.0')
                force = info.get('force', False)
                code = info.get('code', subject)
                protocol = info.get('protocol', 'r1')

                montage_num = montage.split('.')[1]
                localization = montage.split('.')[0]
                inputs = dict(
                    protocol=protocol,
                    subject = subject,
                    montage = montage,
                    montage_num = montage_num,
                    localization=localization,
                    experiment = experiment,
                    new_experiment = new_experiment,
                    force = force,
                    do_compare = True,
                    code=code,
                    session=session,
                    original_session=original_session,
                    groups = (protocol,)
                )



                raw_substitute = info.get('raw_substitute', False)
                if raw_substitute:
                    inputs['substitute_raw_folder'] = raw_substitute
                if is_sys2 or experiment in ('FR3', 'PAL3', 'catFR3'):
                    inputs['groups'] += ('system_2',)
                else:
                    inputs['groups'] += ('system_1',)

                if 'PS' in experiment or 'TH' in experiment or 'YC' in experiment:
                    inputs['do_math'] = False
                else:
                    inputs['groups'] += ('verbal',)

                yield run_convert_import_pipeline, inputs, False, True



def run_from_json_file(filename):
    subjects = json.load(open(filename))
    sorted_subjects = sorted(subjects.keys())
    for subject in sorted_subjects:
        experiments = subjects[subject]
        for new_experiment in experiments:
            sessions = experiments[new_experiment]
            for session in sessions:
                info = sessions[session]
                experiment = info.get('original_experiment', new_experiment)
                original_session = info.get('original_session', session)
                is_sys1 = info.get('system_1', False)
                is_sys2 = info.get('system_2', False)
                montage = info.get('montage', '0.0')
                force = info.get('force', False)
                code = info.get('code', subject)
                protocol = info.get('protocol', 'r1')

                montage_num = montage.split('.')[1]
                localization = montage.split('.')[0]
                inputs = dict(
                    protocol=protocol,
                    subject = subject,
                    montage = montage,
                    montage_num = montage_num,
                    localization=localization,
                    experiment = experiment,
                    new_experiment = new_experiment,
                    force = force,
                    do_compare = True,
                    code=code,
                    session=session,
                    original_session=original_session,
                    groups = (protocol,)
                )



                raw_substitute = info.get('raw_substitute', False)
                if raw_substitute:
                    inputs['substitute_raw_folder'] = raw_substitute
                if is_sys2 or experiment in ('FR3', 'PAL3', 'catFR3'):
                    inputs['groups'] += ('system_2',)
                else:
                    inputs['groups'] += ('system_1',)

                if 'PS' in experiment or 'TH' in experiment or 'YC' in experiment:
                    inputs['do_math'] = False
                else:
                    inputs['groups'] += ('verbal',)

                yield run_full_import_pipeline, inputs, force




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-file', dest='from_file', default=False,
                        help='process from file')
    parser.add_argument('--substitute-raw-for-header', dest='raw_header', action='store_true', default=False,
                        help='Signals input for raw folder with substitute header')
    parser.add_argument('--change-experiment', dest='new_exp', action='store_true', default=False,
                        help='Singals new experiment differs from old')
    parser.add_argument('--change-session', dest='new_session', action='store_true', default=False,
                        help='Signals new session number different than old')
    parser.add_argument('--sys2', dest='sys2', action='store_true', default=False,
                        help='Signals system 2 session')
    parser.add_argument('--sys1', dest='sys1', action='store_true', default=False,
                        help='Signals system 1 session')
    parser.add_argument('--force', dest='force', action='store_true', default=False,
                        help='Force creation even if no changes')
    args = parser.parse_args()
    if args.from_file:
        for test in run_from_json_file(args.from_file):
            test[0](*test[1:])
        sys.exit(0)

    code = raw_input('Enter subject code: ')
    subject = re.sub('_.*', '', code)
    original_experiment = raw_input('Enter original experiment: ')
    original_session = int(raw_input('Enter original session number: '))

    if subject != code:
        montage = raw_input('Enter montage #: ')
    else:
        montage = '0.0'
        session = original_session

    if args.new_session:
        session = int(raw_input('Enter new session number: '))
    else:
        session = original_session

    if args.new_exp:
        experiment = raw_input('Enter new experiment name: ')
    else:
        experiment = original_experiment

    inputs = dict(
        protocol='ltp' if experiment.startswith('ltp') else 'r1',
        subject=subject,
        montage=montage,
        experiment=original_experiment,
        new_experiment=experiment,
        force=False,
        do_compare=True,
        code=code,
        session=session,
        original_session=original_session,
        groups=tuple()
    )

    inputs['groups'] += (inputs['protocol'],)

    if args.raw_header:
        header_substitute = raw_input('Enter raw folder containing substitute for header: ')
        inputs['substitute_raw_folder'] = header_substitute
    if args.sys2:
        inputs['groups'] += ('system_2',)
    else:
        inputs['groups'] += ('system_1',)
    if args.force:
        inputs['force'] = True

    if 'PS' in experiment or 'TH' in experiment:
        inputs['do_math'] = False
    else:
        inputs['groups'] += ('verbal',)

    if experiment[-1] == '3':
        inputs['match_field'] = 'eegoffset'

    #run_full_import_pipeline(inputs)
    run_individual_pipline(build_split_pipeline, inputs)
    #run_individual_pipline(build_events_pipeline, inputs)

