import re
import os
import json
import glob
import numpy as np
import argparse
import collections
import traceback
import files
from collections import defaultdict

from pipelines import  MATLAB_CONVERSION_TYPE
from tasks import CleanDbTask, IndexAggregatorTask
from loggers import logger
from automation import Importer, ImporterCollection
from config import DATA_ROOT, RHINO_ROOT, DB_ROOT

try:
    from ptsa.data.readers.BaseEventReader import BaseEventReader
    PTSA_LOADED=True
except:
    logger.warn('PTSA NOT LOADED')
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
                logger.warn('Skipping montages for {} from {} to {}'.format(code, new_montage_num, montage_code))
                return '%d.%d' % (int(new_montage_num), montage_code)
            else:
                raise Exception('Montage creation error: montages {} to {} do not exist'.format((new_montage_num%1)*10,
                                                                                                montage_code))

        return '%1.1f' % (new_montage_num)

def get_subject_sessions_by_experiment(experiment, protocol='r1', include_montage_changes=False):
    if re.match('catFR[0-4]', experiment):
        ram_exp = 'RAM_{}'.format(experiment[0].capitalize() + experiment[1:])
    else:
        ram_exp = 'RAM_{}'.format(experiment)
    events_dir = os.path.join(DATA_ROOT, '..', 'events', ram_exp)
    events_files = sorted(glob.glob(os.path.join(events_dir, '{}*_events.mat'.format(protocol.upper()))),
                          key=lambda f: f.split('_')[:-1])
    seen_experiments = defaultdict(list)
    for events_file in events_files:
        subject = '_'.join(os.path.basename(events_file).split('_')[:-1])
        subject_no_montage = subject.split('_')[0]
        if '_' in subject and not include_montage_changes:
            continue
        mat_events_reader = BaseEventReader(filename=events_file, common_root=DATA_ROOT)
        logger.debug('Loading matlab events {exp}: {subj}'.format(exp=experiment, subj=subject))
        try:
            mat_events = mat_events_reader.read()
            sessions = np.unique(mat_events['session'])
            version_str = mat_events[-5]['expVersion'] if 'expVersion' in mat_events.dtype.names else '0'
            version = -1
            try:
                version = float(version_str.split('_')[-1])
            except:
                try:
                    version = float(version_str.split('v')[-1])
                except:
                    pass

            for i, session in enumerate(sessions):
                if 'experiment' in mat_events.dtype.names:
                    experiments = np.unique(mat_events[mat_events['session'] == session]['experiment'])
                else:
                    experiments = [experiment]
                for this_experiment in experiments:
                    n_sessions = seen_experiments[subject_no_montage].count(this_experiment)
                    yield subject_no_montage, subject, n_sessions, session, this_experiment, version
                    seen_experiments[subject_no_montage].append(this_experiment)
        except AttributeError:
            traceback.print_exc()
            logger.error('Could not get session from {}'.format(events_file))


def build_json_import_db(out_file, orig_experiments=None, excluded_experiments=None, included_subjects=None,
                         protocol='r1', include_montage_changes=True, **extra_items):
    nested_dict = lambda: collections.defaultdict(nested_dict)
    subjects = nested_dict()
    excluded_experiments = excluded_experiments or []
    for experiment in orig_experiments:
        logger.info("Building json DB for experiment {}".format(experiment))
        subject_sessions = get_subject_sessions_by_experiment(experiment, protocol, include_montage_changes)
        for subject, code, session, orig_session, new_experiment, version in subject_sessions:
            if (not included_subjects is None) and (not code in included_subjects):
                continue
            if excluded_experiments and new_experiment in excluded_experiments:
                continue
            session_dict = {
                'original_session': orig_session,
                'montage': determine_montage_from_code(code, allow_new=True, allow_skip=True),
                'code': code
            }
            if (new_experiment != experiment):
                session_dict['original_experiment'] = experiment
            if (version >= 2):
                session_dict['system_2'] = True
            elif version > 0:
                session_dict['system_1'] = True
            session_dict.update(extra_items)
            subjects[subject][new_experiment][session] = session_dict
    with files.open_with_perms(out_file, 'w') as f:
        json.dump(subjects, f, indent=2, sort_keys=True)


def build_sharing_import_database():
    subjects_for_export = [x.strip() for x in open('subjects_for_export.txt').readlines() if len(x.strip()) > 0 ]
    experiments = ('FR1', 'FR2', 'YC1', 'YC2', 'PAL1', 'PAL2', 'catFR1', 'catFR2')
    build_json_import_db('export_sessions.json', experiments, [], subjects_for_export, 'r1', True)

def build_verbal_import_database():
    experiments = ('FR1', 'FR2', 'FR3', 'PAL1', 'PAL2', 'PAL3', 'catFR1', 'catFR2', 'catFR3')
    build_json_import_db('verbal_sessions.json', experiments)

def build_YC_import_database():
    experiments = ('YC1', 'YC2')
    build_json_import_db('yc_sessions.json', experiments)

def build_TH_import_database():
    experiments = ('TH1', 'TH3')
    build_json_import_db('th_sessions.json', experiments)

RAM_EXPERIMENTS = ('FR1', 'FR2', 'FR3',
                   'YC1', 'YC2', 'YC3',
                   'PAL1', 'PAL2', 'PAL3',
                   'catFR1', 'catFR2', 'catFR3',
                   'TH1', 'TH3',
                   'PS')

def build_ram_import_database():
    build_json_import_db('r1_sessions.json', RAM_EXPERIMENTS, ('PS0',))
    pass


IMPORT_DB_BUILDERS = {
    'sharing': build_sharing_import_database,
    'ram': build_ram_import_database,
}


def run_individual_pipline(pipeline_fn, kwargs, force_run=False):
    pipeline = pipeline_fn(**kwargs)
    pipeline.run(force_run)


def attempt_importers(importers, force):
    success = False
    i=0
    for i, importer in enumerate(importers):
        logger.info("Attempting {}".format(importer.label))
        if importer.should_transfer or force:
            importer.run(force)

        if not importer.errored:
            logger.info("{} succeded".format(importer.label))
            success = True
            break
        logger.warn("{} failed".format(importer.label))
    if not success:
        descriptions = [importer.describe_errors() for importer in importers]
        logger.error("All importers failed. Errors: \n{}".format(', '.join(descriptions)))
    return success, importers[:i+1]


def run_session_import(kwargs, do_import=True, do_convert=False, force_events=False, force_eeg=False):
    """
    :param kwargs:
    :param do_import:
    :param do_convert:
    :param force_events:
    :param force_eeg:
    :return: (success (t/f), attempted pipelines)
    """
    logger.set_subject(kwargs['subject'], kwargs['protocol'])
    logger.set_label('Session Importer')

    ephys_importers = []
    events_importers = []

    if do_import:
        ephys_builder = Importer(Importer.BUILD_EPHYS, **kwargs)
        events_builder = Importer(Importer.BUILD_EVENTS, **kwargs)
        if ephys_builder.initialized:
            ephys_importers.append(ephys_builder)
        if events_builder.initialized:
            events_importers.append(events_builder)

    if do_convert:
        ephys_converter = Importer(Importer.CONVERT_EPHYS, **kwargs)
        events_converter = Importer(Importer.CONVERT_EVENTS, **kwargs)
        if ephys_converter.initialized:
            if ephys_converter.previous_transfer_type() == MATLAB_CONVERSION_TYPE:
                ephys_importers = [ephys_converter] + ephys_importers
            else:
                ephys_importers.append(ephys_converter)

        if events_converter.initialized:
            if events_converter.previous_transfer_type() == MATLAB_CONVERSION_TYPE:
                events_importers = [events_converter] + events_importers
            else:
                events_importers.append(events_converter)

    ephys_success, attempted_ephys = attempt_importers(ephys_importers, force_eeg)

    if ephys_success:
        events_success, attempted_events = attempt_importers(events_importers, force_events)
        logger.unset_subject()
        return events_success and ephys_success, ImporterCollection(attempted_ephys + attempted_events)
    else:
        logger.unset_subject()
        return ephys_success, ImporterCollection(attempted_ephys)


def run_montage_import(kwargs, force=False):
    logger.set_subject(kwargs['subject'], kwargs['protocol'])
    logger.set_label('Montage Importer')

    importer = Importer(Importer.MONTAGE, **kwargs)
    success, importers = attempt_importers([importer], force)
    return success, ImporterCollection(importers)


this_dir = os.path.realpath(os.path.dirname(__file__))



def build_session_inputs(subject, new_experiment, session, info):
    experiment = info.get('original_experiment', new_experiment)
    original_session = info.get('original_session', session)
    is_sys1 = info.get('system_1', False)
    is_sys2 = info.get('system_2', False)
    montage = info.get('montage', '0.0')
    code = info.get('code', subject)
    protocol = info.get('protocol', 'r1')
    attempt_import = info.get('attempt_import', True)
    attempt_conversion = info.get('attempt_conversion', True)

    montage_num = montage.split('.')[1]
    localization = montage.split('.')[0]

    inputs = dict(
        protocol=protocol,
        subject=subject,
        montage=montage,
        montage_num=montage_num,
        localization=localization,
        experiment=experiment,
        new_experiment=new_experiment,
        code=code,
        session=session,
        original_session=original_session,
        groups=(protocol,),
        attempt_conversion=attempt_conversion,
        attempt_import=attempt_import
    )

    if is_sys2 or experiment in ('FR3', 'PAL3', 'catFR3', 'TH3', 'PS2.1'):
        inputs['groups'] += ('system_2',)
    elif is_sys1:
        inputs['groups'] += ('system_1',)

    if experiment.startswith('PS') or experiment.startswith('TH') or experiment.startswith('YC'):
        inputs['do_math'] = False

    if experiment.startswith('FR') or experiment.startswith('catFR') or experiment.startswith('PAL'):
        inputs['groups'] += ('verbal', )

    return inputs


def montage_inputs_from_json(filename):
    subjects = json.load(open(filename))
    sorted_subjects = sorted(subjects.keys())
    completed_codes = set()
    for subject in sorted_subjects:
        experiments = subjects[subject]
        for new_experiment in experiments:
            sessions = experiments[new_experiment]
            for session in sessions:
                info = sessions[session]
                code = info.get('code', subject)
                if code in completed_codes:
                    continue
                inputs = dict(
                    subject=subject,
                    code=code,
                    montage=info.get('montage', '0.0'),
                    protocol='r1'
                )
                completed_codes.add(code)
                yield inputs


def session_inputs_from_json(filename):
    subjects = json.load(open(filename))
    sorted_subjects = sorted(subjects.keys())
    for subject in sorted_subjects:
        experiments = subjects[subject]
        for new_experiment in experiments:
            sessions = experiments[new_experiment]
            for session in sessions:
                info = sessions[session]
                inputs = build_session_inputs(subject, new_experiment, session, info)
                yield inputs

def importer_sort_key(importer):
    return (importer.kwargs['subject'] if 'subject' in importer.kwargs else '',
            importer.kwargs['experiment'] if 'experiment' in importer.kwargs else '',
            importer.kwargs['session'] if 'session' in importer.kwargs else -1,
            importer.label)


def import_sessions_from_json(filename, do_import, do_convert, force_events=False, force_eeg=False):
    successes = []
    failures = []
    interrupted = False
    try:
        for inputs in session_inputs_from_json(filename):
            success, importers = run_session_import(inputs, do_import, do_convert, force_events, force_eeg)
            if success:
                successes.append(importers)
            else:
                failures.append(importers)
    except Exception as e:
        logger.error("Catastrophic failure importing montages: message {}".format(e))
        traceback.print_exc()
    except KeyboardInterrupt:
        logger.error("Keyboard interrupt. Exiting")
        traceback.print_exc()
        interrupted = True
    return successes, failures, interrupted


def import_montages_from_json(filename, force=False):
    successes = []
    failures = []
    interrupted = False
    try:
        for inputs in montage_inputs_from_json(filename):
            success, importers = run_montage_import(inputs, force)
            if success:
                successes.append(importers)
            else:
                failures.append(importers)
    except Exception as e:
        logger.error("Catastrophic failure importing montages: message {}".format(e))
        traceback.print_exc()
    except KeyboardInterrupt:
        logger.error("Keyboard interrupt. Exiting")
        traceback.print_exc()
        interrupted = True
    return successes, failures, interrupted

def run_json_import(filename, do_import, do_convert, force_events=False, force_eeg=False, force_montage=False,
                    log_file='json_import.log'):
    montage_successes, montage_failures, interrupted = import_montages_from_json(filename, force_montage)
    if not interrupted:
        successes, failures, _ = import_sessions_from_json(filename, do_import, do_convert, force_events, force_eeg)
        sorted_failures = sorted(failures + montage_failures, key=importer_sort_key)
        sorted_successes = sorted(successes + montage_successes, key=importer_sort_key)
    else:
        sorted_failures = sorted(montage_failures, key=importer_sort_key)
        sorted_successes = sorted(montage_successes, key=importer_sort_key)

    with files.open_with_perms(log_file, 'w') as output:
        output.write("Successful imports: {}\n".format(len(sorted_successes)))
        output.write("Failed imports: {}\n\n".format(len(sorted_failures)))
        output.write("###### FAILURES #####\n")
        for failure in sorted_failures:
            output.write('{}\n\n'.format(failure.describe()))
        output.write("\n###### SUCCESSES #####\n")
        for success in sorted_successes:
            output.write('{}\n\n'.format(success.describe()))

    return sorted_failures

def prompt_for_session_inputs(**opts):
    code = raw_input('Enter subject code: ')
    subject = re.sub(r'_.*', '', code)
    original_experiment = raw_input('Enter original experiment: ')
    original_session = raw_input('Enter original session number: ')

    if subject != code:
        montage = raw_input('Enter montage as #.#: ')
        montage_num = montage.split('.')[1]
        localization = montage.split('.')[0]
    else:
        montage = '0.0'
        montage_num = 0
        localization = 0

    if opts.get('change_session', False) or subject != code or original_experiment.startswith('PS'):
        session = int(raw_input('Enter new session number: '))
    else:
        session = original_session

    if opts.get('change_experiment', False) or original_experiment.startswith('PS'):
        experiment = raw_input('Enter new experiment: ')
    else:
        experiment = original_experiment

    protocol = 'ltp' if experiment.startswith('ltp') else \
               'r1' if subject.startswith('R') else None

    attempt_conversion = opts.get('allow_convert', False)
    attempt_import = not opts.get('force_convert', False)

    inputs = dict(
        protocol=protocol,
        subject=subject,
        montage=montage,
        montage_num=montage_num,
        localization=localization,
        experiment=original_experiment,
        new_experiment=experiment,
        force=False,
        do_compare=True,
        code=code,
        session=session,
        original_session=original_session,
        groups=(protocol,),
        attempt_import=attempt_import,
        attempt_conversion=attempt_conversion
    )

    if opts.get('sys2', False):
        inputs['groups'] += ('system_2',)
    elif opts.get('sys1', False):
        inputs['groups'] += ('system_1',)

    if experiment.startswith('PS') or experiment.startswith('TH'):
        inputs['do_math'] = False

    if experiment.startswith('FR') or experiment.startswith('catFR') or experiment.startswith('PAL'):
        inputs['groups'] += ('verbal',)

    return inputs


def prompt_for_montage_inputs():
    code = raw_input('Enter original subject code (including _#): ')
    subject = re.sub(r'_.*', '', code)

    montage = raw_input('Enter montage as #.#: ')

    montage_num = montage.split(".")[1]
    subject_split = code.split("_")
    subject_montage = subject_split[1] if len(subject_split)>1 else "0"

    if ( subject_montage ) != montage_num:
        print "WARNING: subject code {} does not match montage number {} !".format(subject_montage, montage_num)
        confirmed = confirm("Are you sure you want to continue? ")
        if not confirmed:
            return False


    inputs = dict(
        subject=subject,
        montage=montage,
        code=code,
        protocol='r1'
    )

    return inputs

def session_exists(protocol, subject, experiment, session):
    session_dir = os.path.join(DB_ROOT, 'protocols', protocol,
                                 'subjects', subject,
                                 'experiments', experiment,
                                 'sessions', str(session))
    behavioral_current = os.path.join(session_dir, 'behavioral', 'current_processed')
    eeg_current = os.path.join(session_dir, 'ephys', 'current_processed')

    return os.path.exists(behavioral_current) and os.path.exists(eeg_current)

def montage_exists(protocol, subject, montage):
    montage_num = montage.split('.')[1]
    localization = montage.split('.')[0]
    montage_dir = os.path.join(DB_ROOT, 'protocols', protocol,
                               'subjects', subject,
                               'localizations', localization,
                               'montages', montage_num)
    neurorad_current = os.path.join(montage_dir, 'neuroradiology', 'current_processed')

    return os.path.exists(neurorad_current)

def confirm(prompt):
    while True:
        resp = raw_input(prompt)
        if resp.lower() in ('y','n', 'yes', 'no'):
            return resp.lower() == 'y' or resp.lower() == 'yes'
        print('Please enter y or n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--montage-only', dest='montage_only', action='store_true', default=False,
                        help='Imports a localization or montage instead of importing events')
    parser.add_argument('--json', dest='json_file', default=False,
                        help='Imports all sessions from the specified JSON file')
    parser.add_argument('--change-experiment', dest='change_experiment', action='store_true', default=False,
                        help='Signals that the name of the experiment changes on import. '
                             'Defaults to true for PS* experiments')
    parser.add_argument('--change-session', dest='change_session', action='store_true', default=False,
                        help='Signals that the session number changes on import. Defaults to true '
                             'for PS* experiments or subjects with montage changes')
    parser.add_argument('--sys2', dest='sys2', action='store_true', default=False,
                        help='Forces system 2 processing if it cannot be determined automatically')
    parser.add_argument('--sys1', dest='sys1', action='store_true', default=False,
                        help='Signals system 1 processing if it cannot be determined automatically')
    parser.add_argument('--allow-convert', dest='allow_convert', action='store_true', default=False,
                        help='Attempt to convert events from matlab if standard events creation fails')
    parser.add_argument('--force-convert', dest='force_convert', action='store_true', default=False,
                        help='ONLY attempt conversion of events from matlab, skipping standard import')
    parser.add_argument('--force-events', dest='force_events', action='store_true', default=False,
                        help='Force events creation even if no changes have occurred')
    parser.add_argument('--force-eeg', dest='force_eeg', action='store_true', default=False,
                        help='Force eeg splitting even if no changes have occurred')
    parser.add_argument('--force-montage', dest='force_montage', action='store_true', default=False,
                        help='Force montage creation even if no changes have occurred')
    parser.add_argument('--clean-only', dest='clean_db', action='store_true', default=False,
                        help='ONLY clean the database, removing empty folders and folders without processed equivalent')
    parser.add_argument('--aggregate-only', dest='aggregate', action='store_true', default=False,
                        help='ONLY aggregate index files')
    parser.add_argument('--build-db', dest='db_name', default=False,
                        help='ONLY build a database for import. \rOptions are [ram, sharing]')

    args = parser.parse_args()

    if args.clean_db:
        print('Cleaning database and ignoring other arguments')
        clean_task = CleanDbTask()
        clean_task.run()
        print('Database cleaned. Exiting.')
        exit(0)

    if args.aggregate:
        print('Aggregating index files and ignoring other arguments')
        aggregate_task = IndexAggregatorTask()
        aggregate_task.run()
        print('Indexes aggregated. Exiting.')
        exit(0)

    if args.db_name:
        db_builder = IMPORT_DB_BUILDERS[args.db_name]
        print('Building import database: {}'.format(args.db_name))
        db_builder()
        print('DB built. Exiting.')
        exit(0)

    attempt_convert = args.allow_convert or args.force_convert
    attempt_import = not args.force_convert

    if args.json_file:
        print('Running from JSON file: {}'.format(args.json_file))
        import_log = 'json_import'
        i=1
        while os.path.exists(import_log + '.log'):
            import_log = 'json_import' + str(i)
            i+=1
        import_log = import_log + '.log'
        failures = run_json_import(args.json_file, attempt_import, attempt_convert,
                                   args.force_events, args.force_eeg, args.force_montage, import_log)
        if failures:
            print('\n******************\nSummary of failures\n******************\n')
            print('\n\n'.join([failure.describe() for failure in failures]))
        else:
            print('No failures.')
        IndexAggregatorTask().run()
        print('Log created: {}. Exiting'.format(import_log))
        exit(0)

    if args.montage_only:
        inputs = prompt_for_montage_inputs()
        if inputs == False:
            exit(0)
        if montage_exists(inputs['protocol'], inputs['subject'], inputs['montage']):
            if not confirm('{subject}, montage {montage} already exists. Continue and overwrite? '.format(**inputs)):
                print('Import aborted! Exiting.')
                exit(0)
        print('Importing montage')
        success, importer = run_montage_import(inputs, args.force_montage)
        print('Success:' if success else 'Failed:')
        print(importer.describe())
        exit(0)



    inputs = prompt_for_session_inputs(**vars(args))

    if session_exists(inputs['protocol'], inputs['subject'], inputs['new_experiment'], inputs['session']):
        if not confirm('{subject} {new_experiment} session {session} already exists. '
                       'Continue and overwrite? '.format(**inputs)):
            print('Import aborted! Exiting.')
            exit(0)
    print('Importing session')
    success, importers = run_session_import(inputs, attempt_import, attempt_convert, args.force_events, args.force_eeg)
    if success:
        IndexAggregatorTask().run()
    print('Success:' if success else "Failed:")
    print(importers.describe())

    exit(0)

