from __future__ import print_function

import sys
from event_creation import confirm
if sys.version_info[0] < 3:
    input = raw_input
from .configuration import config

if __name__ == '__main__':
    config.parse_args()
    import matplotlib
    if not config.show_plots:
        matplotlib.use('agg')
    else:
        matplotlib.use('Qt4Agg')

import re
import os
import json
import glob
import numpy as np
import collections
import traceback
from collections import defaultdict

from . import fileutil
from .configuration import  paths
from .exc import MontageError
from .tasks import CleanDbTask, IndexAggregatorTask
from .events_tasks import ReportLaunchTask
from .log import logger
from .automation import Importer, ImporterCollection

from ptsa.data.readers import JsonIndexReader

try:
    from ptsa.data.readers import BaseEventReader
except:
    logger.warn('PTSA NOT LOADED')


def determine_montage_from_code(code, protocol='r1', allow_new=False, allow_skip=False):
    montage_file = os.path.join(paths.db_root, 'protocols', protocol, 'montages', code, 'info.json')
    if os.path.exists(montage_file):
        info = json.load(open(montage_file))
        return info['montage']
    elif '_' not in code:
        return '0.0'
    elif not allow_new:
        raise MontageError('Could not determine montage for {}'.format(code))
    else:
        montage_code = int(code.split('_')[1])
        ref_localized_file = os.path.join(paths.data_root, code, 'ref_localized.txt')
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


def get_ltp_subject_sessions_by_experiment(experiment):
    events_dir = os.path.join(paths.data_root, 'scalp', 'ltp', experiment, 'behavioral', 'events')
    events_files = sorted(glob.glob(os.path.join(events_dir, 'events_all_LTP*.mat')),
                          key=lambda f: f.split('_')[:-1])
    seen_experiments = defaultdict(list)
    for events_file in events_files:
        print(events_file)
        subject = os.path.basename(events_file)[11:-4]  # Subject number is the basename with events_all_, .mat removed
        subject_no_year = subject.split('_')[0]
        if '_' in subject:
            continue
        mat_events_reader = BaseEventReader(filename=events_file, common_root=paths.data_root)
        logger.debug('Loading matlab events {exp}: {subj}'.format(exp=experiment, subj=subject))
        try:
            mat_events = mat_events_reader.read()
            sessions = np.unique(mat_events['session']) - 1  # MATLAB events start counting sessions at 1 instead of 0
            version = 0.
            for i, session in enumerate(sessions):
                if 'experiment' in mat_events.dtype.names:
                    experiments = np.unique(mat_events[mat_events['session'] == session]['experiment'])
                else:
                    experiments = [experiment]
                for this_experiment in experiments:
                    n_sessions = seen_experiments[subject_no_year].count(this_experiment)
                    yield subject_no_year, subject, n_sessions, session, this_experiment, version
                    seen_experiments[subject_no_year].append(this_experiment)
        except IndexError or AttributeError:
            traceback.print_exc()
            logger.error('Could not get session from {}'.format(events_file))


def get_subject_sessions_by_experiment(experiment, protocol='r1', include_montage_changes=False):
    """

    :param experiment:
    :param protocol:
    :param include_montage_changes:
    :return: subject, subject_code,  session, original_session, experiment, version
    """
    json_reader = JsonIndexReader(os.path.join(paths.rhino_root,'protocols','%s.json'%protocol))
    if experiment in json_reader.experiments():
        subjects = json_reader.subjects(experiment=experiment)
        for subject_no_montage in subjects:
            for montage in json_reader.montages(subject=subject_no_montage, experiment=experiment):
                subject = subject_no_montage if montage == '0' else '%s_%s' % (subject_no_montage, montage)
                sessions = json_reader.sessions(subject=subject_no_montage, montage=montage, experiment=experiment)
                for session in sessions:
                    try:
                        original_session =  json_reader.get_value('original_session',
                                                                  subject=subject_no_montage,experiment=experiment,
                                                                  session=session)
                    except ValueError:
                        original_session = session # not necessarily robust
                    yield subject_no_montage, subject,session, original_session,  experiment, '0'
    else:
        if re.match('catFR[0-4]', experiment):
            ram_exp = 'RAM_{}'.format(experiment[0].capitalize() + experiment[1:])
        else:
            ram_exp = 'RAM_{}'.format(experiment)
        events_dir = os.path.join(paths.data_root,'events',ram_exp)
        events_files = sorted(glob.glob(os.path.join(events_dir, '{}*_events.mat'.format(protocol.upper()))),
                              key=lambda f: f.split('_')[:-1])
        seen_experiments = defaultdict(list)
        for events_file in events_files:
            subject = '_'.join(os.path.basename(events_file).split('_')[:-1])
            subject_no_montage = subject.split('_')[0]
            if '_' in subject:
                if not include_montage_changes:
                    continue
            mat_events_reader = BaseEventReader(filename=events_file, common_root=paths.data_root)
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
        if protocol == 'ltp':
            subject_sessions = get_ltp_subject_sessions_by_experiment(experiment)
        else:
            subject_sessions = get_subject_sessions_by_experiment(experiment, protocol, include_montage_changes)
        for subject, code, session, orig_session, new_experiment, version in subject_sessions:
            if (not included_subjects is None) and (not code in included_subjects):
                continue
            if excluded_experiments and new_experiment in excluded_experiments:
                continue
            if protocol == 'ltp':
                session_dict = {
                    'original_session': orig_session,
                    'montage': '0.0',
                    'code': code
                }
            else:
                session_dict = {
                    'original_session': orig_session,
                    'montage': determine_montage_from_code(code, allow_new=True, allow_skip=True),
                    'code': code
                }
            if (new_experiment != experiment):
                if new_experiment == 'PS2.1':
                    session_dict['original_experiment'] = 'PS21'
                else:
                    session_dict['original_experiment'] = experiment
            # if (version >= 2):
            #     session_dict['system_2'] = True
            # elif version > 0:
            #     session_dict['system_1'] = True
            session_dict.update(extra_items)
            subjects[subject][new_experiment][session] = session_dict

    with fileutil.open_with_perms(out_file, 'w') as f:
        json.dump(subjects, f, indent=2, sort_keys=True)


def build_sharing_import_database():
    subjects_for_export = [x.strip() for x in open(os.path.join(os.path.dirname(__file__),'subjects_for_export.txt')).readlines() if len(x.strip()) > 0 ]
    experiments = ('FR1', 'FR2', 'YC1', 'YC2', 'PAL1', 'PAL2', 'catFR1', 'catFR2','TH1')
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
    """
    Runs each importer in importers until one of them succeeds
    :param importers: A list of importers to attempt
    :param force: Whether to force an import when no change is found
    :return: The importers that were attempted
    """
    success = False
    i=0
    for i, importer in enumerate(importers):
        logger.set_label(importer.label)
        logger.info("Attempting {}".format(importer.label))
        if importer.should_transfer() or (force and importer.initialized):
            importer.run(force)

        if not importer.errored:
            logger.info("{} succeded".format(importer.label))
            success = True
            break
        logger.warn("{} failed".format(importer.label))
    if not success:
        descriptions = [importer.describe_errors() for importer in importers]
        logger.critical("All importers failed. Errors: \n{}".format(', '.join(descriptions)))

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

    attempted_importers = []
    successes = [True]

    if do_import:
        ephys_builder = Importer(Importer.BUILD_EPHYS,**kwargs)
        success,attempts = attempt_importers([ephys_builder],force_eeg)
        attempted_importers.extend(attempts)
        successes.append(success)
        if success:
            events_builder = Importer(Importer.BUILD_EVENTS,**kwargs)
            success,attempts = attempt_importers([events_builder],force_events)
            attempted_importers.extend(attempts)
            if success:
                return all(successes), ImporterCollection(attempted_importers)
            else:
                logger.info('Events builder failed')
                events_builder.remove()

        else:
            logger.info('Ephys builder failed.')
            ephys_builder.remove()
        logger.info('Unwinding.')

    if do_convert:
        ephys_converter = Importer(Importer.CONVERT_EPHYS,**kwargs)
        ephys_success,attempts = attempt_importers([ephys_converter],force_eeg)
        attempted_importers.extend(attempts)
        successes.append(ephys_success)
        if ephys_success:
            events_converter = Importer(Importer.CONVERT_EVENTS,**kwargs)
            events_success,attempts = attempt_importers([events_converter],force_events)
            attempted_importers.extend(attempts)
            successes.append(events_success)
            if not events_success:
                logger.info('Event Conversion failed. Unwinding.')
                events_converter.remove()
        else:
            logger.info('Events %s conversion failed')
            ephys_converter.remove()
    return all(successes), ImporterCollection(attempted_importers)


def run_montage_import(kwargs, do_create= True,do_convert = False,force=False):
    logger.set_subject(kwargs['subject'], kwargs['protocol'])
    logger.set_label('Montage Importer')
    importers = []
    if do_create:
        importers.append( Importer(Importer.CREATE_MONTAGE,**kwargs))
    if do_convert:
        importers.append( Importer(Importer.CONVERT_MONTAGE, **kwargs))
    success, importers = attempt_importers(importers, force)
    return success, ImporterCollection(importers)

def run_localization_import(kwargs, force=False,force_dykstra=False):
    logger.set_subject(kwargs['subject'], kwargs['protocol'])
    logger.set_label("Localization importer")
    localization_kwargs = {k:kwargs[k] for k in ['subject','protocol','localization','code']}

    new_importer = Importer(Importer.LOCALIZATION, is_new=True, force_dykstra=force_dykstra,**localization_kwargs)
    old_importer = Importer(Importer.LOCALIZATION, is_new=False,force_dykstra=force_dykstra,**localization_kwargs)
    success, importers = attempt_importers([new_importer, old_importer], force)
    return success, ImporterCollection(importers)


this_dir = os.path.realpath(os.path.dirname(__file__))


def build_session_inputs(subject, new_experiment, session, info):
    experiment = info.get('original_experiment', new_experiment)
    original_session = info.get('original_session', session)
    is_sys1 = info.get('system_1', False)
    is_sys2 = info.get('system_2', False)
    is_sys3 = info.get('system_3', False)
    montage = info.get('montage', '0.0')
    code = info.get('code', subject)
    protocol = info.get('protocol', 'r1')
    attempt_import = info.get('attempt_import', True)
    attempt_conversion = info.get('attempt_conversion', True)

    montage_num = montage.split('.')[1]
    localization = montage.split('.')[0]

    if experiment.startswith('PS'):
        ram_experiment = 'RAM_PS'
    else:
        if experiment.startswith('catFR'):
            ram_experiment = 'RAM_{}'.format(experiment[0].capitalize() + experiment[1:])
        else:
            ram_experiment = 'RAM_{}'.format(experiment)

    inputs = dict(
        protocol=protocol,
        subject=subject,
        montage=montage,
        montage_num=montage_num,
        localization=localization,
        experiment=experiment,
        new_experiment=new_experiment,
        ram_experiment=ram_experiment,
        code=code,
        session=session,
        original_session=original_session,
        groups=(protocol,),
        attempt_conversion=attempt_conversion,
        attempt_import=attempt_import
    )

    if is_sys3:
        inputs['groups'] += ('system_3',)
    if is_sys2 or new_experiment in ('FR3', 'PAL3', 'catFR3', 'TH3', 'PS2.1'):
        inputs['groups'] += ('system_2',)
    elif is_sys1:
        inputs['groups'] += ('system_1',)

    if experiment.startswith('PS') or experiment.startswith('TH') or experiment.startswith('YC'):
        inputs['do_math'] = False

    if experiment.startswith('FR') or experiment.startswith('catFR') or experiment.startswith('PAL'):
        inputs['groups'] += ('verbal', )

    if experiment.endswith("3"):
        inputs['groups'] += ('stim', )

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
                    protocol='ltp' if subject.startswith('LTP') else 'r1' if subject.startswith('R') else None
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
                if 'protocol' not in info:
                    info['protocol'] = 'ltp' if subject.startswith('LTP') else 'r1' if subject.startswith('R') else None
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
            logger.set_subject(inputs['subject'],inputs['protocol'])
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
            if inputs['protocol'] != 'ltp':  # Skip montage import for LTP participants
                success, importers = run_montage_import(inputs, do_convert=True,force=force)
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

    with fileutil.open_with_perms(log_file, 'w') as output:
        output.write("Successful imports: {}\n".format(len(sorted_successes)))
        output.write("Failed imports: {}\n\n".format(len(sorted_failures)))
        output.write("###### FAILURES #####\n")
        for failure in sorted_failures:
            output.write('{}\n\n'.format(failure.describe()))
        output.write("\n###### SUCCESSES #####\n")
        for success in sorted_successes:
            output.write('{}\n\n'.format(success.describe()))

    return sorted_failures


def get_code_montage(code, protocol='r1'):
    r1 = load_index(protocol)
    try:
        localization_num = r1.get_value('localization', subject_alias=code)
        montage_num = r1.get_value('montage', subject_alias=code)
        return '{}.{}'.format(localization_num, montage_num)
    except ValueError:
        return None


def show_imported_experiments(subject, protocol='r1'):
    r1 = load_index(protocol)
    experiments = r1.experiments(subject=subject)
    if not experiments:
        print('No sessions for this subject')
    for experiment in experiments:
        show_imported_sessions(subject, experiment, protocol)


def show_imported_sessions(subject, experiment, protocol='r1', show_info=False):
    r1 = load_index(protocol)
    sessions = r1.sessions(subject=subject, experiment=experiment)
    print('| Existing {} sessions for {}'.format(experiment, subject))
    if not sessions:
        print('None')
    for session in sessions:
        code = r1.get_value('subject_alias', subject=subject, experiment=experiment, session=session)
        try:
            orig_sess = r1.get_value('original_session', subject=subject, experiment=experiment, session=session)
        except ValueError:
            orig_sess = session
        if code != subject:
            montage_str = ' ({}.{})'.format(r1.get_value('montage', subject=subject, experiment=experiment, session=session),
                                         r1.get_value('localization', subject=subject, experiment=experiment, session=session))
        else:
            montage_str = ''
        import_type = r1.get_value('import_type', subject=subject, experiment=experiment, session=session)
        if import_type == 'build':
            import_type = ''
        else:
            import_type = ' ({})'.format(import_type)
        print('|- {sess}: ({code}{montage}, {exp}_{orig_sess}{type})'.format(sess=session, code=code, exp=experiment, orig_sess=orig_sess,
                                                                             montage=montage_str, type=import_type))

LOADED_INDEXES = {}
def load_index(protocol):
    if protocol not in LOADED_INDEXES:
        index_file = os.path.join(paths.db_root, 'protocols', '{}.json'.format(protocol))
        if not os.path.exists(index_file):
            with open(index_file, 'w') as f:
                json.dump({}, f)
        LOADED_INDEXES[protocol] = JsonIndexReader(index_file)
    return LOADED_INDEXES[protocol]


def get_next_orig_session(code, experiment, protocol='r1'):
    index = load_index(protocol)
    orig_sessions = list(index.aggregate_values('original_session', subject_alias=code, experiment=experiment))
    if orig_sessions:
        return max([int(s) for s in orig_sessions]) + 1
    else:
        return 0


def get_next_new_session(subject, experiment, protocol='r1'):
    index = load_index(protocol)
    sessions = index.sessions(subject=subject, experiment=experiment)
    if len(sessions) > 0:
        return max([int(s) for s in sessions]) + 1
    else:
        return 0


def prompt_for_session_inputs(inputs, **opts):

    code = inputs.code
    if code is None:
        code = input('Enter subject code: ')

    subject = inputs.subject
    if subject is None:
        subject = re.sub(r'_.*', '', code)

    experiment = inputs.experiment
    if experiment is None:
        experiment = input('Enter experiment name: ')
    if re.search('Cat',experiment):
        inputs.original_experiment = experiment
        experiment = re.sub(r'Cat',r'cat',experiment)
        logger.debug('Original experiment: %s'%inputs.original_experiment)

    protocol = inputs.protocol
    if protocol is None:
        protocol = 'ltp' if experiment.startswith('ltp') else \
                   'r1' if subject.startswith('R') else None
    groups = (protocol,)

    montage = inputs.montage
    if montage is None:
        montage = get_code_montage(code, protocol)

    if not montage:
        montage = input('Montage for this subject not found in database. Enter montage as #.#: ')

    montage_num = montage.split('.')[1]

    localization = montage.split('.')[0]

    show_imported_sessions(subject, experiment, protocol)

    suggested_session = get_next_orig_session(code, experiment, protocol)

    original_session = inputs.original_session or inputs.session
    if original_session is None:
        original_session = input('Enter original session number (suggested: {}): '.format(suggested_session))

    session = inputs.session
    if session is None:
        if opts.get('change_session', False) or subject != code or \
                (experiment.startswith('PS') and not experiment.endswith('2.1')):
            suggested_session = get_next_new_session(subject, experiment, protocol)
            session = int(input('Enter new session number (suggested: {}): '.format(suggested_session)))
        else:
            session = original_session

    original_experiment = inputs.original_experiment or inputs.experiment
    if original_experiment is None:
        if opts.get('change_experiment', False):
            original_experiment = input('Enter original experiment: ')
        elif experiment == 'PS2.1':
            is_sys3 = confirm("Is this a system 3 session? ")
            if is_sys3:
                groups+= ('system_3',)
                original_experiment = 'PS2'
            else:
                groups += ('system_2',)
                original_experiment = 'PS21'
        elif experiment.startswith('PS'):
            if '_' not in experiment:
                original_experiment = 'PS'
            else:
                original_experiment = experiment
        else:
            original_experiment = experiment

    attempt_conversion = opts.get('allow_convert', False)
    attempt_import = not opts.get('force_convert', False)
    do_compare = opts.get('do_compare', False)

    if original_experiment.startswith('PS'):
        ram_experiment = 'RAM_PS'
    else:
        if original_experiment.startswith('catFR'):
            ram_experiment = 'RAM_{}'.format(experiment[0].capitalize() + experiment[1:])
        else:
            ram_experiment = 'RAM_{}'.format(experiment)



    inputs = dict(
        protocol=protocol,
        subject=subject,
        montage=montage,
        montage_num=montage_num,
        localization=localization,
        experiment=original_experiment,
        new_experiment=experiment,
        ram_experiment=ram_experiment,
        force=False,
        do_compare=do_compare,
        code=code,
        session=session,
        original_session=original_session,
        groups=groups,
        attempt_import=attempt_import,
        attempt_conversion=attempt_conversion,
        PS4 = ('ps4' in groups)
    )

    if opts.get('sys2', False):
        inputs['groups'] += ('system_2',)
    elif opts.get('sys1', False):
        inputs['groups'] += ('system_1',)

    if any(experiment.startswith(exp) for exp in ['PS', 'TH', 'location']):
        inputs['do_math'] = False

    if experiment.startswith('FR') or experiment.startswith('catFR') or experiment.startswith('PAL'):
        inputs['groups'] += ('verbal',)

    return inputs


def prompt_for_montage_inputs():
    code = input('Enter original subject code (including _#): ')
    subject = re.sub(r'_.*', '', code)

    montage = input('Enter montage as #.#: ')

    montage_num = montage.split(".")[1]
    subject_split = code.split("_")
    subject_montage = subject_split[1] if len(subject_split)>1 else "0"

    if ( subject_montage ) != montage_num:
        print("WARNING: subject code {} does not match montage number {} !".format(subject_montage, montage_num))
        confirmed = confirm("Are you sure you want to continue? ")
        if not confirmed:
            return False
    reference_scheme = ''
    while not reference_scheme.startswith('m') and not reference_scheme.startswith('b'):
        reference_scheme = input('Enter reference scheme \n (monopolar or bipolar) : ')

    if reference_scheme.startswith('m'):
        reference_scheme = 'monopolar'
    elif reference_scheme.startswith('b'):
        reference_scheme = 'bipolar'


    inputs = dict(
        subject=subject,
        montage=montage,
        code=code,
        protocol='r1',
        reference_scheme = reference_scheme
    )

    return inputs


def prompt_for_localization_inputs():
    code = input('Enter original subject code (including _#): ')
    subject = re.sub(r'_.*', '', code)

    localization = ''
    while not localization.isdigit():
        localization = input("Enter localization number: ")

    inputs = dict(
        code = code,
        subject = subject,
        localization = localization,
        protocol = 'r1'
    )

    return inputs

def session_exists(protocol, subject, experiment, session):
    session_dir = os.path.join(paths.db_root, 'protocols', protocol,
                                 'subjects', subject,
                                 'behavioral', experiment,
                                 'sessions', str(session))
    behavioral_current = os.path.join(session_dir, 'behavioral', 'current_processed')
    eeg_current = os.path.join(session_dir, 'ephys', 'current_processed')

    return os.path.exists(behavioral_current) and os.path.exists(eeg_current)

def localization_exists(protocol, subject, localization):
    neurorad_current = os.path.join(paths.db_root, 'protocols', protocol,
                           'subjects', subject,
                           'localizations', localization,
                           'neuroradiology', 'current_processed')
    return os.path.exists(neurorad_current)

def montage_exists(protocol, subject, montage):
    montage_num = montage.split('.')[1]
    localization = montage.split('.')[0]
    montage_dir = os.path.join(paths.db_root, 'protocols', protocol,
                               'subjects', subject,
                               'localizations', localization,
                               'montages', montage_num)
    neurorad_current = os.path.join(montage_dir, 'neuroradiology', 'current_processed')

    return os.path.exists(neurorad_current)


def main():
    if config.log_debug:
        logger.set_stdout_level(0)

    if not config.show_plots:
        import matplotlib
        matplotlib.use('agg')

    if config.clean_db:
        print('Cleaning database and ignoring other arguments')
        clean_task = CleanDbTask()
        clean_task.run()
        print('Database cleaned. Exiting.')
        exit(0)

    if config.aggregate:
        print('Aggregating index files and ignoring other arguments')
        aggregate_task = IndexAggregatorTask()
        aggregate_task.run()
        print('Indexes aggregated. Exiting.')
        exit(0)

    if config.db.name or config.db.experiment:
        if config.db.name:
            db_builder = IMPORT_DB_BUILDERS[config.db.name]
            print('Building import database: {}'.format(config.db.name))
        else: #config.db.experiment
            db_builder = lambda:build_json_import_db('%s_import.json'%config.db.experiment,
                                                     orig_experiments=[config.db.experiment])
            print('Building import database: {}'.format(config.db.experiment))
        db_builder()
        print('DB built. Exiting.')
        exit(0)

    attempt_convert = config.allow_convert or config.force_convert
    attempt_import = not config.force_convert

    if config.json_file:
        print('Running from JSON file: {}'.format(config.json_file))
        import_log = 'json_import'
        i=1
        while os.path.exists(import_log + '.log'):
            import_log = 'json_import' + str(i)
            i += 1
        import_log = import_log + '.log'
        failures = run_json_import(config.json_file, attempt_import, attempt_convert,
                                   config.force_events, config.force_eeg, config.force_montage, import_log)
        if failures:
            print('\n******************\nSummary of failures\n******************\n')
            print('\n\n'.join([failure.describe() for failure in failures]))
        else:
            print('No failures.')
        print("Aggregating indexes. This may take a moment...")
        IndexAggregatorTask().run()
        print('Log created: {}. Exiting'.format(import_log))
        exit(0)

    if config.montage_only:
        inputs = prompt_for_montage_inputs()
        if not inputs:
            exit(0)
        if montage_exists(inputs['protocol'], inputs['subject'], inputs['montage']):
            if not confirm('{subject}, montage {montage} already exists. Continue and overwrite? '.format(**inputs)):
                print('Import aborted! Exiting.')
                exit(0)
        print('Importing montage')
        attempt_create = not config.force_convert
        attempt_convert = config.force_convert or config.allow_convert
        force = config.force_montage or config.force_convert
        success, importer = run_montage_import(inputs, attempt_create,attempt_convert, force=force)
        print('Success:' if success else 'Failed:')
        print(importer.describe())
        exit(0)

    if config.localization_only:
        inputs = prompt_for_localization_inputs()
        if not inputs:
            exit(0)
        exists = localization_exists(inputs['protocol'], inputs['subject'], inputs['localization'])
        logger.debug('Path %s %s'%(os.path.join(paths.db_root, 'protocols', inputs['protocol'],
                           'subjects', inputs['subject'],
                           'localization', inputs['localization'],
                           'neuroradiology', 'current_processed'),'exists' if exists else 'does not exist'))
        if localization_exists(inputs['protocol'], inputs['subject'], inputs['localization']):
            if not confirm('{subject}, loc {localization} already exists. Continue and overwrite? '.format(**inputs)):
                print("Import aborted! Exiting.")
                exit(0)
        print("Importing localization")
        success, importer = run_localization_import(inputs, config.force_localization,config.force_dykstra)
        print('Success:' if success else 'Failed')
        print(importer.describe())
        exit(0)

    if config.view_only:
        code = input("Enter subject code: ")
        subject = re.sub(r'_.*', '', code)
        show_imported_experiments(subject)
        exit(0)

    inputs = prompt_for_session_inputs(**config.options)


    if session_exists(inputs['protocol'], inputs['subject'], inputs['new_experiment'], inputs['session']):
        if not confirm('{subject} {new_experiment} session {session} already exists. '
                       'Continue and overwrite? '.format(**inputs)):
            print('Import aborted! Exiting.')
            exit(0)
    print('Importing session')
    success, importers = run_session_import(inputs, attempt_import, attempt_convert, config.force_events,
                                        config.force_eeg)
    if success:
        print("Aggregating indexes...")
        IndexAggregatorTask().run_single_subject(inputs['subject'], inputs['protocol'])
        print("Requesting report")
        ReportLaunchTask(subject=inputs['code'],experiment=inputs['experiment'],session=inputs['session']).request()
    print('Success:' if success else "Failed:")
    print(importers.describe())
    exit(0)

if __name__ == "__main__":
    main()
