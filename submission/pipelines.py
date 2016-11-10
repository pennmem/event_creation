

from transferer import Transferer, generate_ephys_transferer, generate_session_transferer,\
                       generate_montage_transferer, UnTransferrableException, TRANSFER_INPUTS, DATA_ROOT

from tasks import SplitEEGTask, MatlabEEGConversionTask, MatlabEventConversionTask, \
                  EventCreationTask, CompareEventsTask, EventCombinationTask, \
                  ImportJsonMontageTask, IndexAggregatorTask, MontageLinkerTask, CleanDbTask

from parsers.base_log_parser import get_version_num

from parsers.mat_converter import MathMatConverter

from parsers.math_parser import MathLogParser
from parsers.ltpfr_log_parser import LTPFRSessionLogParser
from parsers.ltpfr2_log_parser import LTPFR2SessionLogParser

from loggers import logger
import files

import json
import shutil
import os
import re
import traceback

GROUPS = {
    'FR': ('verbal', 'stim'),
    'PAL': ('verbal', 'stim'),
    'catFR': ('verbal', 'stim'),
    'PS': ('stim',),
    'ltpFR': ('verbal',)
}

MATLAB_CONVERSION_TYPE = 'MATLAB_CONVERSION'
SOURCE_IMPORT_TYPE = 'IMPORT'

def determine_groups(protocol, subject, experiment, session, group_dict, *args, **kwargs):
    exp_type = re.sub(r'[^A-Za-z]', '', experiment)

    groups = (protocol,)
    if exp_type in GROUPS:
        groups += GROUPS[exp_type]
    groups += (exp_type, experiment)

    groups += tuple(args)


    if protocol == 'r1':
        source_file_info = Transferer.load_groups(group_dict, groups)
        if experiment.startswith('TH'):
            if 'eeg_log' in source_file_info:
                eeg_log_info = source_file_info['eeg_log']
                eeg_log_file = Transferer.get_origin_files(eeg_log_info,
                                                           protocol=protocol,
                                                           subject=subject,
                                                           code=subject,
                                                           experiment=experiment,
                                                           original_session=session,
                                                           data_root=DATA_ROOT)[0]
                if len(open(eeg_log_file).read().strip()) == 0:
                    groups += ('system_2',)
                else:
                    groups += ('system_1',)
            else:
                groups += ('system_2',)
        elif 'session_log' in source_file_info:
            session_log_info = source_file_info['session_log']
            session_log_file = Transferer.get_origin_files(session_log_info,
                                                           protocol=protocol,
                                                           subject=subject,
                                                           code=subject,
                                                           experiment=experiment,
                                                           original_session=session,
                                                           data_root=DATA_ROOT)[0]
            version = get_version_num(session_log_file)
            if version >= 2:
                groups += ('system_2',)
            else:
                groups += ('system_1',)
        if experiment.endswith("3"):
            groups += ("stim", )
    return groups

class TransferPipeline(object):

    CURRENT_PROCESSED_DIRNAME = 'current_processed'
    INDEX_FILE = 'index.json'

    def __init__(self, transferer, *pipeline_tasks, **info):
        self.transferer = transferer
        self.pipeline_tasks = pipeline_tasks
        self.exports = {}
        self.destination_root = self.transferer.destination_root
        self.destination = os.path.join(self.destination_root, self.processed_label)
        self.current_dir = os.path.join(self.destination_root, self.CURRENT_PROCESSED_DIRNAME)
        if not os.path.exists(self.destination):
            files.makedirs(self.destination)
        for task in self.pipeline_tasks:
            task.set_pipeline(self)
        self.log_filenames = [
            os.path.join(self.destination_root, 'log.txt'),
        ]
        self.output_files = {}
        self.output_info = info

    def previous_transfer_type(self):
        return self.transferer.previous_transfer_type()

    def current_transfer_type(self):
        return self.transferer.transfer_type

    def register_output(self, filename, label):
        self.output_files[label] = os.path.join(self.current_dir, filename)

    def register_info(self, info_key, info_value):
        self.output_info[info_key] = info_value

    @property
    def source_label(self):
        return self.transferer.get_label()

    @property
    def processed_label(self):
        return '{}_processed'.format(self.transferer.get_label())

    def create_index(self):
        index = {}
        if len(self.output_files) > 0:
            index['files'] = {}
            for name, path in self.output_files.items():
                index['files'][name] = os.path.relpath( path, self.current_dir)
        if len(self.output_info) > 0:
            index['info'] = self.output_info
        if len(index) > 0:
            with files.open_with_perms(os.path.join(self.current_dir, self.INDEX_FILE), 'w') as f:
                json.dump(index, f, indent=2, sort_keys=True)

    def run(self, force=False):
        if not os.path.exists(self.destination):
            files.makedirs(self.destination)
        logger.set_label('{} Transfer initialization'.format(self.current_transfer_type()))

        logger.info('Transfer pipeline to {} started'.format(self.destination_root))
        missing_files, expected_dir = self.transferer.missing_required_files()
        if missing_files:
            logger.error("Missing file {}. Deleting processed folder {}".format(missing_files, self.destination))
            shutil.rmtree(self.destination)
            raise UnTransferrableException('Missing file {}. '
                                           'Expected in {}'.format(missing_files, expected_dir))

        should_transfer = self.transferer.check_checksums()
        if should_transfer != True:
            logger.info('No changes to transfer...')
            if not os.path.exists(self.current_dir):
                logger.info('{} does not exist! Continuing anyway!'.format(self.current_dir))
            else:
                self.transferer.transfer_aborted = True
                if not force:
                    logger.debug('Removing processed folder {}'.format(self.destination))
                    logger.info('Transfer pipeline ended without transfer')
                    try:
                        shutil.rmtree(self.destination)
                    except OSError:
                        logger.warn('Could not remove destination {}'.format(self.destination))
                    return
                else:
                    logger.info('Forcing transfer to happen anyway')

        logger.set_label('Transfer in progress')
        transferred_files = self.transferer.transfer_with_rollback()
        pipeline_task = None
        try:
            for i, pipeline_task in enumerate(self.pipeline_tasks):

                logger.info('Executing task {}: {}'.format(i+1, pipeline_task.name))
                logger.set_label(pipeline_task.name)
                pipeline_task.run(transferred_files, self.destination)
                logger.info('Task {} finished successfully'.format(pipeline_task.name))

            if os.path.islink(self.current_dir):
                os.unlink(self.current_dir)
            os.symlink(self.processed_label, self.current_dir)

        except Exception as e:
            logger.error('Task {} failed with message {}, Rolling back transfer'.format(pipeline_task.name if pipeline_task else
                                                                   'initialization', e))
            traceback.print_exc()

            self.transferer.remove_transferred_files()
            logger.debug('Transfer pipeline errored: {}'.format(e.message))
            logger.debug('Removing processed folder {}'.format(self.destination))
            if os.path.exists(self.destination):
                shutil.rmtree(self.destination)
            raise

        logger.info('Transfer pipeline ended normally')
        self.create_index()


def build_split_pipeline(subject, montage, experiment, session, protocol='r1', groups=tuple(), code=None,
                         original_session=None, new_experiment=None, **kwargs):
    new_experiment = new_experiment if not new_experiment is None else experiment
    transferer = generate_ephys_transferer(subject, experiment, session, protocol, groups + ('transfer',),
                                           code=code,
                                           original_session=original_session,
                                           new_experiment=new_experiment,
                                           **kwargs)
    transferer.set_transfer_type(SOURCE_IMPORT_TYPE)
    task = SplitEEGTask(subject, montage, new_experiment, session, protocol, **kwargs)
    return TransferPipeline(transferer, task)


def build_convert_eeg_pipeline(subject, montage, experiment, session, protocol='r1', code=None,
                               original_session=None, new_experiment=None, **kwargs):
    new_experiment = new_experiment if not new_experiment is None else experiment
    if experiment[:-1] == 'catFR':
        experiment = 'CatFR'+experiment[-1]

    kwargs['groups'] = kwargs['groups'] + ('conversion',) if 'groups' in kwargs else ('conversion',)

    transferer = generate_ephys_transferer(subject, experiment, session, protocol,
                                           code=code,
                                           original_session=original_session, new_experiment=new_experiment, **kwargs)
    transferer.set_transfer_type(MATLAB_CONVERSION_TYPE)

    tasks = [MatlabEEGConversionTask(subject, experiment, original_session)]


    return TransferPipeline(transferer, *tasks)


def build_events_pipeline(subject, montage, experiment, session, do_math=True, protocol='r1', code=None,
                          groups=tuple(), do_compare=False, **kwargs):

    original_session = kwargs['original_session'] if 'original_session' in kwargs else session
    code = code or subject

    groups +=  determine_groups(protocol, code, experiment, original_session,
                                json.load(open(TRANSFER_INPUTS['behavioral'])), 'transfer')

    transferer = generate_session_transferer(subject, experiment, session, protocol, groups,
                                             code=code, **kwargs)
    transferer.set_transfer_type(SOURCE_IMPORT_TYPE)
    if protocol == 'r1':
        tasks = [MontageLinkerTask(protocol, subject, montage)]
        tasks.append(EventCreationTask(protocol, subject, montage, experiment, session, 'system_2' in groups))
    elif protocol == 'ltp':
        if experiment == 'ltpFR':
            do_math = False
            tasks = [EventCreationTask(protocol, subject, montage, experiment, session, False, parser_type=LTPFRSessionLogParser)]
        elif experiment == 'ltpFR2':
            tasks = [EventCreationTask(protocol, subject, montage, experiment, session, False, parser_type=LTPFR2SessionLogParser)]
        else:
            raise Exception('Unknown experiment %s under protocol \'ltp')
    else:
        raise Exception('Unknown protocol %s' % protocol)

    if do_math:
        tasks.append(EventCreationTask(protocol, subject, montage, experiment, session, 'system_2' in groups,
                                       'math', MathLogParser, critical=False))
        tasks.append(EventCombinationTask(('task', 'math'), critical=False))

    if do_compare:
        tasks.append(CompareEventsTask(subject, montage, experiment, session, protocol, code, original_session,
                                       match_field=kwargs['match_field'] if 'match_field' in kwargs else None))

    localization = montage.split('.')[0]
    montage_num = montage.split('.')[1]

    # Have to wait to aggregate index until after submission
    # tasks.append(IndexAggregatorTask())

    info = dict(
        localization=localization,
        montage=montage_num,
        subject_alias=code,
        import_type='build'
    )
    if original_session != session:
        info['original_session'] = original_session
    if 'new_experiment' in kwargs and kwargs['new_experiment'] != experiment:
        info['original_experiment'] = experiment

    return TransferPipeline(transferer, *tasks, **info)


def build_convert_events_pipeline(subject, montage, experiment, session, do_math=True, protocol='r1', code=None,
                                  original_session=None, new_experiment=None, **kwargs):
    if experiment[:-1] == 'catFR':
        experiment = 'CatFR' + experiment[-1]
        new_experiment = 'catFR' + experiment[-1]

    new_groups = determine_groups(protocol, code, experiment, original_session,
                                         json.load(open(TRANSFER_INPUTS['behavioral'])),
                                         'conversion')
    kwargs['groups'] = kwargs['groups'] + new_groups if 'groups' in kwargs else new_groups

    new_experiment = new_experiment if not new_experiment is None else experiment
    transferer = generate_session_transferer(subject, experiment, session, protocol,
                                             code=code, original_session=original_session,
                                             new_experiment=new_experiment, **kwargs)
    transferer.set_transfer_type(MATLAB_CONVERSION_TYPE)

    if protocol == 'r1':
        tasks = [MontageLinkerTask(protocol, subject, montage)]
    else:
        tasks = []

    tasks.append(MatlabEventConversionTask(protocol, subject, montage, new_experiment, session,
                                       original_session=original_session, **kwargs))

    if do_math:
        tasks.append(MatlabEventConversionTask(protocol, subject, montage, new_experiment, session,
                                               event_label='math', converter_type=MathMatConverter,
                                               original_session=original_session, critical=False, **kwargs ))
        tasks.append(EventCombinationTask(('task', 'math'), critical=False))

    localization = montage.split('.')[0]
    montage_num = montage.split('.')[1]
    # Have to wait to aggregate index until after submission
    #tasks.append(IndexAggregatorTask())

    info = dict(
        localization=localization,
        montage=montage_num,
        subject_alias=code,
        import_type='conversion'
    )
    if original_session != session:
        info['original_session'] = original_session
    if experiment != new_experiment:
        info['original_experiment'] = experiment

    return TransferPipeline(transferer, *tasks, **info)


def build_import_montage_pipeline(subject, montage, protocol, code):
    transferer = generate_montage_transferer(subject, montage, protocol, code)

    tasks = [ImportJsonMontageTask(subject, montage)]
    return TransferPipeline(transferer, *tasks)

