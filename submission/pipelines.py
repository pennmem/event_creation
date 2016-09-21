

from transferer import Transferer, generate_ephys_transferer, generate_session_transferer,\
                       generate_montage_transferer, UnTransferrableException, TRANSFER_INPUTS

from tasks import SplitEEGTask, MatlabEEGConversionTask, MatlabEventConversionTask, \
                  EventCreationTask, CompareEventsTask, EventCombinationTask, \
                  ImportJsonMontageTask, IndexAggregatorTask, MontageLinkerTask

from parsers.base_log_parser import get_version_num

from parsers.mat_converter import MathMatConverter

from parsers.math_parser import MathLogParser
from loggers import log, logger

import json
import shutil
import os
import re

GROUPS = {
    'FR': ('verbal', 'stim'),
    'PAL': ('verbal', 'stim'),
    'catFR': ('verbal', 'stim'),
    'PS': ('stim',)
}

def determine_groups(protocol, experiment, group_dict):
    exp_type = re.sub('\d', '', experiment)

    groups = tuple()
    if exp_type in GROUPS:
        groups += GROUPS[exp_type]
    groups += (exp_type, experiment)


    if protocol == 'r1':
        source_files = Transferer.load_groups(group_dict, groups)
        if 'session_log' in source_files:
            version = get_version_num(source_files['session_log'])
            if version >= 2:
                groups += ('system_2')
            else:
                groups += ('system_1')

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
            os.makedirs(self.destination)
        for task in self.pipeline_tasks:
            task.set_pipeline(self)
        self.log_filenames = [
            os.path.join(self.destination_root, 'log.txt'),
            os.path.join(self.destination, 'log.txt')
        ]
        self.output_files = {}
        self.output_info = info

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

    def start_logging(self):
        try:
            logger.add_log_files(*self.log_filenames)
        except:
            log('Could not set logging path.')

    def create_index(self):
        index = {}
        if len(self.output_files) > 0:
            index['files'] = {}
            for name, path in self.output_files.items():
                index['files'][name] = os.path.relpath( path, self.current_dir)
        if len(self.output_info) > 0:
            index['info'] = self.output_info
        if len(index) > 0:
            json.dump(index, open(os.path.join(self.current_dir, self.INDEX_FILE), 'w'),
                      indent=2, sort_keys=True)

    def stop_logging(self):
        logger.remove_log_files(*self.log_filenames)

    def run(self, force=False):
        logger.set_label('Transfer initialization')
        self.start_logging()

        log('Transfer pipeline to {} started'.format(self.destination_root))
        missing_files = self.transferer.missing_required_files()
        if missing_files:
            log("Missing file {}. Deleting processed folder {}".format(missing_files, self.destination), 'CRITICAL')
            self.stop_logging()
            shutil.rmtree(self.destination)
            raise UnTransferrableException('Missing file {}'.format(missing_files))

        should_transfer = self.transferer.check_checksums()
        if should_transfer != True:
            log('No changes to transfer...')
            if not os.path.exists(self.current_dir):
                log('{} does not exist! Continuing anyway!'.format(self.current_dir))
            else:
                self.transferer.transfer_aborted = True
                if not force:
                    log('Removing processed folder {}'.format(self.destination))
                    log('Transfer pipeline ended without transfer')
                    self.stop_logging()
                    shutil.rmtree(self.destination)
                    return
                else:
                    log('Forcing transfer to happen anyway')

        logger.set_label('Transfer in progress')
        transferred_files = self.transferer.transfer_with_rollback()
        pipeline_task = None
        try:
            for i, pipeline_task in enumerate(self.pipeline_tasks):

                log('Executing task {}: {}'.format(i+1, pipeline_task.name))
                pipeline_task.run(transferred_files, self.destination)
                log('Task {} finished successfully'.format(pipeline_task.name))

            if os.path.islink(self.current_dir):
                os.unlink(self.current_dir)
            os.symlink(self.processed_label, self.current_dir)

        except Exception as e:
            log('Task {} failed with message {}, \nRolling back transfer'.format(pipeline_task.name if pipeline_task else
                                                                   'initialization', e), 'CRITICAL')


            self.transferer.remove_transferred_files()
            log('Transfer pipeline errored: {}'.format(e.message), 'CRITICAL')
            log('Removing processed folder {}'.format(self.destination), 'CRITICAL')
            self.stop_logging()
            if os.path.exists(self.destination):
                shutil.rmtree(self.destination)
            raise

        log('Transfer pipeline ended normally')
        self.create_index()
        self.stop_logging()


def build_split_pipeline(subject, montage, experiment, session, protocol='r1', groups=tuple(), code=None,
                         original_session=None, new_experiment=None, **kwargs):
    new_experiment = new_experiment if not new_experiment is None else experiment
    transferer = generate_ephys_transferer(subject, experiment, session, protocol, groups + ('transfer',),
                                           code=code,
                                           original_session=original_session,
                                           new_experiment=new_experiment,
                                           **kwargs)
    task = SplitEEGTask(subject, montage, experiment, session, **kwargs)
    return TransferPipeline(transferer, task)


def build_convert_eeg_pipeline(subject, montage, experiment, session, protocol='r1', code=None,
                               original_session=None, new_experiment=None, **kwargs):
    new_experiment = new_experiment if not new_experiment is None else experiment
    kwargs['groups'] += ('conversion',)

    transferer = generate_ephys_transferer(subject, experiment, session, protocol,
                                           code=code,
                                           original_session=original_session, new_experiment=new_experiment, **kwargs)
    task = MatlabEEGConversionTask(subject, experiment, original_session)
    return TransferPipeline(transferer, task)


def build_events_pipeline(subject, montage, experiment, session, do_math=True, protocol='r1', code=None,
                          groups=tuple(), do_compare=False, **kwargs):

    groups +=  determine_groups(protocol, experiment, json.load(open(TRANSFER_INPUTS['behavioral'])))

    transferer = generate_session_transferer(subject, experiment, session, protocol, groups + ('transfer',),
                                             code=code, **kwargs)

    if protocol == 'r1':
        tasks = [MontageLinkerTask(protocol, subject, montage)]
    else:
        tasks = []

    tasks.append(EventCreationTask(protocol, subject, montage, experiment, session, 'system_2' in groups))

    if do_math:
        tasks.append(EventCreationTask(protocol, subject, montage, experiment, session, 'system_2' in groups,
                                       'math', MathLogParser))
        tasks.append(EventCombinationTask(('task', 'math')))

    if do_compare:
        tasks.append(CompareEventsTask(subject, montage, experiment, session, protocol, code,
                                       kwargs['original_session'] if 'original_session' in kwargs else None,
                                       match_field=kwargs['match_field'] if 'match_field' in kwargs else None))


    tasks.append(IndexAggregatorTask())
    return TransferPipeline(transferer, subject_alias=code, *tasks)


def build_convert_events_pipeline(subject, montage, experiment, session, do_math=True, protocol='r1', code=None,
                                  original_session=None, new_experiment=None, **kwargs):
    new_experiment = new_experiment if not new_experiment is None else experiment
    kwargs['groups'] += ('conversion',)
    transferer = generate_session_transferer(subject, experiment, session, protocol,
                                             code=code, original_session=original_session,
                                             new_experiment=new_experiment, **kwargs)

    tasks = [MatlabEventConversionTask(protocol, subject, montage, experiment, session,
                                       original_session=original_session, **kwargs)]

    if do_math:
        tasks.append(MatlabEventConversionTask(protocol, subject, montage, experiment, session,
                                               event_label='math', converter_type=MathMatConverter,
                                               original_session=original_session, **kwargs))
        tasks.append(EventCombinationTask(('task', 'math')))

    localization = montage.split('.')[0]
    montage_num = montage.split('.')[1]
    tasks.append(IndexAggregatorTask())

    return TransferPipeline(transferer, localization=localization, montage=montage_num, subject_alias=code, *tasks)


def build_import_montage_pipeline(subject, montage, protocol, code):
    transferer = generate_montage_transferer(subject, montage, protocol, code)

    tasks = [ImportJsonMontageTask(subject, montage)]

    return TransferPipeline(transferer, subject_alias=code, *tasks)

