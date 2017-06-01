import json
import os
import re
import shutil
import traceback

from . import files
from .configuration import paths
from .events_tasks import SplitEEGTask, MatlabEEGConversionTask, MatlabEventConversionTask, \
                  EventCreationTask, CompareEventsTask, EventCombinationTask, \
                  MontageLinkerTask, RecognitionFlagTask
from .neurorad_tasks import LoadVoxelCoordinatesTask, CorrectCoordinatesTask, CalculateTransformsTask, \
                           AddContactLabelsTask, AddMNICoordinatesTask, WriteFinalLocalizationTask
from .parsers.base_log_parser import get_version_num
from .parsers.ltpfr2_log_parser import LTPFR2SessionLogParser
from .parsers.ltpfr_log_parser import LTPFRSessionLogParser
from .parsers.mat_converter import MathMatConverter
from .parsers.ps_log_parser import PS4Sys3LogParser
from .parsers.math_parser import MathLogParser
from .transfer_config import TransferConfig
from .tasks import ImportJsonMontageTask, CleanLeafTask
from .transferer import generate_ephys_transferer, generate_session_transferer, generate_localization_transferer,\
                       generate_montage_transferer, UnTransferrableException, TRANSFER_INPUTS, find_sync_file
from .loggers import logger

GROUPS = {
    'FR': ('verbal', 'stim'),
    'PAL': ('verbal', 'stim'),
    'catFR': ('verbal', 'stim'),
    'PS': ('stim',),
    'ltpFR': ('verbal',)
}

MATLAB_CONVERSION_TYPE = 'MATLAB_CONVERSION'
SOURCE_IMPORT_TYPE = 'IMPORT'

N_PS4_SESSIONS = 10

def determine_groups(protocol, subject, experiment, session, transfer_cfg_file, *args, **kwargs):
    exp_type = re.sub(r'[^A-Za-z]', '', experiment)

    groups = (protocol,)
    if exp_type in GROUPS:
        groups += GROUPS[exp_type]
    groups += (exp_type, experiment)

    groups += tuple(args)

    if 'PS4' in experiment:
        groups += ('PS4',)
    if protocol == 'r1' and experiment.endswith('5') and 'PS4' not in groups:
        groups += ('recog',)

    if protocol == 'r1' and 'system_1' not in groups and 'system_2' not in groups and 'system_3' not in groups:
        kwargs['original_session'] = session
        inputs = dict(protocol=protocol,
                      subject=subject,
                      code=subject,
                      session=session,
                      experiment=experiment,
                      **kwargs)
        inputs.update(**paths.options)

        systems = ('system_1', 'system_2', 'system_3_1','system_3_0')

        for sys in systems:
            try:
                print "I AM HERE!"
                logger.info("Checking if this system is {}".format(sys))
                transfer_cfg = TransferConfig(transfer_cfg_file, groups + (sys,), **inputs)
                transfer_cfg.locate_origin_files()

                missing_files = transfer_cfg.missing_required_files()

                if len(missing_files) > 0:
                    names = [missing_file.name for missing_file in missing_files]
                    logger.info("Determined due to missing files ({}) that this system is not {}".format(names, sys))
                    continue

                match = r1_system_match(experiment, transfer_cfg, sys)

                if match:
                    logger.info("Making a very educated guess that this system is {}".format(sys))
                    break
                else:
                    logger.info("Determined from log files that this system is not {}".format(sys))

            except Exception as e:
                logger.debug("This system is probably not {} due to error: {}".format(sys, e))
                continue
        else:
            logger.debug("System_# determination failed. I'm a failure. Nobody loves me.")
            logger.error("Could not determine system of r1 subject. Continuing as if system is {}"
                         ", but this will likely fail very soon!".format(sys))
        groups += (sys,)

        #
        # session_log = transfer_cfg.get_file('session_log')
        # eeg_log = transfer_cfg.get_file('eeg_log')
        #
        # if experiment.startswith('TH'):
        #     if eeg_log is not None:
        #         if len(open(eeg_log.origin_paths[0]).read().strip()) == 0:
        #             groups += ('system_2',)
        #         else:
        #             groups += ('system_1',)
        #     else:
        #         groups += ('system_2',)
        # elif session_log is not None:
        #     if len(session_log.origin_paths) < 1:
        #         logger.warn("Could not find session log file! Assuming system_1 ")
        #         groups += ('system_1', )
        #     else:
        #         version = get_version_num(session_log.origin_paths[0])
        #         if version >= 3:
        #             groups += ('system_3')
        #         elif version >= 2:
        #             groups += ('system_2',)
        #         else:
        #             groups += ('system_1',)

        if experiment.endswith("3") or experiment.endswith("5"):
            groups += ("stim", )
    return groups

def r1_system_match(experiment, transfer_cfg, sys):
    """
    Determines whether the provided system_# matches the provided TransferConfig given the presence of system log files
    It is liberal with matching -- if it could possibly be that system, it returns True
    :param experiment:
    :param transfer_cfg:
    :param sys:
    :return:
    """
    session_log = transfer_cfg.get_file('session_log')
    eeg_log = transfer_cfg.get_file('eeg_log')
    if experiment.startswith('TH'):
        if eeg_log is not None:
            with open(eeg_log.origin_paths[0]) as eeg_file:
                if len(eeg_file.read().strip()) > 0:
                    logger.debug("This appears to be system_1 because the eeg_log is not empty")
                    return sys == 'system_1'
                else:
                    logger.debug("This appears to not be system_1 because the eeg_log is empty")
                    return sys != 'system_1'
        else:
            return sys != 'system_1'

    if session_log is not None:
        if len(session_log.origin_paths) > 0:
            try:
                version = get_version_num(session_log.origin_paths[0])
                logger.debug("The version number in session_log is {}".format(version))
                if version >= 3:
                    logger.debug("This appears to be system_3 due to version number")
                    return sys == 'system_3_0'
                elif version >= 2:
                    logger.debug("This appears to be system_2 due to version number")
                    return sys == 'system_2'
            except Exception as e:
                logger.debug("Error trying to get version number: {}".format(e))
                return sys=='system_3_1'

    return True





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
        self.stored_objects = {}
        self.output_files = {}
        self.output_info = info
        self.on_failure = lambda: CleanLeafTask(False).run([], self.destination)

    def previous_transfer_type(self):
        return self.transferer.previous_transfer_type()

    def current_transfer_type(self):
        return self.transferer.transfer_type

    def register_output(self, filename, label):
        self.output_files[label] = os.path.join(self.current_dir, filename)

    def register_info(self, info_key, info_value):
        self.output_info[info_key] = info_value

    def store_object(self, name, item):
        self.stored_objects[name] = item

    def retrieve_object(self, name):
        return self.stored_objects[name]

    @property
    def source_dir(self):
        return self.transferer.destination_labelled

    @property
    def source_label(self):
        return self.transferer.get_label()

    @property
    def processed_label(self):
        return '{}_processed'.format(self.transferer.label)

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

    def _initialize(self, force=False):
        if not os.path.exists(self.destination):
            files.makedirs(self.destination)
        logger.set_label('{} Transfer initialization'.format(self.current_transfer_type()))

        logger.info('Transfer pipeline to {} started'.format(self.destination_root))
        missing_files = self.transferer.missing_files()
        if missing_files:
            logger.warn("Missing files {}. "
                         "Deleting processed folder {}".format([f.name for f in missing_files], self.destination))
            shutil.rmtree(self.destination)
            raise UnTransferrableException('Missing file {}: '
                                           'expected in {}'.format(missing_files[0].name,
                                                                   missing_files[0].formatted_origin_dir))

        should_transfer = not self.transferer.matches_existing_checksum()
        if should_transfer != True:
            logger.info('No changes to transfer...')
            if not os.path.exists(self.current_dir):
                logger.info('{} does not exist! Continuing anyway!'.format(self.current_dir))
            elif force:
                logger.info('Forcing transfer to happen anyway')
            else:
                self.transferer.transfer_aborted = True
                logger.debug('Removing processed folder {}'.format(self.destination))
                logger.info('Transfer pipeline ended without transfer')
                try:
                    shutil.rmtree(self.destination)
                except OSError:
                    logger.warn('Could not remove destination {}'.format(self.destination))
                return False
        return True

    def _execute_tasks(self):
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

    def run(self, force=False):
        try:
            if not self._initialize(force):
                self.on_failure()
                return
            self._execute_tasks()
            logger.info('Transfer pipeline ended normally')
            self.create_index()
        except Exception as e:
            self.on_failure()
            raise

# def build_wav_pipeline(subject,experiment,session,protocol,**kwargs):
#     transferer = generate_wav_transferer(subject,experiment,session,protocol,code=subject)
#     transferer.set_transfer_type(SOURCE_IMPORT_TYPE)
#     return TransferPipeline(transferer)


def build_split_pipeline(subject, montage, experiment, session, protocol='r1', groups=tuple(), code=None,
                         original_session=None, new_experiment=None, **kwargs):
    logger.set_label("Building EEG Splitter")
    new_experiment = new_experiment if not new_experiment is None else experiment

    groups = determine_groups(protocol, code, experiment, original_session,
                              TRANSFER_INPUTS['ephys'], 'transfer', *groups, **kwargs)

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
    logger.set_label("Building EEG Converter")
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

    logger.set_label("Building Event Creator")


    original_session = kwargs['original_session'] if 'original_session' in kwargs else session
    code = code or subject

    try:
        kwargs['sync_folder'], kwargs['sync_filename'] = \
            find_sync_file(code, experiment, original_session)
    except:
        logger.debug("Couldn't find sync pulses, which is fine unless this is system_1")

    groups =  determine_groups(protocol, code, experiment, original_session,
                               TRANSFER_INPUTS['behavioral'], 'transfer', *groups, **kwargs)

    transferer = generate_session_transferer(subject, experiment, session, protocol, groups,
                                             code=code, **kwargs)
    transferer.set_transfer_type(SOURCE_IMPORT_TYPE)

    system = [x for x in groups if 'system' in x][0]
    system = system.partition('_')[-1]

    if protocol == 'r1':
        tasks = [MontageLinkerTask(protocol, subject, montage,critical=('3' in system))]

        tasks.append(EventCreationTask(protocol, subject, montage, experiment, session, system,**kwargs))
    elif protocol == 'ltp':
        if experiment == 'ltpFR':
            tasks = [EventCreationTask(protocol, subject, montage, experiment, session, False, parser_type=LTPFRSessionLogParser)]
        elif experiment == 'ltpFR2':
            tasks = [EventCreationTask(protocol, subject, montage, experiment, session, False, parser_type=LTPFR2SessionLogParser)]
        else:
            raise Exception('Unknown experiment %s under protocol \'ltp')
    else:
        raise Exception('Unknown protocol %s' % protocol)

    other_events = ()

    if 'ps4' in groups:
        tasks.append(EventCreationTask(protocol, subject, montage, experiment, session, system,
                                       'ps4',parser_type=PS4Sys3LogParser,**kwargs))
        other_events+=('ps4',)

    if do_math:
        tasks.append(EventCreationTask(protocol, subject, montage, experiment, session, system,
                                       'math', MathLogParser, critical=False,**kwargs))
        other_events+=('math',)

    if other_events:
        tasks.append(EventCombinationTask(('task',)+other_events, critical=False,))

    # if 'ps4' in groups:
    #     if kwargs.get('new_experiment')=='PS4':
    #         comparator = lambda events: events.list<N_PS4_SESSIONS
    #     else:
    #         comparator = lambda events: events.list>= N_PS4_SESSIONS
    #     tasks.append(PruneEventsTask(comparator))

    if 'recog' in groups:
        tasks.append(RecognitionFlagTask(critical=False))

    if do_compare:
        tasks.append(CompareEventsTask(subject, montage, experiment, session, protocol, code, original_session,
                                       match_field=kwargs['match_field'] if 'match_field' in kwargs else None))

    localization = montage.split('.')[0]
    montage_num = montage.split('.')[1]

    # Have to wait to aggregate index until after submission
    # tasks.append(IndexAggregatorTask())

    if protocol == 'ltp':
        info = dict(
            subject_alias=code,
            import_type='build',
            original_session=original_session
        )
    else:
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

    logger.set_label("Building Event Converter")

    if experiment[:-1] == 'catFR':
        experiment = 'CatFR' + experiment[-1]
        new_experiment = 'catFR' + experiment[-1]

    if 'groups' in kwargs:
        no_group_kwargs = {k:v for k,v in kwargs.items() if k not in ('groups', )}
    else:
        no_group_kwargs = kwargs

    new_groups = determine_groups(protocol, code, experiment, original_session,
                                         TRANSFER_INPUTS['behavioral'], 'conversion', **no_group_kwargs)
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
    if protocol == 'ltp':
        info = dict(
            subject_alias=code,
            import_type='conversion',
            original_session=original_session
        )
    else:
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

def build_import_localization_pipeline(subject, protocol, localization, code, is_new):

    logger.set_label("Building Localization Creator")

    transferer = generate_localization_transferer(subject, protocol, localization, code, is_new)

    tasks = [
        LoadVoxelCoordinatesTask(subject, localization, is_new),
        CalculateTransformsTask(subject, localization),
        CorrectCoordinatesTask(subject, localization),
        AddContactLabelsTask(subject, localization),
        AddMNICoordinatesTask(subject, localization),
        WriteFinalLocalizationTask()

    ]

    return TransferPipeline(transferer, *tasks)


def build_import_montage_pipeline(subject, montage, protocol, code):
    transferer = generate_montage_transferer(subject, montage, protocol, code)

    tasks = [ImportJsonMontageTask(subject, montage)]
    return TransferPipeline(transferer, *tasks)


def test_split_sys3():
    pipeline = build_split_pipeline('R9999X', 0.0, 'FR1', 1, groups=('r1', 'transfer', 'system_3'), localization=0, montage_num=0)
    pipeline.run()

def test_create_sys3_events():
    pipeline = build_events_pipeline('R9999X', '0.0', 'FR1', 1, True, 'r1',
                                     new_experiment='FR1',
                                     localization=0, montage_num=0,
                                     code='R9999X',
                                     original_session=1, sync_folder='', sync_filename='')
    pipeline.run()

if __name__ == '__main__':
    logger.set_stdout_level(0)
    test_split_sys3()
    test_create_sys3_events()
