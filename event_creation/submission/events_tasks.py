import glob
import os
import re
import traceback
import requests
import json
import numpy as np
from ptsa.data.readers import BaseEventReader, JsonIndexReader

from ..tests.test_event_creation import SYS1_COMPARATOR_INPUTS, SYS2_COMPARATOR_INPUTS, \
    SYS1_STIM_COMPARISON_INPUTS, SYS2_STIM_COMPARISON_INPUTS, LTP_COMPARATOR_INPUTS

from .alignment.LTPAligner import LTPAligner
from .alignment.system1 import System1Aligner
from .alignment.system2 import System2Aligner
from .alignment.system3 import System3Aligner
from .configuration import paths
from .cleaning.artifact_detection import ArtifactDetector
from .cleaning.lcf import run_lcf
from .parsers.ltpfr_log_parser import LTPFRSessionLogParser
from .parsers.ltpfr2_log_parser import LTPFR2SessionLogParser
from .parsers.raa_log_parser import RAASessionLogParser
from .parsers.vffr_log_parser import VFFRSessionLogParser
from .parsers.prelim_log_parser import PrelimSessionLogParser
from .parsers.base_log_parser import EventComparator
from .parsers.base_log_parser import StimComparator, EventCombiner
from .parsers.catfr_log_parser import CatFRSessionLogParser
from .parsers.fr_log_parser import FRSessionLogParser
from .parsers.fr_sys3_log_parser import FRSys3LogParser, catFRSys3LogParser
from .parsers.mat_converter import FRMatConverter, MatlabEEGExtractor, PALMatConverter, \
                                  CatFRMatConverter, PSMatConverter, MathMatConverter, YCMatConverter, \
                                  THMatConverter
from .parsers.pal_log_parser import PALSessionLogParser
from .parsers.pal_sys3_log_parser import PALSys3LogParser
from .parsers.ps_log_parser import PSLogParser
from .parsers.th_log_parser import THSessionLogParser
from .parsers.thr_log_parser import THSessionLogParser as THRSessionLogParser
from .parsers.math_parser import MathSessionLogParser
from .parsers.hostpc_parsers import FRHostPCLogParser, catFRHostPCLogParser,\
        TiclFRParser
from .readers.eeg_reader import get_eeg_reader
from .tasks import PipelineTask
from .quality.util import get_time_field

from .viewers.recarray import to_json, from_json
from .log import logger
from .exc import NoEventsError, ProcessingError


class SplitEEGTask(PipelineTask):

    SPLIT_FILENAME = '{subject}_{experiment}_{session}_{time}'

    def __init__(self, subject, montage, experiment, session, protocol, critical=True, **kwargs):
        super(SplitEEGTask, self).__init__(critical)
        self.name = 'Splitting {exp}_{sess}'.format(exp=experiment, sess=session)
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.protocol = protocol
        self.kwargs = kwargs

    @staticmethod
    def group_ns2_files(raw_eegs):
        raw_eeg_groups = []
        for raw_eeg in raw_eegs:
            for group in raw_eeg_groups:
                if raw_eeg.replace('np1', 'np2') == group[0].replace('np1', 'np2'):
                    group.append(raw_eeg)
                    break
            else:
                raw_eeg_groups.append([raw_eeg])
        raw_eeg_groups = [group[0] if len(group) == 1 else group for group in raw_eeg_groups]
        return raw_eeg_groups

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        raw_eegs = files['raw_eeg']
        if not isinstance(raw_eegs, list):
            raw_eegs = [raw_eegs]

        raw_eeg_groups = self.group_ns2_files(raw_eegs)

        # Scalp EEG post-processing
        if self.protocol == 'ltp':
            # Process each EEG recording separately, if multiple exist
            success = np.zeros(len(raw_eeg_groups), dtype=bool)
            for i, raw_eeg in enumerate(raw_eeg_groups):
                # Only use .raw if no corresponding .mff exists (or if the .mff could not be read), as MNE is able to
                # read .mff recordings much faster than .raw
                if raw_eeg.endswith('.raw'):
                    has_mff_version = False
                    for j, other_eeg in enumerate(raw_eeg_groups):
                        if other_eeg.endswith('.mff') and other_eeg.startswith(raw_eeg[:15]) and success[j]:
                            has_mff_version = True
                    if has_mff_version:
                        continue
                # Create EEG reader
                reader = get_eeg_reader(raw_eeg, None)
                # Set up ephys directory (note: does not actually "split" the data for scalp EEG sessions)
                success[i] = reader.process_eeg(os.path.join(self.pipeline.destination))

        # RAM post-processing
        else:
            if 'experiment_config' in files:
                jacksheet_files = files['experiment_config']  # Jacksheet embedded in hdf5 file
            elif 'contacts' in files:
                jacksheet_files = [files['contacts']] * len(raw_eeg_groups)
            elif 'jacksheet' in files:
                jacksheet_files = [files['jacksheet']] * len(raw_eeg_groups)
            else:
                raise KeyError("Cannot find jacksheet mapping! No 'contacts' or 'jacksheet'!")

            channel_map = files.get('channel_map')

            for raw_eeg, jacksheet_file in zip(raw_eeg_groups, jacksheet_files):
                try:
                    reader = get_eeg_reader(raw_eeg, jacksheet_file, channel_map_filename=channel_map)
                except KeyError as k:
                    traceback.print_exc()
                    logger.warn('Cannot split file with extension {}'.format(k))
                    continue

                split_eeg_filename = self.SPLIT_FILENAME.format(subject=self.subject,
                                                                experiment=self.experiment,
                                                                session=self.session,
                                                                time=reader.get_start_time_string())
                reader.split_data(db_folder, split_eeg_filename)
            num_split_files = (len(glob.glob(os.path.join(db_folder, 'noreref', '*.[0-9]*')))
                               + len(glob.glob(os.path.join(db_folder, 'noreref', '*.h5'))))

            if num_split_files == 0:
                raise ProcessingError(
                    'Seems like splitting did not properly occur. No split files found in {}. Check jacksheet'.format(
                        db_folder))


class MatlabEEGConversionTask(PipelineTask):

    def __init__(self, subject, experiment, original_session, critical=True, **kwargs):
        super(MatlabEEGConversionTask, self).__init__(critical)
        self.name = 'matlab EEG extraction {exp}_{sess}'.format(exp=experiment,
                                                                sess=original_session)
        self.original_session = original_session
        self.kwargs = kwargs

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        extractor = MatlabEEGExtractor(self.original_session, files)
        extractor.copy_ephys(db_folder)


class EventCreationTask(PipelineTask):

    @classmethod
    def R1_PARSERS(cls, sys_num):
        if sys_num <= 3.0:
            return {
                'FR': FRSessionLogParser,
                'PAL': PALSessionLogParser,
                'catFR': CatFRSessionLogParser,
                'PS': PSLogParser,  # which has its own dispatching system ...
                'TH': THSessionLogParser,
                'THR': THRSessionLogParser,
            }
        elif sys_num < 3.3:
            return {
                'FR': FRSys3LogParser,
                'catFR': catFRSys3LogParser,
                'PS': PSLogParser,
                'PS_FR': PSLogParser,
                'PS_catFR': PSLogParser,
                'PAL': PALSys3LogParser,
                'THR': THRSessionLogParser,
            }
        elif sys_num == 3.3:
            return {
                'FR': FRHostPCLogParser,
                'catFR': catFRHostPCLogParser,
                'PS': PSLogParser,
                'PS_FR': PSLogParser,
                'PS_catFR': PSLogParser,
                'PAL': PALSys3LogParser,
            }
        elif sys_num == 3.4:
            return {
                'FR': FRHostPCLogParser,
                'catFR': catFRHostPCLogParser,
                'PS': PSLogParser,
                'PS_FR': PSLogParser,
                'PS_catFR': PSLogParser,
                'PAL': PALSys3LogParser,
                'TICL_FR': FRHostPCLogParser,
                'TICL_catFR': TiclFRParser,
                'LocationSearch': PSLogParser,
            }

        else:
            raise KeyError

    LTP_PARSERS = {
                    'ltpFR': LTPFRSessionLogParser,
                    'ltpFR2': LTPFR2SessionLogParser,
                    'FR1': FRSessionLogParser,
                    'Remembering_Across_America': RAASessionLogParser,
                    'VFFR': VFFRSessionLogParser,
                    'prelim': PrelimSessionLogParser
                  }

    @property
    def r1_sys_num(self):
        if not self._r1_sys_num:
            return 0.0
        return float(self._r1_sys_num.replace('_', '.'))

    @property
    def parser_type(self):
        if self._parser_type is None:
            if self.protocol == 'r1':
                if self.event_label == 'math':
                    new_experiment = 'math'
                else:
                    new_experiment = self.kwargs.get('new_experiment') or self.experiment
                try:
                    self._parser_type = self.R1_PARSERS(self.r1_sys_num)[re.sub(r'[\d.]', '', new_experiment)]
                except KeyError:
                    raise KeyError('Experiment %s not supported for system %s' % (new_experiment, self.r1_sys_num))
            elif self.protocol == 'ltp':
                self._parser_type = self.LTP_PARSERS[self.experiment]
        return self._parser_type

    def __init__(self, protocol, subject, montage, experiment, session, r1_sys_num='', event_label='task',
                 parser_type=None, critical=True, **kwargs):
        super(EventCreationTask, self).__init__(critical)
        experiment = kwargs.get('new_experiment') or experiment
        self.name = '{label} Event Creation for {exp}_{sess}'.format(label=event_label, exp=experiment, sess=session)
        self.protocol = protocol
        self.subject = subject
        self.montage = montage
        self.experiment = experiment
        self.session = session
        self._r1_sys_num = r1_sys_num
        self.kwargs = kwargs
        self.event_label = event_label
        self.filename = '{label}_events.json'.format(label=event_label)
        self.pipeline = None
        self._parser_type = parser_type

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        logger.debug('self._parser_type is %s' % (None if not self._parser_type else str(self._parser_type)))
        if self.r1_sys_num >= 3:
            with open(files['event_log'][0]) as event_log:
                self._r1_sys_num = json.load(event_log)['versions']['Ramulator'].rpartition('.')[0].replace('.', '_')

        parser = self.parser_type(self.protocol, self.subject, self.montage, self.experiment, self.session, files)
        logger.debug('Using %s' % str(self.parser_type))
        unaligned_events = parser.parse()
        # SCALP LAB SPECIFIC PROCESSING - Alignment, blink detection, and data cleaning
        if self.protocol == 'ltp':
            sync_log = files['eeg_log'] if 'eeg_log' in files else []
            ephys_dir = os.path.join(os.path.dirname(os.path.dirname(db_folder)), 'ephys', 'current_processed')
            # Align scalp EEG data with events
            aligner = LTPAligner(unaligned_events, sync_log, ephys_dir)
            events = aligner.align()
            # Detect trials contaminated by blinks and other artifacts
            artifact_detector = ArtifactDetector(events, aligner.eeg, ephys_dir, self.experiment)
            del aligner
            events = artifact_detector.run()
            # Create a cleaned version of the EEG data using localized component filtering
            run_lcf(events, artifact_detector.eeg, ephys_dir, method='fastica', highpass_freq=.5,
                    badchan_method='interpolate', iqr_thresh=3, lcf_winsize=.1)
            del artifact_detector
        # RAM SPECIFIC PROCESSING - Alignment
        elif self.protocol == 'r1':
            self.pipeline.register_info('system_version', self.r1_sys_num)
            if self.event_label == 'ps4' or not parser.DO_ALIGNMENT:
                events = unaligned_events
            else:
                if self.r1_sys_num == 1.0:
                    aligner = System1Aligner(unaligned_events, files)
                    events = aligner.align()
                else:
                    if self.r1_sys_num == 2.0:
                        aligner = System2Aligner(unaligned_events, files, db_folder)
                    elif 3.0 <= self.r1_sys_num <= 3.4:
                        aligner = System3Aligner(unaligned_events, files, db_folder)
                    else:
                        raise ProcessingError(
                            "r1_sys_num must be in (1, 3.3) for protocol==r1. Current value: {}".format(
                                self.r1_sys_num))

                    if parser.ADD_STIM_EVENTS:
                        logger.debug("Adding stimulation events")
                        aligner.add_stim_events(parser.event_template, parser.persist_fields_during_stim)

                    if self.experiment.startswith("TH"):
                        start_type = "CHEST"
                    elif self.experiment == 'PS21':
                        start_type = 'NP_POLL'
                    else:
                        start_type = "SESS_START"
                    if parser.DO_ALIGNMENT:
                        events = aligner.align(start_type)
                    else:
                        events = unaligned_events
                    if issubclass(type(aligner), System3Aligner):
                        aligner.apply_eeg_file(events)

        events = parser.clean_events(events) if events.shape != () else events
        self.pipeline.importer.tests.extend(parser.check_event_quality(events, files))
        self.create_file(self.filename, to_json(events),
                         '{}_events'.format(self.event_label))


class PruneEventsTask(PipelineTask):
    def __init__(self, cond):
        super(PruneEventsTask, self).__init__()
        self.filter = cond

    def _run(self, files, db_folder):
        event_files = glob.glob(os.path.join(db_folder, '*_events.json'))
        for fid in event_files:
            events = from_json(fid)
            filtered_events = events[self.filter(events)]
            if len(filtered_events) == 0 or events is None:
                logger.info('No events for this experiment. If there are subsequent PS4 sessions, do not panic.')
                raise NoEventsError()
            filtered_events = to_json(filtered_events)
            self.create_file(fid, filtered_events, os.path.splitext(os.path.basename(fid))[0])


class RecognitionFlagTask(PipelineTask):
    def _run(self, files, db_folder):
        event_file = os.path.join(db_folder, 'task_events.json')
        events = from_json(event_file)
        self.pipeline.register_info('Recognition', any(['RECOG' in tipe for tipe in np.unique(events.type)]))


class ReportLaunchTask(PipelineTask):

    def __init__(self, subject, experiment, session):
        super(ReportLaunchTask, self).__init__(critical=False)
        self.subject = subject
        self.experiment = experiment
        self.session = session

    def request(self):
        from configuration import paths
        api_url = paths.report_url
        parameters = {
            'username': 'cmlbrainbuilder',
            'password': 'BoBtheBuilder',
            'subject': self.subject,
            'experiment': self.experiment,
            'sessions': self.session,
            'joint_report': False,
            'rerun': True,
            'use_classifier_excluded_leads': False,
            'save_location': '/scratch/report_database/html_reports',  # TODO: CHANGE/PARAMETRIZE THIS
            'report_database': '/scratch/report_database'
        }
        response = None
        error = None
        try:
            response = requests.post(api_url, data=parameters)
            succeeded = response.status_code == 200
        except requests.RequestException as error:
            succeeded = False
        if not succeeded:
            if response is not None:
                logger.error('Request failed with error code %s' % response.status_code)
            else:
                logger.error('Request failed with message %s' % str(error))

    def _run(self, files, db_folder):
        self.request()


class EventCombinationTask(PipelineTask):

    COMBINED_LABEL = 'all'

    def __init__(self, event_labels, sort_field=None, critical=True):
        super(EventCombinationTask, self).__init__(critical)
        self.name = 'Event combination: {}'.format(event_labels)
        self.event_labels = event_labels
        self.sort_field = sort_field

    def _run(self, files, db_folder):
        if self.sort_field is None:
            sort_field = get_time_field(files)
        else:
            sort_field = self.sort_field
        event_files = [os.path.join(db_folder, '{}_events.json'.format(label)) for label in self.event_labels]
        event_files = [f for f in event_files if os.path.isfile(f)]
        events = [from_json(event_file) for event_file in event_files]
        combiner = EventCombiner(events, sort_field=sort_field)
        combined_events = combiner.combine()

        self.create_file('{}_events.json'.format(self.COMBINED_LABEL),
                         to_json(combined_events), '{}_events'.format(self.COMBINED_LABEL))


class MontageLinkerTask(PipelineTask):
    """
    Simple task to verify that the appropriate montage files exist for this patient, then
    link them within the pipeline so they will be indicated in the index file
    """

    MONTAGE_PATH = os.path.join('protocols', '{protocol}',
                                'subjects', '{subject}',
                                'localizations', '{localization}',
                                'montages', '{montage}',
                                'neuroradiology', 'current_processed')

    FILES = {'pairs': 'pairs.json',
             'contacts': 'contacts.json'}

    def __init__(self, protocol, subject, montage, critical=True):
        super(MontageLinkerTask, self).__init__(critical)
        self.name = 'Montage linker'
        self.protocol = protocol
        self.subject = subject
        self.montage = montage
        self.localization = montage.split('.')[0]
        self.montage_num = montage.split('.')[1]

    def _run(self, files, db_folder):
        montage_path = self.MONTAGE_PATH.format(protocol=self.protocol,
                                                subject=self.subject,
                                                localization=self.localization,
                                                montage=self.montage_num)
        self.pipeline.register_info('localization', self.localization)
        self.pipeline.register_info('montage', self.montage_num)
        for name, file in self.FILES.items():
            fullfile = os.path.join(montage_path, file)
            if not os.path.exists(os.path.join(paths.db_root, fullfile)):
                raise ProcessingError("Cannot find montage for {} in {}".format(self.subject, fullfile))
            logger.info('File {} found'.format(file))
            self.pipeline.register_info(name, fullfile)


class MatlabEventConversionTask(PipelineTask):
    CONVERTERS = {
        'FR': FRMatConverter,
        'PAL': PALMatConverter,
        'catFR': CatFRMatConverter,
        'PS': PSMatConverter,
        'YC': YCMatConverter,
        'TH': THMatConverter,
    }

    def __init__(self, protocol, subject, montage, experiment, session,
                 event_label='task', converter_type=None, original_session=None, critical=True, **kwargs):
        super(MatlabEventConversionTask, self).__init__(critical)
        self.name = '{label} Event Creation: {exp}_{sess}'.format(label=event_label, exp=experiment, sess=session)
        self.converter_type = converter_type or self.CONVERTERS[re.sub(r'[^A-Za-z]', '', experiment)]
        self.protocol = protocol
        self.subject = subject
        self.montage = montage
        self.experiment = experiment
        self.session = session
        self.original_session = original_session if original_session is not None else session
        self.kwargs = kwargs
        self.event_label = event_label
        self.filename = '{label}_events.json'.format(label=event_label)
        self.pipeline = None

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        converter = self.converter_type(self.protocol, self.subject, self.montage, self.experiment, self.session,
                                        self.original_session, files)
        events = converter.convert()

        self.create_file(self.filename, to_json(events), '{}_events'.format(self.event_label))


class ImportEventsTask(PipelineTask):
    PARSERS = {
        'FR': FRSessionLogParser,
        'PAL': PALSessionLogParser,
        'catFR': CatFRSessionLogParser,
        'math': MathSessionLogParser,
        'PS': PSLogParser,
    }

    CONVERTERS = {
        'FR': FRMatConverter,
        'PAL': PALMatConverter,
        'catFR': CatFRMatConverter,
        'math': MathMatConverter,
        'PS': PSMatConverter
    }

    def __init__(self, protocol, subject, montage, experiment, session, is_sys2, event_label='task',
                 converter_type=None, parser_type=None, original_session=None, critical=True, **kwargs):
        super(ImportEventsTask, self).__init__(critical)
        self.name = '{label} Event Import: {exp}_{sess}'.format(label=event_label, exp=experiment, sess=session)
        self.converter_type = converter_type or self.CONVERTERS[re.sub(r'[^A-Za-z]', '', experiment)]
        self.parser_type = parser_type or self.PARSERS[re.sub(r'[^A-Za-z]', '', experiment)]
        self.protocol = protocol
        self.subject = subject
        self.montage = montage
        self.experiment = experiment
        self.session = session
        self.original_session = original_session
        self.is_sys2 = is_sys2
        self.kwargs = kwargs
        self.event_label = event_label
        self.filename = '{label}_events.json'.format(label=event_label)
        self.pipeline = None

    def _run(self, files, db_folder):
        try:
            EventCreationTask.run(self, files, db_folder)
        except Exception as e:
            logger.error("Exception occurred creating events: {}! Defaulting to event conversion!".format(e))
            MatlabEventConversionTask.run(self, files, db_folder)


class CompareEventsTask(PipelineTask):
    def __init__(self, subject, montage, experiment, session, protocol='r1', code=None, original_session=None,
                 match_field=None, critical=True):
        super(CompareEventsTask, self).__init__(critical)
        self.name = 'Comparator {}_{}'.format(experiment, session)
        self.subject = subject
        self.code = code if code else subject
        self.original_session = int(original_session if original_session is not None else session)
        self.montage = montage
        self.experiment = experiment
        self.session = session
        self.protocol = protocol
        self.match_field = match_field if match_field else 'mstime'

    def get_matlab_event_file(self):
        if self.protocol == 'r1':
            ram_exp = 'RAM_{}'.format(self.experiment[0].upper() + self.experiment[1:])
            event_directory = os.path.join(paths.rhino_root, 'data', 'events', ram_exp,
                                           '{}_events.mat'.format(self.code))
        elif self.protocol == 'ltp':
            event_directory = os.path.join(paths.rhino_root, 'data', 'eeg', 'scalp', 'ltp', self.experiment, self.code,
                                           'session_{}'.format(self.original_session), 'events.mat')
        else:
            raise NotImplementedError('Only R1 and LTP event comparison implemented')

        return os.path.join(event_directory)

    def _run(self, files, db_folder):
        logger.set_label(self.name)

        mat_file = self.get_matlab_event_file()
        if not os.path.exists(mat_file):
            logger.warn("Could not find existing MATLAB file. Not executing comparison!")
            return

        mat_events_reader = \
            BaseEventReader(
                filename=mat_file,
                common_root=paths.rhino_root,
                eliminate_events_with_no_eeg=False,
            )
        logger.debug('Loading matlab events')
        mat_events = mat_events_reader.read()
        mat_session = self.original_session + (1 if self.protocol == 'ltp' else 0)
        self.sess_mat_events = mat_events[mat_events.session == mat_session]  # TODO: dependent on protocol
        new_events = from_json(os.path.join(db_folder, 'task_events.json'))
        if self.protocol == 'r1':
            try:
                major_version = '.'.join(new_events[-1].exp_version.split('.')[:1])
                major_version_num = float(major_version.split('_')[-1])
            except:
                major_version_num = 0
            if major_version_num >= 2 or ((not self.experiment.startswith("PS")) and self.experiment.endswith("3")):
                comparator_inputs = SYS2_COMPARATOR_INPUTS[self.experiment]
            else:
                comparator_inputs = SYS1_COMPARATOR_INPUTS[self.experiment]
        elif self.protocol == 'ltp':
            comparator_inputs = LTP_COMPARATOR_INPUTS[self.experiment]
        else:
            raise NotImplementedError('Only r1 and ltp event comparison implemented')

        comparator = EventComparator(new_events, self.sess_mat_events, **comparator_inputs)
        logger.debug('Comparing events...')

        found_bad, error_message = comparator.compare()

        if found_bad is True:
            logger.error("Event comparison failed!")
            logger.debug(error_message)
            assert False, 'Event comparison failed!'
        else:
            logger.debug('Comparison Success!')

        if self.protocol == 'r1':
            logger.debug('Comparing stim events...')
            try:
                major_version = '.'.join(new_events[-1].exp_version.split('.')[:1])
                if float(major_version.split('_')[-1]) >= 2:
                    comparator_inputs = SYS2_STIM_COMPARISON_INPUTS[self.experiment]
                else:
                    comparator_inputs = SYS1_STIM_COMPARISON_INPUTS[self.experiment]
            except ValueError:
                comparator_inputs = SYS2_STIM_COMPARISON_INPUTS[self.experiment]
            stim_comparator = StimComparator(new_events, self.sess_mat_events, **comparator_inputs)
            errors = stim_comparator.compare()

            if errors:
                logger.error("Stim comparison failed!")
                logger.debug(errors)
                assert False, 'Stim comparison failed!'
            else:
                logger.debug('Stim comparison success!')
