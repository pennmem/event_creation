import re
import os
import numpy as np
import glob
import traceback

from ptsa.data.readers import BaseEventReader

from alignment.system1 import System1Aligner
from alignment.system2 import System2Aligner
from alignment.system3 import System3Aligner
from alignment.LTPAligner import LTPAligner

from parsers.fr_log_parser import FRSessionLogParser
from parsers.pal_log_parser import PALSessionLogParser
from parsers.catfr_log_parser import CatFRSessionLogParser
from parsers.math_parser import MathLogParser
from parsers.base_log_parser import EventComparator
from parsers.ps_log_parser import PSLogParser
from parsers.th_log_parser import THSessionLogParser
from parsers.thr_log_parser import THSessionLogParser as THRSessionLogParser
from parsers.base_log_parser import StimComparator, EventCombiner
from parsers.mat_converter import FRMatConverter, MatlabEEGExtractor, PALMatConverter, \
                                  CatFRMatConverter, PSMatConverter, MathMatConverter, YCMatConverter

from readers.eeg_reader import get_eeg_reader

from viewers.view_recarray import to_json, from_json

from detection.artifact_detection import ArtifactDetector

from configuration import paths

import files

from tasks import PipelineTask, UnProcessableException
from loggers import logger

from tests.test_event_creation import SYS1_COMPARATOR_INPUTS, SYS2_COMPARATOR_INPUTS, \
    SYS1_STIM_COMPARISON_INPUTS, SYS2_STIM_COMPARISON_INPUTS, LTP_COMPARATOR_INPUTS


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

        if self.protocol == 'ltp':
            for i in range(len(raw_eeg_groups)):
                raw_eeg = raw_eeg_groups[i]
                reader = get_eeg_reader(raw_eeg, None)
                split_eeg_filename = self.SPLIT_FILENAME.format(subject=self.subject,
                                                                experiment=self.experiment,
                                                                session=self.session,
                                                                time=reader.get_start_time_string())
                # Split raw data file by channel & apply postprocessing
                reader.split_data(os.path.join(self.pipeline.destination), split_eeg_filename)
                bad_chans = reader.find_bad_chans(files['artifact_log'][i], threshold=600000) if 'artifact_log' in files else np.array([])
                np.savetxt(os.path.join(self.pipeline.destination, 'bad_chans.txt'), bad_chans, fmt='%s')
                # Calculate common average reference and save it to a file
                reader.reref(bad_chans, os.path.join(self.pipeline.destination, 'reref'))

            # Detect EGI channel files
            num_split_files = len(glob.glob(os.path.join(self.pipeline.destination, 'noreref', '*.[0-9]*')))
            # Detect Biosemi channel files
            num_split_files += len(glob.glob(os.path.join(self.pipeline.destination, 'noreref', '*.[A-Z]*')))
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
            num_split_files = len(glob.glob(os.path.join(db_folder, 'noreref', '*.[0-9]*')))

        if num_split_files == 0:
            raise UnProcessableException(
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

    PARSERS = {
        'FR': FRSessionLogParser,
        'PAL': PALSessionLogParser,
        'catFR': CatFRSessionLogParser,
        'math': MathLogParser,
        'PS': PSLogParser,
        'TH': THSessionLogParser,
        'THR': THRSessionLogParser
    }

    def __init__(self, protocol, subject, montage, experiment, session, r1_sys_num=0, event_label='task',
                 parser_type=None, critical=True, **kwargs):
        super(EventCreationTask, self).__init__(critical)
        self.name = '{label} Event Creation for {exp}_{sess}'.format(label=event_label, exp=experiment, sess=session)
        self.parser_type = parser_type or self.PARSERS[re.sub(r'\d', '', experiment)]
        self.protocol = protocol
        self.subject = subject
        self.montage = montage
        self.experiment = experiment
        self.session = session
        self.r1_sys_num = r1_sys_num
        self.kwargs = kwargs
        self.event_label = event_label
        self.filename = '{label}_events.json'.format(label=event_label)
        self.pipeline = None

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        parser = self.parser_type(self.protocol, self.subject, self.montage, self.experiment, self.session, files)
        unaligned_events = parser.parse()
        if self.protocol == 'ltp':
                aligner = LTPAligner(unaligned_events, files, db_folder)
                events = aligner.align()
                artifact_detector = ArtifactDetector(events, aligner.root_names, aligner.noreref_dir,
                                                     aligner.reref_dir)
                events = artifact_detector.run()
        elif self.r1_sys_num in (2, 3):
            if self.r1_sys_num == 2:
                aligner = System2Aligner(unaligned_events, files, db_folder)
            else:
                aligner = System3Aligner(unaligned_events, files, db_folder)

            if self.event_label != 'math':
                logger.debug("Adding stimulation events")
                aligner.add_stim_events(parser.event_template, parser.persist_fields_during_stim)

            if self.experiment.startswith("TH"):
                start_type = "CHEST"
            elif self.experiment == 'PS21':
                start_type = 'NP_POLL'
            else:
                start_type = "SESS_START"
            events = aligner.align(start_type)
        elif self.r1_sys_num == 1:
            aligner = System1Aligner(unaligned_events, files)
            events = aligner.align()
        else:
            raise UnProcessableException("r1_sys_num must be in (1, 3) for protocol==r1. Current value: {}".format(self.r1_sys_num))
        events = parser.clean_events(events) if events.shape != () else events
        self.create_file(self.filename, to_json(events),
                         '{}_events'.format(self.event_label))

class NoEventsError(Exception):
    pass

class PruneEventsTask(PipelineTask):
    def __init__(self,cond):
        super(PruneEventsTask,self).__init__()
        self.filter = cond

    def _run(self,files,db_folder):
        event_files = glob.glob(os.path.join(db_folder,'*_events.json'))
        for fid in event_files:
            events = from_json(fid)
            filtered_events = events[self.filter(events)]
            if len(filtered_events) == 0 or events is None:
                logger.info('No events for this experiment. If there are subsequent PS4 sessions, do not panic.')
                raise NoEventsError()
            events = to_json(events)
            self.create_file(fid,events,os.path.splitext(os.path.basename(fid))[0])


class EventCombinationTask(PipelineTask):

    COMBINED_LABEL='all'

    def __init__(self, event_labels, sort_field='mstime', critical=True):
        super(EventCombinationTask, self).__init__(critical)
        self.name = 'Event combination: {}'.format(event_labels)
        self.event_labels = event_labels
        self.sort_field = sort_field

    def _run(self, files, db_folder):
        event_files = [os.path.join(db_folder, '{}_events.json'.format(label)) for label in self.event_labels]
        event_files = [f for f in event_files if os.path.isfile(f)]
        events = [from_json(event_file) for event_file in event_files]
        combiner = EventCombiner(events)
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
                raise UnProcessableException("Cannot find montage for {} in {}".format(self.subject, fullfile))
            logger.info('File {} found'.format(file))
            self.pipeline.register_info(name, fullfile)

class MatlabEventConversionTask(PipelineTask):

    CONVERTERS = {
        'FR': FRMatConverter,
        'PAL': PALMatConverter,
        'catFR': CatFRMatConverter,
        'PS': PSMatConverter,
        'YC': YCMatConverter,
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
        self.original_session = original_session if not original_session is None else session
        self.kwargs = kwargs
        self.event_label = event_label
        self.filename = '{label}_events.json'.format(label=event_label)
        self.pipeline = None

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        converter = self.converter_type(self.protocol, self.subject, self.montage, self.experiment, self.session,
                                        self.original_session, files)
        events = converter.convert()

        self.create_file( self.filename, to_json(events),
                         '{}_events'.format(self.event_label))

class ImportEventsTask(PipelineTask):

    PARSERS = {
        'FR': FRSessionLogParser,
        'PAL': PALSessionLogParser,
        'catFR': CatFRSessionLogParser,
        'math': MathLogParser,
        'PS': PSLogParser
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
        self.original_session = int(original_session if not original_session is None else session)
        self.montage = montage
        self.experiment = experiment
        self.session = session
        self.protocol = protocol
        self.match_field = match_field if match_field else 'mstime'

    def get_matlab_event_file(self):
        if self.protocol == 'r1':
            ram_exp = 'RAM_{}'.format(self.experiment[0].upper() + self.experiment[1:])
            event_directory = os.path.join(paths.rhino_root, 'data', 'events', ram_exp, '{}_events.mat'.format(self.code))
        elif self.protocol == 'ltp':
            event_directory = os.path.join(paths.rhino_root, 'data', 'eeg', 'scalp', 'ltp', self.experiment, self.code, 'session_{}'.format(self.original_session), 'events.mat')
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
