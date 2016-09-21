import os
import glob
import re
import json
import datetime

from parsers.pal_log_parser import PALSessionLogParser
from alignment.system1 import System1Aligner
from alignment.system2 import System2Aligner
from readers.eeg_reader import get_eeg_reader
from viewers.view_recarray import to_json, from_json
from parsers.fr_log_parser import FRSessionLogParser
from parsers.catfr_log_parser import CatFRSessionLogParser
from parsers.math_parser import MathLogParser
from parsers.base_log_parser import EventComparator
from parsers.ps_log_parser import PSLogParser
from parsers.base_log_parser import StimComparator, EventCombiner
from parsers.mat_converter import FRMatConverter, MatlabEEGExtractor, PALMatConverter, \
                                  CatFRMatConverter, PSMatConverter, MathMatConverter
from loggers import log, logger
from transferer import DATA_ROOT, DB_ROOT

from tests.test_event_creation import SYS1_COMPARATOR_INPUTS, SYS2_COMPARATOR_INPUTS, \
    SYS1_STIM_COMPARISON_INPUTS, SYS2_STIM_COMPARISON_INPUTS


try:
    from ptsa.data.readers.BaseEventReader import BaseEventReader
    PTSA_LOADED = True
except:
    log('PTSA NOT LOADED')
    PTSA_LOADED = False

class PipelineTask(object):

    def __init__(self):
        self.name = str(self)
        self.pipeline = None

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def create_file(self, filename, contents, label, index_file=True):
        with open(os.path.join(self.pipeline.destination, filename), 'w') as f:
            f.write(contents)
        if index_file:
            self.pipeline.register_output(filename, label)


class ImportJsonMontageTask(PipelineTask):

    def __init__(self, subject, montage):
        super(ImportJsonMontageTask, self).__init__()
        self.name = 'Importing {subj} montage {montage}'.format(subj=subject, montage=montage)

    def run(self, files, db_folder):
        for file in ('contacts', 'pairs'):
            with open(files[file]) as f:
                filename = '{}.json'.format(file)
                output = json.load(f)
                self.create_file(filename, json.dumps(output, indent=2, sort_keys=True), file, False)

class SplitEEGTask(PipelineTask):

    SPLIT_FILENAME = '{subject}_{experiment}_{session}_{time}'

    def __init__(self, subject, montage, experiment, session, **kwargs):
        super(SplitEEGTask, self).__init__()
        self.name = 'Splitting {subj} {exp}_{sess}'.format(subj=subject, exp=experiment, sess=session)
        self.subject = subject
        self.experiment = experiment
        self.session = session
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


    def run(self, files, db_folder):
        logger.set_label(self.name)
        raw_eegs = files['raw_eeg']
        if not isinstance(raw_eegs, list):
            raw_eegs = [raw_eegs]

        raw_eeg_groups = self.group_ns2_files(raw_eegs)

        for raw_eeg in raw_eeg_groups:
            if 'substitute_raw_file_for_header' in files:
                reader = get_eeg_reader(raw_eeg,
                                       files['jacksheet'],
                                       substitute_raw_file_for_header=files['substitute_raw_file_for_header'])
            else:
                try:
                    reader = get_eeg_reader(raw_eeg,
                                            files['jacksheet'])
                except KeyError as k:
                    log('Cannot split file with extension {}'.format(k), 'WARNING')
                    continue

            split_eeg_filename = self.SPLIT_FILENAME.format(subject=self.subject,
                                                            experiment=self.experiment,
                                                            session=self.session,
                                                            time=reader.get_start_time_string())
            reader.split_data(self.pipeline.destination, split_eeg_filename)
        num_split_files = len(glob.glob(os.path.join(self.pipeline.destination, 'noreref', '*.[0-9]*')))
        if num_split_files == 0:
            raise UnProcessableException(
                'Seems like splitting did not properly occur. No split files found in {}. Check jacksheet'.format(self.pipeline.destination))


class MatlabEEGConversionTask(PipelineTask):

    def __init__(self, subject, experiment, original_session, **kwargs):
        super(MatlabEEGConversionTask, self).__init__()
        self.name = 'matlab EEG extraction for {subj}: {exp}_{sess}'.format(subj=subject,
                                                                            exp=experiment,
                                                                            sess=original_session)
        self.original_session = original_session
        self.kwargs = kwargs
    def run(self, files, db_folder):
        logger.set_label(self.name)
        extractor = MatlabEEGExtractor(self.original_session, files)
        extractor.copy_ephys(db_folder)

class EventCreationTask(PipelineTask):

    PARSERS = {
        'FR': FRSessionLogParser,
        'PAL': PALSessionLogParser,
        'catFR': CatFRSessionLogParser,
        'math': MathLogParser,
        'PS': PSLogParser
    }

    def __init__(self, protocol, subject, montage, experiment, session, is_sys2, event_label='task',
                 parser_type=None, **kwargs):
        super(EventCreationTask, self).__init__()
        self.name = '{label} Event Creation for {subj}: {exp}_{sess}'.format(label=event_label, subj=subject, exp=experiment, sess=session)
        self.parser_type = parser_type or self.PARSERS[re.sub(r'\d', '', experiment)]
        self.protocol = protocol
        self.subject = subject
        self.montage = montage
        self.experiment = experiment
        self.session = session
        self.is_sys2 = is_sys2
        self.kwargs = kwargs
        self.event_label = event_label
        self.filename = '{label}_events.json'.format(label=event_label)
        self.pipeline = None

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def run(self, files, db_folder):
        logger.set_label(self.name)
        parser = self.parser_type(self.protocol, self.subject, self.montage, self.experiment, self.session, files)
        unaligned_events = parser.parse()
        if self.is_sys2:
            aligner = System2Aligner(unaligned_events, files, db_folder)
            if self.event_label != 'math':
                aligner.add_stim_events(parser.event_template, parser.persist_fields_during_stim)
            events = aligner.align('SESS_START')
        else:
            aligner = System1Aligner(unaligned_events, files)
            events = aligner.align()
        events = parser.clean_events(events)
        self.create_file(self.filename, to_json(events),
                         '{}_events'.format(self.event_label))

class EventCombinationTask(PipelineTask):

    COMBINED_LABEL='all'

    def __init__(self, event_labels, sort_field='mstime'):
        super(EventCombinationTask, self).__init__()
        self.name = 'Event combination task for events {}'.format(event_labels)
        self.event_labels = event_labels
        self.sort_field = sort_field

    def run(self, files, db_folder):
        event_files = [os.path.join(db_folder, '{}_events.json'.format(label)) for label in self.event_labels]
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


    def __init__(self, protocol, subject, montage):
        super(MontageLinkerTask, self).__init__()
        self.name = 'Montage linker'
        self.protocol = protocol
        self.subject = subject
        self.montage = montage

    def run(self, files, db_folder):
        montage_path = self.MONTAGE_PATH.format(protocol=self.protocol,
                                                subject=self.subject,
                                                localization=self.montage.split('.')[0],
                                                montage=self.montage.split('.')[1])
        for name, file in self.FILES.items():
            fullfile = os.path.join(montage_path, file)
            if not os.path.exists(os.path.join(DB_ROOT, fullfile)):
                raise UnProcessableException("Cannot find montage for {} in {}".format(self.subject, fullfile))
            log('File {} found'.format(file))
            self.pipeline.register_info(name, fullfile)



class MatlabEventConversionTask(PipelineTask):

    CONVERTERS = {
        'FR': FRMatConverter,
        'PAL': PALMatConverter,
        'catFR': CatFRMatConverter,
        'PS': PSMatConverter
    }

    def __init__(self, protocol, subject, montage, experiment, session,
                 event_label='task', converter_type=None, original_session=None, **kwargs):
        super(MatlabEventConversionTask, self).__init__()
        self.name = '{label} Event Creation for {subj}: {exp}_{sess}'.format(label=event_label, subj=subject, exp=experiment, sess=session)
        self.converter_type = converter_type or self.CONVERTERS[re.sub(r'\d', '', experiment)]
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

    def run(self, files, db_folder):
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
                 converter_type=None, parser_type=None, original_session=None, **kwargs):
        super(ImportEventsTask, self).__init__()
        self.name = '{label} Event Import for {subj}: {exp}_{sess}'.format(label=event_label, subj=subject, exp=experiment, sess=session)
        self.converter_type = converter_type or self.CONVERTERS[re.sub(r'\d', '', experiment)]
        self.parser_type = parser_type or self.PARSERS[re.sub(r'\d', '', experiment)]
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

    def run(self, files, db_folder):
        try:
            EventCreationTask.run(self, files, db_folder)
        except Exception:
            log("Exception occurred creating events! Defaulting to event conversion!")
            MatlabEventConversionTask.run(self, files, db_folder)


class IndexAggregatorTask(PipelineTask):

    INDEX_LOCATION = os.path.join(DB_ROOT, 'protocols', 'index.json')
    PROCESSED_DIRNAME = 'current_processed'
    INDEX_FILENAME = 'index.json'

    def __init__(self):
        super(IndexAggregatorTask, self).__init__()

    @classmethod
    def build_index(cls):
        index_files = cls.find_index_files()
        d = {}
        for index_file in index_files:
            cls.build_single_file_index(index_file, d)
        return d


    @classmethod
    def find_index_files(cls):
        result = []
        for root, dirs, files in os.walk(DB_ROOT):
            if cls.PROCESSED_DIRNAME in dirs and \
                    cls.INDEX_FILENAME in os.listdir(os.path.join(root, cls.PROCESSED_DIRNAME)):
                result.append(os.path.join(root, cls.PROCESSED_DIRNAME, cls.INDEX_FILENAME))
        return result

    @classmethod
    def build_single_file_index(cls, index_path, d):
        index = json.load(open(index_path))
        info_list = cls.list_from_index_path(index_path)

        sub_d = d
        for entry in info_list:
            if entry[0] not in sub_d:
                sub_d[entry[0]] = {}
            if entry[1] not in sub_d[entry[0]]:
                sub_d[entry[0]][entry[1]] = {}
            sub_d = sub_d[entry[0]][entry[1]]

        current_dir = os.path.dirname(index_path)
        rel_dirname = os.path.relpath(current_dir, DB_ROOT)
        if 'files' in index:
            for name, file in index['files'].items():
                sub_d[name] = os.path.join(rel_dirname, file)
        if 'info' in index:
            sub_d.update(index['info'])

    @classmethod
    def list_from_index_path(cls, index_path):
        """
        IndexAggregatorTask.dict_from_path('/protocols/r1') == {'protocols': {'r1': {}}}
        :param path:
        :return:
        """
        processed_dir = os.path.dirname(index_path)
        type_dir = os.path.dirname(processed_dir)

        value_dir = os.path.dirname(type_dir)
        path_list = []
        while os.path.realpath(value_dir) != os.path.realpath(DB_ROOT):
            key_dir = os.path.dirname(value_dir)
            path_list.append((os.path.basename(key_dir), os.path.basename(value_dir)))
            value_dir = os.path.dirname(key_dir)
            if os.path.basename(key_dir) == '':
                raise Exception('Could not locate {} in {}'.format(DB_ROOT, index_path))
        return path_list[::-1]

    def run(self, *_):
        index = self.build_index()
        json.dump(index, open(self.INDEX_LOCATION,'w'), sort_keys=True, indent=2)

def test_list_from_index_path():
    import pprint
    pprint.pprint(IndexAggregatorTask.build_index())

"""
class HierarchicalAggregatorTask(PipelineTask):

    DIRECTORY_ORDER = None
    GLOBAL_AGGREGATIONS = {}

    def __init__(self, additional_values=None, additional_keys=None, **kwargs):
        super(HierarchicalAggregatorTask, self).__init__()
        self.additional_values = additional_values or {}
        self.additional_keys = additional_keys or {}
        self.kwargs = kwargs

    @classmethod
    def aggregate_hierarchy(cls, directory_order, global_aggregations, files, additional_entries, **kwargs):
        # Once for the globals
        for aggregation_label, aggregation_key in global_aggregations.items():
            directory_path = DB_ROOT
            for i, (directory, key) in enumerate(directory_order):
                log('Aggregating {}: {} in {}'.format(aggregation_label, aggregation_key.format(**kwargs), directory))
                directory_path = os.path.join(directory_path, directory)
                cls.build_aggregate(directory_path, directory_order[i:],
                                    aggregation_label, aggregation_key, files, additional_entries, **kwargs)

                directory_path = os.path.join(directory_path, key.format(**kwargs))
                if directory == aggregation_label:
                    break
        # Once for indexing of current directory
        directory_path = DB_ROOT
        for i, (directory, key) in enumerate(directory_order):
            directory_path = os.path.join(directory_path, directory)
            if not directory in global_aggregations:
                log('Aggregating {}: {} in {}'.format(directory, key.format(**kwargs), directory))
                cls.build_aggregate(directory_path, directory_order[i:],
                                    directory, key, files, additional_entries, **kwargs)
            directory_path = os.path.join(directory_path, key.format(**kwargs))

    @staticmethod
    def build_aggregate(directory_path, directory_order, aggregation_label, aggregation_key, files,
                        additional_values, **kwargs):
        agg_file = os.path.join(directory_path, '{}.json'.format(aggregation_label))
        if os.path.exists(agg_file):
            aggregate = json.load(open(agg_file))
        else:
            aggregate = {}

        key_value = aggregation_key.format(**kwargs)

        if key_value not in aggregate:
            aggregate[key_value] = {}

        aggregate_level = aggregate[key_value]

        for entry, entry_key in directory_order:
            if entry == aggregation_label:
                continue
            entry_key_value = entry_key.format(**kwargs)
            if entry not in aggregate_level:
                aggregate_level[entry] = {}

            aggregate_entry = aggregate_level[entry]
            if entry_key_value not in aggregate_entry:
                aggregate_entry[entry_key_value] = {}

            aggregate_level = aggregate_entry[entry_key_value]

        aggregate_level.clear()
        for label, file in files.items():
            aggregate_level[label] = os.path.relpath(file, directory_path)
        aggregate_level.update(additional_values)

        json.dump(aggregate, open(agg_file, 'w'), indent=2, sort_keys=True)

    def run(self, files, db_folder):
        if not self.DIRECTORY_ORDER:
            raise NotImplementedError("Must use subclass specifying directory order")
        self.aggregate_hierarchy(self.DIRECTORY_ORDER, self.GLOBAL_AGGREGATIONS, self.pipeline.output_files,
                                 self.additional_values, **self.kwargs)


class FlatAggregatorTask(PipelineTask):


    DIRECTORY_ORDER = None
    AGGREGATE_NAME = None
    AGGREGATE_KEY = None

    def __init__(self, info=None, **kwargs):
        super(FlatAggregatorTask, self).__init__()
        self.info = info or {}
        self.kwargs = kwargs

    @classmethod
    def aggregate_info(cls, directory_order, aggregate_name, aggregate_key, additional_keys, files, **kwargs):
        directory_path = DB_ROOT
        for i, (directory, key) in enumerate(directory_order):
            log('Aggregating {}: {} in {}'.format(aggregate_name, aggregate_key.format(**kwargs),
                                                  directory))
            directory_path = os.path.join(directory_path, directory)
            cls.build_aggregate(directory_path, aggregate_name, aggregate_key, additional_keys, files, **kwargs)

            used_kwargs = []
            for kwarg, kwkey in kwargs.items():
                try:
                    directory_path = os.path.join(directory_path, key.format(**{kwarg:kwkey}))
                    used_kwargs.append(kwarg)
                except KeyError:
                    pass
            for kwarg in used_kwargs:
                del kwargs[kwarg]


    @staticmethod
    def build_aggregate(directory_path, aggregate_name, aggregate_key, info, files, **kwargs):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        agg_file = os.path.join(directory_path, '{}.json'.format(aggregate_name))
        if os.path.exists(agg_file):
            aggregate = json.load(open(agg_file))
        else:
            aggregate = {}

        key_value = aggregate_key.format(**kwargs)

        aggregate[key_value] = info.copy()

        for label, file in files.items():
            aggregate[key_value][label] = os.path.relpath(file, directory_path)

        aggregate[key_value].update(kwargs)

        json.dump(aggregate, open(agg_file, 'w'), indent=2, sort_keys=True)

    def run(self, files, db_folder):
        if not self.DIRECTORY_ORDER or not self.AGGREGATE_KEY or not self.AGGREGATE_NAME:
            raise NotImplementedError("Must use subclass specifying directory order")
        self.aggregate_info(self.DIRECTORY_ORDER, self.AGGREGATE_NAME, self.AGGREGATE_KEY, self.info, files,
                            **self.kwargs)

class SessionAggregatorTask(HierarchicalAggregatorTask):


    DIRECTORY_ORDER = (('protocols', '{protocol}'),
                       ('subjects', '{subject}'),
                       ('experiments', '{experiment}'),
                       ('sessions', '{session}'))

    GLOBAL_AGGREGATIONS = {'protocols': '{protocol}',
                            'subjects': '{subject}',
                            'experiments': '{experiment}'}

    def __init__(self, montage, localization, **kwargs):
        super(SessionAggregatorTask, self).__init__({'montage': montage, 'localization': localization}, **kwargs)


class CodeAgggregatorTask(FlatAggregatorTask):

    DIRECTORY_ORDER = (('protocols', '{protocol}'),
                       ('subjects', '{subject}'),
                       ('localizations', '{localization}'),
                       ('montages', '{montage}'))

    AGGREGATE_NAME = 'codes'
    AGGREGATE_KEY = '{code}'
"""

class CompareEventsTask(PipelineTask):

    def __init__(self, subject, montage, experiment, session, protocol='r1', code=None, original_session=None,
                 match_field=None):
        super(CompareEventsTask, self).__init__()
        self.name = 'Comparator {}: {}_{}'.format(subject, experiment, session)
        self.subject = subject
        self.code = code if code else subject
        self.original_session = original_session if not original_session is None else session
        self.montage = montage
        self.experiment = experiment
        self.session = session
        self.protocol = protocol
        self.match_field = match_field if match_field else 'mstime'

    def run(self, files, db_folder):
        logger.set_label(self.name)
        mat_events_reader = \
            BaseEventReader(
                filename=os.path.join(DATA_ROOT, '..', 'events', 'RAM_{}'.format(self.experiment[0].upper() + self.experiment[1:]),
                                      '{}_events.mat'.format(self.code)),
                common_root=DATA_ROOT,
                eliminate_events_with_no_eeg=False,
            )
        log('Loading matlab events')
        mat_events = mat_events_reader.read()
        self.sess_mat_events = mat_events[mat_events.session == self.original_session]
        if not PTSA_LOADED:
            raise UnProcessableException('Cannot compare events without PTSA')
        new_events = from_json(os.path.join(db_folder, 'task_events.json'))
        major_version = '.'.join(new_events[-1].exp_version.split('.')[:1])

        if float(major_version.split('_')[-1]) >= 2:
            comparator_inputs = SYS2_COMPARATOR_INPUTS[self.experiment]
        else:
            comparator_inputs = SYS1_COMPARATOR_INPUTS[self.experiment]
        comparator = EventComparator(new_events, self.sess_mat_events, match_field=self.match_field, **comparator_inputs)
        log('Comparing events...')

        found_bad, error_message = comparator.compare()

        if found_bad is True:
            assert False, error_message
        else:
            log('Comparison Success!')

        log('Comparing stim events...')


        if float(major_version.split('_')[-1]) >= 2:
            comparator_inputs = SYS2_STIM_COMPARISON_INPUTS[self.experiment]
        else:
            comparator_inputs = SYS1_STIM_COMPARISON_INPUTS[self.experiment]

        stim_comparator = StimComparator(new_events, self.sess_mat_events, **comparator_inputs)
        errors = stim_comparator.compare()

        if errors:
            assert False, errors
        else:
            log('Stim comparison success!')







class UnProcessableException(Exception):
    pass

def change_current(source_folder, *args):
    destination_directory = os.path.join(DB_ROOT, *args)
    destination_source = os.path.join(destination_directory, source_folder)
    destination_processed = os.path.join(destination_directory, '{}_processed'.format(source_folder))
    if not os.path.exists(destination_source):
        raise UnProcessableException('Source folder {} does not exist'.format(destination_source))
    if not os.path.exists(destination_processed):
        raise UnProcessableException('Processed folder {} does not exist'.format(destination_processed))

    current_source = os.path.join(destination_directory, 'current_source')
    current_processed = os.path.join(destination_directory, 'current_processed')

    previous_current_source = os.path.basename(os.path.realpath(current_source))

    log('Unlinking current source: {}'.format(os.path.realpath(current_source)))
    os.unlink(current_source)
    try:
        log('Linking current source to {}'.format(source_folder))
        os.symlink(source_folder, current_source)
    except Exception as e:
        log('ERROR {}. Rolling back'.format(e.message))
        os.symlink(previous_current_source, current_source)
        raise

    previous_current_processed = os.path.basename(os.path.realpath(current_processed))
    try:
        log('Unlinking current processed: {}'.format(os.path.realpath(current_processed)))
        os.unlink(current_processed)
    except Exception as e:
        log('ERROR {}. Rolling back'.format(e.message))
        os.symlink(previous_current_source, current_source)

    try:
        processed_folder = '{}_processed'.format(source_folder)
        log('Linking current processed to {}'.format(processed_folder))
        os.symlink(processed_folder, current_processed)
    except Exception as e:
        log('ERROR {}. Rolling back'.format(e.message))
        os.symlink(previous_current_source, current_source)
        os.symlink(previous_current_processed, current_processed)

def xtest_change_current():
    from convenience import build_split_pipeline
    import time
    subject, montage, experiment, session, protocol = 'R1001P', '0.0', 'FR1', 0, 'r1',
    pipeline_1 = build_split_pipeline(subject, montage, experiment, session)
    pipeline_1.run()
    previous_label = pipeline_1.source_label
    time.sleep(5)
    pipeline_2 = build_split_pipeline(subject, montage, experiment, session)
    pipeline_2.run(force=True)
    change_current(previous_label,
                            'protocols', protocol,
                            'subjects', subject,
                            'experiments', experiment,
                            'sessions', str(session),
                            'ephys')

