import os
import re
import json
import traceback
import shutil

import fileutil
from .log import logger
from .configuration import paths
from .exc import ProcessingError

try:
    from ptsa.data.readers import BaseEventReader
except:
    logger.warn('PTSA NOT LOADED')

class PipelineTask(object):
    """Base class for running tasks in a pipeline.

    Parameters
    ----------
    critical : bool
       TODO: what does this mean?

    """
    def __init__(self, critical=True):
        self.critical = critical
        self.name = str(self)
        self.pipeline = None
        self.destination = None
        self.error = None

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def create_file(self, filename, contents, label, index_file=True):
        with fileutil.open_with_perms(os.path.join(self.destination, filename), 'w') as f:
            f.write(contents)
        if index_file:
            self.pipeline.register_output(filename, label)

    def run(self, files, db_folder):
        self.destination = db_folder
        try:
            self._run(files, db_folder)
        except Exception as e:
            if self.critical:
                raise
            else:
                traceback.print_exc()
                self.error = e.message

    def _run(self, files, db_folder):
        raise NotImplementedError()


class ImportJsonMontageTask(PipelineTask):
    """

    """
    def __init__(self, subject, montage, critical=True):
        super(ImportJsonMontageTask, self).__init__(critical)
        self.name = 'Importing {subj} montage {montage}'.format(subj=subject, montage=montage)

    def _run(self, files, db_folder):
        for file in ('contacts', 'pairs'):
            with open(files[file]) as f:
                filename = '{}.json'.format(file)
                output = json.load(f)
                self.create_file(filename, json.dumps(output, indent=2, sort_keys=True), file, False)


class CleanDbTask(PipelineTask):
    """

    """
    SOURCE_REGEX = '^\d{8}\.\d{6}$'
    PROCESSED_REGEX = '^\d{8}\.\d{6}_processed$'

    @classmethod
    def run(cls, files=None, db_folder=None):

        for root, dirs, files in os.walk(os.path.join(paths.db_root, 'protocols'), False):
            if len(dirs) == 0 and len(files) == 1 and 'log.txt' in files:
                os.remove(os.path.join(root, 'log.txt'))
                logger.debug('Removing {}'.format(root))
                os.rmdir(root)
            elif len(os.listdir(root)) == 0:
                logger.debug('Removing {}'.format(root))
                os.rmdir(root)
        for root, dirs, files in os.walk(os.path.join(paths.db_root, 'protocols'), False):
            for dir in dirs:
                if not os.path.exists(os.path.join(root, dir)):
                    # Directory may have already been deleted
                    logger.warn("Path {} {} already deleted?".format(root, dir))
                    continue
                if re.match(cls.SOURCE_REGEX, dir):
                    processed_dir = '{}_processed'.format(dir)
                    if not processed_dir in dirs:
                        logger.warn("Removing {} in {}".format(dir, root))
                        shutil.rmtree(os.path.join(root, dir))
                if re.match(cls.PROCESSED_REGEX, dir):
                    source_dir = dir.replace('_processed', '')
                    if not source_dir in dirs:
                        logger.warn("Removing {} in {}".format(dir, root))
                        shutil.rmtree(os.path.join(root, dir))


class CleanLeafTask(PipelineTask):
    """

    """
    SOURCE_REGEX = '^\d{8}\.\d{6}$'
    PROCESSED_REGEX = '^\d{8}\.\d{6}_processed$'

    def __init__(self, critical=False):
        super(CleanLeafTask, self).__init__(critical)
        self.name = 'Clean leaf of database'

    @classmethod
    def _run(cls, files, db_folder):
        abs_path = os.path.abspath(db_folder)

        if not os.path.exists(abs_path):
            abs_path = os.path.abspath(os.path.join(abs_path, '..'))

        if not os.path.exists(abs_path):
            return

        while not os.path.samefile(abs_path, paths.db_root):

            contents = os.listdir(abs_path)

            contains_stuff = len(contents) > 0

            if contains_stuff and len(contents) == 1:
                if contents[0] == 'log.txt':
                    logger.debug("Removing log file in {}".format(abs_path))
                    os.remove(os.path.join(abs_path, contents[0]))
                    contains_stuff = False

            if not contains_stuff:
                logger.debug("Removing empty directory {}".format(abs_path))
                new_abs_path = os.path.abspath(os.path.join(abs_path, '..'))
                os.rmdir(abs_path)
                abs_path = new_abs_path
            else:
                logger.debug("Stopped cleaning due to contents {}".format(contents))
                break


class IndexAggregatorTask(PipelineTask):
    """

    """
    PROTOCOLS_DIR = os.path.join(paths.db_root, 'protocols')
    PROTOCOLS = ('r1', 'ltp')
    PROCESSED_DIRNAME = 'current_processed'
    INDEX_FILENAME = 'index.json'

    @classmethod
    def build_index(cls, protocol):
        index_files = cls.find_index_files(os.path.join(cls.PROTOCOLS_DIR, protocol))
        d = {}
        for index_file in index_files:
            cls.build_single_file_index(index_file, d)
        return d

    @classmethod
    def find_index_files(cls, root_dir):
        result = []
        for root, dirs, files in os.walk(root_dir):
            if cls.PROCESSED_DIRNAME in dirs and \
                    cls.INDEX_FILENAME in os.listdir(os.path.join(root, cls.PROCESSED_DIRNAME)):
                result.append(os.path.join(root, cls.PROCESSED_DIRNAME, cls.INDEX_FILENAME))
        return result

    @classmethod
    def build_single_file_index(cls, index_path, d):
        """
        Adds to the current index "d" with the information at "index_path"
        :param index_path: Path to the index file (in subject folder) to be aggregated
        :param d: dictionary to be appended to
        :return:
        """
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
        rel_dirname = os.path.relpath(current_dir, paths.db_root)
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
        while os.path.realpath(value_dir) != os.path.realpath(paths.db_root):
            key_dir = os.path.dirname(value_dir)
            path_list.append((os.path.basename(key_dir), os.path.basename(value_dir)))
            value_dir = os.path.dirname(key_dir)
            if os.path.basename(key_dir) == '':
                raise Exception('Could not locate {} in {}'.format(paths.db_root, index_path))
        return path_list[::-1]

    def run(self, protocols=None, *_):

        # Protocols can be input as a string for a single protocol, or an iterable of protocols. Otherwise, use the
        # default protocols defined in self.PROTOCOLS.
        if isinstance(protocols, str):
            protocols = [protocols]
        elif not hasattr(protocols, '__iter__'):
            protocols = self.PROTOCOLS

        protocols = self.PROTOCOLS if protocols is None else protocols
        for protocol in protocols:
            index = self.build_index(protocol)
            try:
                with fileutil.open_with_perms(os.path.join(self.PROTOCOLS_DIR, '{}.json'.format(protocol)), 'w') as f:
                    json.dump(index, f, sort_keys=True, indent=2)
            except IOError:
                logger.warn('Unable to open file ' + os.path.join(self.PROTOCOLS_DIR, '{}.json'.format(protocol)) + ' with write permissions.')

    def run_single_subject(self, subject, protocol):
        try:
            index_file = open(os.path.join(self.PROTOCOLS_DIR, '{}.json'.format(protocol)), 'r')
            index = json.load(index_file)
        except:
            index = {}
        subject_dir = os.path.join(self.PROTOCOLS_DIR, protocol, 'subjects', subject)
        subj_index_files = self.find_index_files(subject_dir)
        for subj_index_file in subj_index_files:
            self.build_single_file_index(subj_index_file, index)

        with fileutil.open_with_perms(os.path.join(self.PROTOCOLS_DIR, '{}.json'.format(protocol)), 'w') as f:
            json.dump(index, f, sort_keys=True, indent=2)


def change_current(source_folder, *args):
    """

    :param source_folder:
    :param args:
    :return:
    """
    destination_directory = os.path.join(paths.db_root, *args)
    destination_source = os.path.join(destination_directory, source_folder)
    destination_processed = os.path.join(destination_directory, '{}_processed'.format(source_folder))
    if not os.path.exists(destination_source):
        raise ProcessingError('Source folder {} does not exist'.format(destination_source))
    if not os.path.exists(destination_processed):
        raise ProcessingError('Processed folder {} does not exist'.format(destination_processed))

    current_source = os.path.join(destination_directory, 'current_source')
    current_processed = os.path.join(destination_directory, 'current_processed')

    previous_current_source = os.path.basename(os.path.realpath(current_source))

    logger.info('Unlinking current source: {}'.format(os.path.realpath(current_source)))
    os.unlink(current_source)
    try:
        logger.info('Linking current source to {}'.format(source_folder))
        os.symlink(source_folder, current_source)
    except Exception as e:
        logger.error('ERROR {}. Rolling back'.format(e.message))
        os.symlink(previous_current_source, current_source)
        raise

    previous_current_processed = os.path.basename(os.path.realpath(current_processed))
    try:
        logger.info('Unlinking current processed: {}'.format(os.path.realpath(current_processed)))
        os.unlink(current_processed)
    except Exception as e:
        logger.error('ERROR {}. Rolling back'.format(e.message))
        os.symlink(previous_current_source, current_source)

    try:
        processed_folder = '{}_processed'.format(source_folder)
        logger.info('Linking current processed to {}'.format(processed_folder))
        os.symlink(processed_folder, current_processed)
    except Exception as e:
        logger.error('ERROR {}. Rolling back'.format(e.message))
        os.symlink(previous_current_source, current_source)
        os.symlink(previous_current_processed, current_processed)
