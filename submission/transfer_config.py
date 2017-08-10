
import json
import copy
import glob
import os
import yaml
import hashlib
import fileutil
import shutil
from collections import defaultdict

from .configuration import paths
from .log import logger
from .exc import ConfigurationError


def build_group_index(input, groups):
    transfer_files = dict()

    for file in input:

        do_process = True
        for group in file['groups']:
            if group.startswith('!'):
                if group.strip('!') in groups:
                    do_process = False
                    break
            else:
                if group not in groups:
                    do_process = False
                    break

        if not do_process:
            continue

        transfer_file = TransferFile(**file)
        transfer_file.expand_files(groups)
        transfer_files[transfer_file.name] = transfer_file

    return transfer_files


class TransferConfig(object):

    JSON_FILES = {}

    def __init__(self, filename, groups, **kwargs):
        self.filename = filename
        self.groups = groups
        self.kwargs = kwargs

        self._raw_config = yaml.load(open(filename))
        self._files = build_group_index(self._raw_config['files'], groups)

        for file_ in self._files.values():
            file_.format(**kwargs)

    @property
    def valid_files(self):
        return [file for file in self._files.values() if file.valid]
        # return {name: file_ for name, file_ in self._files.items() if file_.valid}

    def located_files(self):
        located_files = []
        for file in self.valid_files:
            if not file.location_attempted:
                file.locate()
            if file.located:
               located_files.append(file)
        return located_files

    def get_file(self, name):
        return self._files.get(name)

    def locate_origin_files(self):
        logger.debug("Locating files {}".format(self._files))
        for file in self.valid_files:
            try:
                file.locate()
                if file.located:
                    logger.debug("File {} located".format(file.name))
                else:
                    logger.debug("Could not locate {}".format(file.name))
            except ConfigurationError:
                logger.debug("Could not locate {}".format(file.name))

    def missing_files(self):
        missing_files = []
        for file in self.valid_files:
            try:
                file.locate()
            except ConfigurationError:
                missing_files.append(file)
        return missing_files

    def missing_required_files(self):
        missing_files = []
        for file in self._files.values():
            if file.required:
                if not file.located:
                    try:
                        file.locate()
                    except ConfigurationError:
                        pass
                if not file.located:
                    logger.debug("Required file {file.name} is missing".format(file=file))
                    missing_files.append(file)
        return missing_files

class TransferFile(object):

    REQUIRED_PROPERTIES = ('name', 'type', 'groups', 'multiple', 'required', 'checksum_contents',
                           'origin_directory', 'origin_file', 'destination')

    OPTIONAL_PROPERTIES = { 'files': [] }


    def __init__(self, **kwargs):

        for name in self.REQUIRED_PROPERTIES:
            if name not in kwargs:
                raise ConfigurationError("Property {} not provided for transfer configuration entry {}".format(name, kwargs))
            setattr(self, '_'+name, kwargs[name])

        for name in self.OPTIONAL_PROPERTIES:
            setattr(self, '_'+name, kwargs.get(name, self.OPTIONAL_PROPERTIES[name]))

        for name in kwargs:
            if name not in self.REQUIRED_PROPERTIES and name not in self.OPTIONAL_PROPERTIES:
                raise ConfigurationError("Unknown property {} provided in configuration entry {}".format(name, kwargs))

        self._expanded_files = None
        self.formatted_origin_dir = ''
        self.formatted_origin_filenames = []
        self._origin_paths = []
        self._located = False
        self.destination_directories = []
        self.destination_filenames = []

        self._valid = False

        self._roots_located = []

        self._checksum = hashlib.md5()
        self._checksum_calculated = False

        self._transferred_files = []

    @property
    def located(self):
        return self._located

    @property
    def location_attempted(self):
        return len(self._roots_located) > 0

    @property
    def valid(self):
        return self._valid

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def required(self):
        return self._required

    @property
    def origin_directory(self):
        return self._origin_directory

    @property
    def files(self):
        return self._expanded_files

    @property
    def origin_paths(self):
        if not self._located:
            raise Exception("Attempt to access origin paths of {} before locating".format(self.name))
        return self._origin_paths

    def expand_files(self, groups):
        expanded_files = build_group_index(self._files, groups)
        self._expanded_files = expanded_files

    @property
    def multiple(self):
        return self._multiple

    def origin_containing_directory(self, root=''):
        return os.path.join(root, self.formatted_origin_dir)

    def destination_containing_directory(self, root):
        if self.multiple:
            destination_directory_name = self._destination
        else:
            destination_directory_name = os.path.dirname(self._destination)

        return os.path.join(root, destination_directory_name)

    def transfer(self, root):

        if not self.located:
            self.locate(root)

        containing_dir = self.destination_containing_directory(root)

        if not os.path.exists(containing_dir):
            fileutil.makedirs(containing_dir)

        if self.type == 'directory':
            for file in self.files.values():
                file.transfer(containing_dir)
            return

        for origin, destination_dir in zip(self.origin_paths, self.destination_directories):

            destination_filename = os.path.split(origin)[-1] if self.multiple else self._destination
            logger.debug('destination_filename:%s'%destination_filename)

            destination_path= os.path.join(root, destination_dir, destination_filename)

            if self.type == 'file':
                if not os.path.exists(os.path.dirname(destination_path)):
                    os.makedirs(os.path.dirname(destination_path))
                logger.debug("Copying file {} to {}".format(origin, destination_path))
                shutil.copyfile(origin, destination_path)
                logger.debug("File {} copied successfully".format(origin))
                self._transferred_files.append(destination_path)

            elif self.type == 'link':
                link = os.path.relpath(os.path.realpath(origin), os.path.realpath(os.path.dirname(destination_path)))
                logger.debug("Linking {} to {}".format(link, destination_path))
                os.symlink(link, destination_path)
                logger.debug("Linked successfully")
                self._transferred_files.append(destination_path)
            else:
                raise ConfigurationError("File type {} not known. Must be 'file', 'directory', or 'link'".format(self.type))


    @property
    def checksum(self):
        if not self._checksum_calculated:
            self.calculate_checksum()
        return self._checksum

    def contents_to_check(self):
        contents = []
        if not self._checksum_contents:
            for filename in self.origin_paths:
                contents.append(os.path.basename(filename))
        else:
            for filename in self.origin_paths:
                contents.append(open(filename).read())

        for file in self.files.values():
            contents.extend(file.contents_to_check())

        return contents

    def calculate_checksum(self):
        for element in self.contents_to_check():
            self._checksum.update(element)
        self._checksum_calculated = True

    def format(self, **kwargs):
        new_kwargs  = dict(**kwargs)

        if 'subject' in kwargs:
            if 'protocols' in self.origin_directory:
                new_kwargs['subject'] = kwargs['subject'].split('_')[0]
        formatted = self.attempt_format(self.origin_directory, self.required, **new_kwargs)

        if formatted is None:
            self._valid = False
            logger.debug("Invalid formatting of {}".format(self.origin_directory))
            return
        self.formatted_origin_dir = formatted

        if not isinstance(self._origin_file, list):
            origin_files = [self._origin_file]
        else:
            origin_files = self._origin_file

        formatted = []
        for origin_file in origin_files:
            f = self.attempt_format(origin_file, self.required, **new_kwargs)
            if f is not None:
                formatted.append(f)

        if len(formatted) == 0:
            logger.debug("Invalid formatting of {}".format(self.origin_directory))
            self._valid = False
            return

        self.formatted_origin_filenames = formatted

        for file in self.files.values():
            file.format(**kwargs)

        self._valid = all(file.valid for file in self.files.values())

    def locate(self, root=''):

        if root in self._roots_located:
            return

        logger.debug("Locating {}".format(self.name))

        if self.name == 'vertices':
            pass

        containing_directory = self.origin_containing_directory(root)

        if self.type == 'link':
            containing_directory = os.path.relpath(os.path.realpath(containing_directory))

        new_origin_paths = []
        new_destination_directories = []

        for origin_filename in self.formatted_origin_filenames:
            origin_path = os.path.join(containing_directory, origin_filename)
            logger.debug('Looking for {}'.format(origin_path))
            new_files = glob.glob(origin_path)

            if len(new_files) == 0:
                logger.debug("Could not find files at {}".format(os.path.abspath(origin_path)))

            new_origin_paths.extend(new_files)

            for new_file in new_files:
                new_destination_directory = os.path.relpath(os.path.dirname(new_file), containing_directory, )
                if self._multiple:
                    new_destination_directories.append(os.path.join(os.path.basename(root),
                                                                    self._destination,
                                                                    new_destination_directory))
                else:
                    new_destination_directories.append(os.path.join(os.path.basename(root),
                                                                    new_destination_directory))


        for path in new_origin_paths:
            for file in self.files.values():
                file.locate(path)

        if self.required and len(new_origin_paths) == 0:
            raise ConfigurationError("File {} is required, but cannot be found. "
                                          "(Location: {}/{})".format(self.name, containing_directory,
                                                                     self.formatted_origin_filenames))

        if len(new_origin_paths) > 1 and not self.multiple:
            raise ConfigurationError("Multiple files matching {} found in {}/{} "
                                          "but multiple==False".format(self.name, self.formatted_origin_filenames,
                                                                       containing_directory))

        self._origin_paths.extend(new_origin_paths)
        self.destination_directories.extend(new_destination_directories)
        self._roots_located.append(root)
        self._located = len(new_origin_paths) > 0


    @staticmethod
    def attempt_format(to_format, required, **kwargs):
        formatted = None
        try:
            formatted = to_format.format(**kwargs)
        except KeyError as e:
            if required:
                logger.debug("Keyword error {} for {}".format(e, to_format))
                raise ConfigurationError("Could not find keyword {} for {}".format(e, to_format))
            else:
                logger.debug("Could not format {}".format(to_format))
        return formatted

    def transferred_index(self):
        index = {
            self.name: dict(
                origin_files=[os.path.relpath(origin_path, paths.rhino_root) for origin_path in self.origin_paths],
                md5=self.checksum.hexdigest(),
            )
        }

        for file in self.files.values():
            index.update(file.transferred_index())

        return index

    def transferred_filenames(self, force_multiple=False):
        files = self._transferred_files
        multiple = self.multiple or force_multiple
        index = {} if not multiple else defaultdict(list)
        if len(files) > 0:
            index[self.name] = files[0] if not multiple else files

        for file in self.files.values():
            transferred_files = file.transferred_filenames(multiple)
            for name, filename in transferred_files.items():
                if multiple:
                    index[name].extend(filename)
                else:
                    index[name] = filename

        return index

    def matches_transferred_index(self, entry):
        if self.checksum.hexdigest() == entry['md5']:
            return True
        # TODO: Check contents of self.files? Not really necessary
        return False
