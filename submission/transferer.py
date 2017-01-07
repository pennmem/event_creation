import json
import os
import glob
import shutil
import datetime
import hashlib


from config import DATA_ROOT, LOC_DB_ROOT, DB_ROOT, EVENTS_ROOT
import files
from loggers import logger
from collections import defaultdict

TRANSFER_INPUTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),'transfer_inputs')

TRANSFER_INPUTS = {
    'behavioral': os.path.join(TRANSFER_INPUTS_DIR, 'behavioral_inputs.json'),
    'ephys': os.path.join(TRANSFER_INPUTS_DIR, 'ephys_inputs.json'),
    'montage': os.path.join(TRANSFER_INPUTS_DIR, 'montage_inputs.json')
}

class UnTransferrableException(Exception):
    pass

class Transferer(object):

    CURRENT_NAME = 'current_source'
    INDEX_NAME='index.json'
    STRFTIME = '%Y%m%d.%H%M%S'
    TRANSFER_TYPE_NAME='TRANSFER_TYPE'

    JSON_FILES = {}

    def __init__(self, json_file, groups, destination, **kwargs):
        self.groups = groups
        self.destination_root = os.path.abspath(destination)
        self.destination_current = os.path.join(self.destination_root, self.CURRENT_NAME)
        self.label = datetime.datetime.now().strftime(self.STRFTIME)
        logger.debug('Transferer {} created'.format(self.label))
        self.kwargs = kwargs
        self.transferred_files = {}
        self.transfer_dict = self.load_groups(self.load_json_input(json_file), groups)
        self.old_symlink = None
        self.transfer_aborted = False
        self.previous_label = self.get_current_target()
        self._should_transfer = True
        self.transfer_type = 'IMPORT'

    @classmethod
    def load_json_input(cls, json_file):
        if json_file not in cls.JSON_FILES:
            cls.JSON_FILES[json_file] = json.load(open(json_file))
        return cls.JSON_FILES[json_file]

    def get_label(self):
        if self.transfer_aborted:
            return self.previous_label
        else:
            return self.label

    def set_transfer_type(self, transfer_type):
        self.transfer_type = transfer_type

    def build_transferred_index(self):
        md5_dict = {}
        destination_labeled = os.path.join(self.destination_root, self.label)
        for name, files in self.transferred_files.items():
            info = self.transfer_dict[name]
            md5_dict.update(self.build_transfer_index_entry(name, files, info, destination_labeled))

        return md5_dict

    @classmethod
    def build_transfer_index_entry(cls, name, files, info, destination_labeled):
        index_dict = {}
        if info['type'] == 'directory':
            sub_dir = {}
            for sub_dir in files:
                for sub_name, sub_files in sub_dir.items():
                    sub_info = info['contents'][sub_name]
                    new_index = cls.build_transfer_index_entry(sub_name, sub_files, sub_info, destination_labeled)
                    if sub_name not in index_dict:
                        index_dict.update(cls.build_transfer_index_entry(sub_name, sub_files, sub_info, destination_labeled))
                    else:
                        index_dict[sub_name]['files'].extend(new_index[sub_name]['files'])
            for sub_name, sub_files in sub_dir.items():
                md5_sum = hashlib.md5()
                sub_info = info['contents'][sub_name]
                for each_file in index_dict[sub_name]['files']:
                    if 'checksum_filename_only' in sub_info and sub_info['checksum_filename_only']:
                        md5_sum.update(os.path.basename(each_file))
                    else:
                        md5_sum.update(open(os.path.join(destination_labeled, each_file)).read())
                index_dict[sub_name]['md5'] = md5_sum.hexdigest()

        elif len(files) > 0:
            if not isinstance(files, list):
                files = [files]
            index_dict[name] = {'files': []}
            md5_sum = hashlib.md5()
            for each_file in files:
                index_dict[name]['files'].append(os.path.relpath(each_file, destination_labeled))
                if 'checksum_filename_only' in info and info['checksum_filename_only']:
                    md5_sum.update(os.path.basename(each_file))
                else:
                    md5_sum.update(open(each_file).read())
            info['md5'] = md5_sum.hexdigest()

        return index_dict

    def get_current_target(self):
        current = os.path.join(self.destination_current)
        if not os.path.exists(current):
            return None
        else:
            return os.path.basename(os.path.realpath(current))

    def load_previous_index(self):
        old_index_filename = os.path.join(self.destination_current, self.INDEX_NAME)
        if not os.path.exists(old_index_filename):
            return {}
        with open(old_index_filename, 'r') as old_index_file:
            return json.load(old_index_file)

    def write_index_file(self):
        index = self.build_transferred_index()
        with files.open_with_perms(os.path.join(self.destination_current, self.INDEX_NAME), 'w') as index_file:
            json.dump(index, index_file, indent=2)

    def write_transfer_type(self):
        with files.open_with_perms(os.path.join(self.destination_current, self.TRANSFER_TYPE_NAME), 'w') as type_file:
            type_file.write(self.transfer_type)

    def previous_transfer_type(self):
        try:
            with open(os.path.join(self.destination_current, self.TRANSFER_TYPE_NAME), 'r') as type_file:
                return type_file.read()
        except Exception:
            logger.info('No type file found')
            return None

    @classmethod
    def load_groups(cls, full_dict, groups):
        if "default" in full_dict:
            out_dict = cls.load_groups(full_dict["default"], groups)
        else:
            out_dict = dict()

        for key, value in full_dict.items():
            if not isinstance(value, dict):
                continue
            if value["type"] == "group" and key in groups:
                out_dict.update(cls.load_groups(value, groups))
            elif value["type"] != "group":
                out_dict[key] = value
        return out_dict

    def get_destination_path(self, info, local_destination_root=None):
        if info['multiple']:
            destination_directory = info['destination']
        else:
            destination_directory = os.path.dirname(info['destination'])

        if local_destination_root is None:
            local_destination_root = os.path.join(self.destination_root, self.label)

        destination_path = os.path.join(local_destination_root, destination_directory)
        return destination_path

    @classmethod
    def get_origin_files(cls, info, root='', **kwargs):
        try:
            origin_directory = info['origin_directory'].format(**kwargs)
        except:
            if info['required']:
                logger.error("Could not determine all keys for {} when formatting {}".format(info['destination'], info['origin_directory']))
                raise
            else:
                return []

        origin_directory = os.path.join(root, origin_directory)

        if info['type'] == 'link':
            origin_directory = os.path.relpath(os.path.realpath(origin_directory))

        origin_file_entry = info['origin_file']

        if not isinstance(origin_file_entry, list):
            origin_file_entry = [origin_file_entry]

        origin_files = []

        for origin_file in origin_file_entry:
            origin_filename = origin_file.format(**kwargs)
            origin_path = os.path.join(origin_directory, origin_filename)
            new_files = glob.glob(origin_path)

            if len(new_files) == 0:
                logger.debug("Could not find any files in {}".format(origin_path))
            origin_files += new_files

        return origin_files

    def missing_required_files(self):
        for name, info in self.transfer_dict.items():
            origin_files = self.get_origin_files(info, **self.kwargs)

            if info['required'] and not origin_files:
                logger.error("Could not locate file {} in {}".format(name,
                                                                     info['origin_directory'].format(**self.kwargs)))
                return name, info['origin_directory'].format(**self.kwargs)

            if (not info['multiple']) and len(origin_files) > 1:
                logger.error("multiple = {}, but {} files found".format(info['multiple'],
                                                                        len(origin_files)))
                return name, info['origin_directory'].format(**self.kwargs)
        return False, None

    def check_all_checksums(self):
        old_index = self.load_previous_index()

        md5s = defaultdict(hashlib.md5)

        self.get_dict_checksums(self.transfer_dict, md5s=md5s)

        for name, this_md5 in md5s.items():
            if not name in old_index:
                logger.info("Found new file: {}".format(name))
                return True

            if not old_index[name]['md5'] == this_md5.hexdigest():
                logger.info("Found differing file: {}".format(name))
                return True

        return False

    def get_dict_checksums(self, transfer_dict,  md5s=None, root_dir=''):
        transferrable_files = self.get_files_to_transfer(transfer_dict, root_dir)

        for transfer_item in transferrable_files:
            self.get_item_checksums(transfer_item, md5s, root_dir,)

    def get_item_checksums(self, transfer_item, md5s, root_dir=''):
        name = transfer_item['name']
        info = transfer_item['info']
        files = transfer_item['files']

        if info['type'] == 'directory':
            for file in files:
                self.get_dict_checksums(info['contents'], md5s, file)
            return

        this_md5 = md5s[name]

        for origin_file in files:
            if 'checksum_filename_only' in info and info['checksum_filename_only']:
                this_md5.update(os.path.basename(origin_file))
            else:
                this_md5.update(open(origin_file).read())


    def get_files_to_transfer(self, transfer_dict = None, root_dir=''):

        transferrable_files = []

        if transfer_dict is None:
            transfer_dict = self.transfer_dict

        for name, info in transfer_dict.items():
            logger.debug('Checking {}'.format(name))

            origin_files = self.get_origin_files(info, root_dir, **self.kwargs)

            if info['required'] and not origin_files:
                raise UnTransferrableException("Could not locate file {}".format(name))

            if (not info['multiple']) and len(origin_files) > 1:
                raise UnTransferrableException("multiple = {}, but {} files found".format(info['multiple'],
                                                                                          len(origin_files)))

            if not origin_files:
                continue

            transferrable_files.append({'name': name,
                                        'info': info,
                                        'files': origin_files,
                                        'directory': info['origin_directory'].format(**self.kwargs)})

        return transferrable_files

    def _transfer_files(self):
        if not os.path.exists(self.destination_root):
            files.makedirs(self.destination_root)

        logger.info('Transferring into {}'.format(self.destination_root))

        file_dicts = self.get_files_to_transfer()
        if not file_dicts:
            logger.info('No files to transfer.')
            self.transfer_aborted = True
            return None

        for file_dict in file_dicts:
            self.transferred_files[file_dict['name']] = self._transfer_file_entry(file_dict)


        if os.path.islink(self.destination_current):
            self.old_symlink = os.path.relpath(os.path.realpath(self.destination_current), self.destination_root)
            os.unlink(self.destination_current)
        os.symlink(self.label, self.destination_current)

        self.write_index_file()
        self.write_transfer_type()

        return self.transferred_files

    def _transfer_file_entry(self, file_dict, local_destination_root = None):
        name = file_dict['name']
        info = file_dict['info']
        origin_files = file_dict['files']

        logger.info('Transferring {}'.format(name))

        destination_path = self.get_destination_path(info, local_destination_root)
        if not os.path.exists(destination_path):
            files.makedirs(destination_path)

        this_files_transferred = []

        if local_destination_root is None:
            local_destination_root = os.path.join(self.destination_root, self.label)

        for origin_file in origin_files:
            if info['multiple']:
                destination_file_path = origin_file.replace(file_dict['directory'], '')
                while destination_file_path[0] == '/':
                    destination_file_path = destination_file_path[1:]
                destination_file = os.path.join(destination_path, destination_file_path)
            else:
                destination_file = os.path.join(local_destination_root, info['destination'])

            if info['type'] == 'file':
                if not os.path.exists(os.path.dirname(destination_file)):
                    files.makedirs(os.path.dirname(destination_file))
                shutil.copyfile(origin_file, destination_file)
                logger.debug('Copied file {} to {}'.format(origin_file, destination_file))
            elif info['type'] == 'directory':
                copied_files = self._copy_directory(info, origin_file, destination_file)
                logger.debug('Copied directory {} to {}'.format(origin_file, destination_file))
                destination_file = copied_files
            elif info['type'] == 'link':
                if os.path.islink(destination_file):
                    logger.warn('Removing link %s' % destination_file)
                    os.unlink(destination_file)
                link = os.path.relpath(os.path.realpath(origin_file), os.path.realpath(destination_path))
                os.symlink(link, destination_file)
                logger.debug('Linking {} to {}'.format(link, destination_file))
            else:
                raise UnTransferrableException("Type {} not known." + \
                                               "Must be 'file', 'directory', or 'link'".format(info['type']))
            this_files_transferred.append(destination_file)

        if not info['multiple'] and len(this_files_transferred) > 1:
            raise Exception("How did this happen??")

        return this_files_transferred if info['multiple'] else this_files_transferred[0]


    def _copy_directory(self, info, origin, destination):
        os.makedirs(destination)
        contents = info['contents']
        file_dicts = self.get_files_to_transfer(contents, origin)
        files_transferred = {}
        for file_dict in file_dicts:
            logger.debug("Copying directory contents {}".format(file_dict['name']))
            files_transferred[file_dict['name']] = self._transfer_file_entry(file_dict, destination)
        return files_transferred


    def transfer_with_rollback(self):
            try:
                return self._transfer_files()
            except Exception as e:
                logger.error('Exception encountered: %s' % e.message)

                self.remove_transferred_files()
                raise

    def remove_transferred_files(self):
        new_transferred_files = {k:v for k,v in self.transferred_files.items()}
        for file in self.transferred_files:
            logger.debug('Removing Entry : {} '.format(file))
            if file not in self.transfer_dict: # Must have been added later. This is messy. Should be fixed.
                os.remove(self.transferred_files[file])
                continue
            info = self.transfer_dict[file]
            destination_path = self.get_destination_path(info)

            origin_file_entry = info['origin_file']
            if not isinstance(origin_file_entry, list):
                origin_file_entry = [origin_file_entry]

            for origin_file in origin_file_entry:
                if info['multiple']:
                    destination_files = glob.glob(os.path.join(destination_path, origin_file.format(**self.kwargs)))
                    for destination_file in destination_files:
                        if info['type'] == 'file':
                            logger.debug('removing %s' % destination_file)
                            os.remove(destination_file)
                        elif info['type'] == 'directory':
                            logger.debug('removing %s' % destination_file)
                            shutil.rmtree(destination_file)
                else:
                    destination_file = os.path.join(self.destination_root, self.label, info['destination'])
                    logger.debug('removing %s' % destination_file)
                    if info['type'] == 'link':
                        os.unlink(destination_file)
                    else:
                        os.remove(destination_file)


            del new_transferred_files[file]
        self.transferred_files = new_transferred_files

        index_file = os.path.join(self.destination_current, self.INDEX_NAME)
        if os.path.exists(index_file):
            os.remove(index_file)

        transfer_type_file = os.path.join(self.destination_current, self.TRANSFER_TYPE_NAME)
        if os.path.exists(transfer_type_file):
            os.remove(transfer_type_file)


        self.delete_if_empty(self.destination_root)
        if os.path.islink(self.destination_current):
            logger.debug('removing symlink: %s' % self.destination_current)
            os.unlink(self.destination_current)
            if self.old_symlink:
                os.symlink(self.old_symlink, self.destination_current)
        else:
            logger.warn('Could not find symlink %s' % self.destination_current)


    @classmethod
    def delete_if_empty(cls, path):
        try:
            for (subpath, _, files) in os.walk(path, topdown=False):
                if len(files) == 1 and files[0] == 'log.txt':
                    os.remove(os.path.join(subpath, files[0]))
                if subpath == path:
                    continue
                cls.delete_if_empty(subpath)
            os.rmdir(path)
        except Exception as e:
            pass
            #print 'Couldn\'t remove {}: {}'.format(path, e)



def find_sync_file(subject, experiment, session):
    subject_dir = os.path.join(DATA_ROOT, subject)
    # Look in raw folder first
    raw_sess_dir = os.path.join(subject_dir, 'raw', '{exp}_{sess}'.format(exp=experiment, sess=session))
    sync_files = glob.glob(os.path.join(raw_sess_dir, '*.sync'))
    if len(sync_files) == 1:
        return raw_sess_dir, sync_files[0]
    # Now look in eeg.noreref
    noreref_dir = os.path.join(subject_dir, 'eeg.noreref')
    sync_pattern = os.path.join(noreref_dir, '*.{exp}_{sess}.sync.txt'.format(exp=experiment, sess=session))
    sync_files = glob.glob(sync_pattern)
    if len(sync_files) == 1:
        return noreref_dir, sync_files[0]
    # Now look for the exp_# anywhere in the basename
    sync_pattern = os.path.join(noreref_dir, '*{exp}_{sess}_*.sync.txt'.format(exp=experiment, sess=session))
    sync_files = glob.glob(sync_pattern)
    if len(sync_files) == 1:
        return noreref_dir, sync_files[0]
    raise UnTransferrableException("{} sync files found at {}, expected 1".format(len(sync_files), sync_pattern))


def generate_ephys_transferer(subject, experiment, session, protocol='r1', groups=tuple(),
                              code=None, original_session=None, new_experiment=None, **kwargs):
    json_file = TRANSFER_INPUTS['ephys']
    if new_experiment is None:
        new_experiment = experiment
    destination = os.path.join(DB_ROOT,
                               'protocols', protocol,
                               'subjects', subject,
                               'experiments', new_experiment,
                               'sessions', str(session),
                               'ephys')
    code = code or subject
    original_session = original_session if not original_session is None else session


    return Transferer(json_file, (experiment,) + groups, destination,
                      protocol=protocol,
                      subject=subject, experiment=experiment, new_experiment=new_experiment, session=original_session,
                      data_root=DATA_ROOT, db_root=DB_ROOT, events_root=EVENTS_ROOT,
                      code=code, original_session=original_session, **kwargs)



def generate_montage_transferer(subject, montage, protocol, code=None, groups=tuple(), **kwargs):

    groups = groups + ('json_import',)
    json_file = TRANSFER_INPUTS['montage']
    code = code or subject

    localization = montage.split('.')[0]
    montage_num = montage.split('.')[1]

    destination = os.path.join(DB_ROOT,
                               'protocols', protocol,
                               'subjects', subject,
                               'localizations', localization,
                               'montages', montage_num,
                               'neuroradiology')

    transferer = Transferer(json_file, groups, destination, protocol=protocol, subject=subject, code=code,
                            loc_db_root=LOC_DB_ROOT)
    return transferer


def generate_session_transferer(subject, experiment, session, protocol='r1', groups=tuple(), code=None,
                                original_session=None, new_experiment=None, **kwargs):
    json_file = TRANSFER_INPUTS['behavioral']

    code = code or subject
    original_session = original_session if not original_session is None else session

    kwarg_inputs = dict(protocol=protocol,
                        experiment=experiment, session=session,
                        code=code, original_session=original_session,
                        data_root=DATA_ROOT, events_root=EVENTS_ROOT, db_root=DB_ROOT, **kwargs)

    kwarg_inputs['subject'] = subject

    if 'system_1' in groups and 'transfer' in groups:
        try:
            kwarg_inputs['sync_folder'], kwarg_inputs['sync_filename'] = \
                find_sync_file(code, experiment, original_session)
        except UnTransferrableException:
            logger.warn("******* Could not find syncs! Will likely fail soon!")

    if not new_experiment:
        new_experiment = experiment

    destination = os.path.join(DB_ROOT,
                               'protocols', protocol,
                               'subjects', subject,
                               'experiments', new_experiment,
                               'sessions', str(session),
                               'behavioral')

    transferer= Transferer(json_file, (experiment,) + groups, destination, new_experiment=new_experiment,**kwarg_inputs)
    return transferer




def xtest_load_groups():
    from pprint import pprint
    transferer = Transferer('./behavioral_inputs.json', ('system_2', "TH"))
    pprint(transferer.transfer_dict)

def xtest_transfer_files_sys2():
    transferer = Transferer('./behavioral_inputs.json', ('system_2', 'r1'), '../tests/test_output/test_transfer',
                            data_root='../tests/test_data',
                            db_root=DB_ROOT,
                            subject='R1124J_1',
                            experiment='FR3',
                            session=1)
    transferer.transfer_with_rollback()


def test_transfer_files_sys3():
    transferer = Transferer('./transfer_inputs/behavioral_inputs.json', ('system_3', 'transfer', 'r1', 'PS'), '../tests/test_output/test_transfer',
                            data_root='../tests/test_input',
                            db_root=DB_ROOT,
                            code='R9999X',
                            subject='R9999X',
                            experiment='PS2',
                            new_experiment='PS2.1',
                            original_session=37,
                            session=0,
                            protocol='r1')

    if transferer.check_all_checksums():
        print 'Checksum failed, which is okay!'
    output = transferer.transfer_with_rollback()
    from pprint import pprint
    pprint(output)
    if transferer.check_all_checksums():
        print 'Checksum failed, which it should not!'
    else:
        print 'Checksum succeeded!'

def test_transfer_files_sys3_2():
    transferer = Transferer('./transfer_inputs/behavioral_inputs.json', ('system_3', 'transfer', 'r1', 'FR'), '../tests/test_output/test_transfer',
                            data_root='../tests/test_input',
                            db_root=DB_ROOT,
                            code='R9999X',
                            subject='R9999X',
                            experiment='FR1',
                            new_experiment='PS2.1',
                            original_session=0,
                            session=0,
                            protocol='r1')

    if transferer.check_all_checksums():
        print 'Checksum failed, which is okay!'
    output = transferer.transfer_with_rollback()
    from pprint import pprint
    pprint(output)
    if transferer.check_all_checksums():
        print 'Checksum failed, which it should not!'
    else:
        print 'Checksum succeeded!'

if __name__ == '__main__':
    logger.set_stdout_level(0)
    test_transfer_files_sys3_2()
