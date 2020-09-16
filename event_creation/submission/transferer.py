import datetime
import glob
import json
import os
import traceback

import yaml

from . import fileutil
from .exc import TransferError
from .configuration import paths
from .log import logger
from .transfer_config import TransferConfig
from .transfer_inputs import TRANSFER_INPUTS



def yml_join(loader, node):
    return os.path.join(*[str(i) for i in loader.construct_sequence(node)])

yaml.add_constructor('!join', yml_join)


class Transferer(object):

    CURRENT_NAME = 'current_source'
    INDEX_NAME='index.json'
    STRFTIME = '%Y%m%d.%H%M%S'
    TRANSFER_TYPE_NAME='TRANSFER_TYPE'

    JSON_FILES = {}

    def __init__(self, config_filename, groups, destination, **kwargs):
        self.groups = groups
        self.destination_root = os.path.abspath(destination)
        self.destination_current = os.path.join(self.destination_root, self.CURRENT_NAME)

        self._label = datetime.datetime.now().strftime(self.STRFTIME)
        logger.debug('Transferer {} created'.format(self._label))
        self.destination_labelled = os.path.join(self.destination_root, self._label)


        self.kwargs = kwargs
        self.transferred_files = []
        self.transferred_filenames = {}

        self.transfer_config = TransferConfig(config_filename, groups, **kwargs)

        self.old_symlink = None
        self.transfer_aborted = False
        self.previous_label = self.get_current_target()
        self._should_transfer = True

        self.transfer_type = 'IMPORT'

    @property
    def label(self):
        if self.transfer_aborted:
            return self.previous_label
        else:
            return self._label

    def missing_files(self):
        logger.debug("Searching for missing files")
        self.transfer_config.locate_origin_files()
        return self.transfer_config.missing_files()

    def set_transfer_type(self, transfer_type):
        self.transfer_type = transfer_type

    def transferred_index(self):
        index = {}
        for file in self.transferred_files:
            index.update(file.transferred_index())
        return index

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
            old_index = json.load(old_index_file)
            if old_index is None:
                return {}
            return old_index

    def write_transferred_index(self):
        index = self.transferred_index()
        with fileutil.open_with_perms(os.path.join(self.destination_current, self.INDEX_NAME), 'w') as index_file:
            json.dump(index, index_file, indent=2)

    def write_transfer_type(self):
        with fileutil.open_with_perms(os.path.join(self.destination_current, self.TRANSFER_TYPE_NAME), 'w') as type_file:
            type_file.write(self.transfer_type)

    def previous_transfer_type(self):
        try:
            with open(os.path.join(self.destination_current, self.TRANSFER_TYPE_NAME), 'r') as type_file:
                return type_file.read()
        except Exception:
            logger.info('No type file found')
            return None

    def matches_existing_checksum(self):
        old_index = self.load_previous_index()
        self.transfer_config.locate_origin_files()
        for file in self.transfer_config.located_files():

            if file.name not in old_index:
                logger.info("Found new file: {}".format(file.name))
                return False

            if not file.matches_transferred_index(old_index[file.name]):
                logger.info("Found differing file: {}".format(file.name))
                return False

        return True

    def _transfer_files(self):
        if not os.path.exists(self.destination_root):
            fileutil.makedirs(self.destination_root)

        logger.info('Transferring into {}'.format(self.destination_root))

        if len(self.transfer_config.located_files()) == 0:
            logger.info("No files to transfer.")
            self.transfer_aborted = True
            raise TransferError("No files to transfer")

        for file in self.transfer_config.located_files():
            file.transfer(self.destination_labelled)
            self.transferred_files.append(file)
            self.transferred_filenames.update(file.transferred_filenames())

        if os.path.islink(self.destination_current):
            self.old_symlink = os.path.relpath(os.path.realpath(self.destination_current), self.destination_root)
            os.unlink(self.destination_current)

        os.symlink(self.label, self.destination_current)

        self.write_transferred_index()
        self.write_transfer_type()

        return self.transferred_filenames

    def transfer_with_rollback(self):
        try:
            return self._transfer_files()
        except Exception as e:
            traceback.print_exc()
            logger.error('Exception encountered: %s' % e.message)

            self.remove_transferred_files()
            raise

    def remove_transferred_files(self):
        if os.path.islink(self.destination_current):
            logger.debug("Removing symlink: {}".format(self.destination_current))
            os.unlink(self.destination_current)
            if self.old_symlink:
                os.symlink(self.old_symlink, self.destination_current)
        else:
            logger.warn("Could not find symlink {}".format(self.destination_current))

        self.warn_and_delete(self.destination_labelled)

    @classmethod
    def warn_and_delete(cls, path):
        try:
            for (subpath, subdirs, files) in os.walk(path, topdown=False):
                for file in files:
                    logger.info("Removing file {}".format(file))
                    if os.path.islink(os.path.join(subpath, file)):
                        os.unlink(os.path.join(subpath, file))
                    else:
                        os.remove(os.path.join(subpath, file))
                for subdir in subdirs:
                    logger.info("Removing directory {}".format(subdir))
                    os.rmdir(os.path.join(subpath, subdir))
            os.rmdir(path)
        except Exception as e:
            logger.warn("Failed to delete a file! {}".format(e))
            traceback.print_exc()



def find_sync_file(subject, experiment, session):
    subject_dir = os.path.join(paths.data_root, subject)
    # Look in raw folder first
    raw_sess_dir = os.path.join(subject_dir, 'raw', '{exp}_{sess}'.format(exp=experiment, sess=session))
    sync_files = glob.glob(os.path.join(raw_sess_dir, '*.sync.txt'))
    if len(sync_files) == 1:
        return raw_sess_dir, sync_files[0]

    # Now look in eeg.noreref
    noreref_dir = os.path.join(subject_dir, 'eeg.noreref')
    sync_pattern = os.path.join(noreref_dir, '*{exp}_{sess}.sync.txt'.format(exp=experiment, sess=session))
    sync_files = glob.glob(sync_pattern)
    if len(sync_files) == 1:
        return noreref_dir, sync_files[0]

    # Now look for the exp_# anywhere in the basename
    sync_pattern = os.path.join(noreref_dir, '*{exp}_{sess}*.sync.txt'.format(exp=experiment, sess=session))
    sync_files = glob.glob(sync_pattern)
    if len(sync_files) == 1:
        return noreref_dir, sync_files[0]

    sync_pattern = os.path.join(noreref_dir, '*{exp}_{sess}*.sync.txt'.format(exp=experiment, sess=session))
    sync_files = glob.glob(sync_pattern)
    if len(sync_files) == 1:
        return noreref_dir, sync_files[0]

    raw_sess_dir = os.path.join(subject_dir, 'raw', 'session_{sess}'.format(exp=experiment, sess=session), "")
    sync_pattern = os.path.join(raw_sess_dir, '*int32_sync.txt')
    sync_files = glob.glob(sync_pattern)
    if len(sync_files) == 1:
        print("found freiburg sync")
        return raw_sess_dir, sync_files[0]

    raw_sess_dir = os.path.join(subject_dir, 'raw', '{exp}_{sess}'.format(exp=experiment, sess=session), "")
    sync_pattern = os.path.join(raw_sess_dir, '*int32_sync.txt')
    sync_files = glob.glob(sync_pattern)
    if len(sync_files) == 1:
        print("found freiburg sync")
        return raw_sess_dir, sync_files[0]

    raise TransferError("{} sync files found at {}, expected 1".format(len(sync_files), sync_pattern))


def generate_wav_transferer(subject,experiment,session,protocol='r1',groups=('r1'),
                            original_session=None,new_experiment=None,**kwargs):
    cfg_file = TRANSFER_INPUTS['wav']
    dest = os.path.join(paths.db_root,'protocols',protocol,'subjects',subject,'experiments',experiment,
                        'sessions',str(session),'behavioral')
    original_session = session if original_session is None else original_session
    new_experiment = experiment if new_experiment is None else new_experiment

    return Transferer(cfg_file,groups,dest,
                      protocol=protocol,subject=subject,experiment=experiment, session = session,
                      original_session=original_session,new_experiment = new_experiment,
                      data_root=paths.data_root,db_root=paths.db_root,events_root=paths.events_root,**kwargs)

def generate_ephys_transferer(subject, experiment, session, protocol='r1', groups=tuple(),
                              code=None, original_session=None, new_experiment=None, **kwargs):
    cfg_file = TRANSFER_INPUTS['ephys']
    if new_experiment is None:
        new_experiment = experiment
    destination = os.path.join(paths.db_root,
                               'protocols', protocol,
                               'subjects', subject,
                               'experiments', new_experiment,
                               'sessions', str(session),
                               'ephys')
    code = code or subject
    original_session = original_session if not original_session is None else session


    return Transferer(cfg_file, (experiment,) + groups, destination,
                      protocol=protocol,
                      subject=subject, experiment=experiment, new_experiment=new_experiment, session=original_session,
                      data_root=paths.data_root, db_root=paths.db_root, events_root=paths.events_root,
                      code=code, original_session=original_session, **kwargs)

def generate_localization_transferer(subject, protocol, localization, code, is_new):

    cfg_file = TRANSFER_INPUTS['localization']

    destination = os.path.join(paths.db_root,
                               'protocols', protocol,
                               'subjects', subject,
                               'localizations', localization,
                               'neuroradiology')

    if is_new:
        groups = ('new', )
    else:
        groups = ('old', )

    return Transferer(cfg_file, groups, destination,
                      protocol=protocol,
                      subject=subject,
                      localization=localization,
                      code=code,
                      data_root=paths.data_root, db_root=paths.db_root)


def generate_import_montage_transferer(subject, montage, protocol, code=None, groups=tuple(), **kwargs):

    groups = groups + ('r1', 'json_import',)
    cfg_file = TRANSFER_INPUTS['montage']
    code = code or subject

    localization = montage.split('.')[0]
    montage_num = montage.split('.')[1]

    destination = os.path.join(paths.db_root,
                               'protocols', protocol,
                               'subjects', subject,
                               'localizations', localization,
                               'montages', montage_num,
                               'neuroradiology')

    transferer = Transferer(cfg_file, groups, destination, protocol=protocol, subject=subject, code=code,
                            loc_db_root=paths.loc_db_root)
    return transferer


def generate_create_montage_transferer(subject,montage,protocol,code=None,groups=tuple(),**kwargs):
    groups += ('r1',)
    cfg_file = TRANSFER_INPUTS['montage']
    code = code or subject

    localization = montage.split('.')[0]
    montage_num = montage.split('.')[1]

    destination = os.path.join(paths.db_root,
                               'protocols', protocol,
                               'subjects', subject,
                               'localizations', localization,
                               'montages', montage_num,
                               'neuroradiology')

    transferer = Transferer(cfg_file, groups, destination, protocol=protocol, subject=subject, code=code,
                            localization=localization,montage=montage,
                            loc_db_root=paths.loc_db_root,data_root=paths.data_root,db_root=paths.db_root)
    return transferer



def generate_session_transferer(subject, experiment, session, protocol='r1', groups=tuple(), code=None,
                                original_session=None, new_experiment=None, **kwargs):
    cfg_file = TRANSFER_INPUTS['behavioral']

    code = code or subject
    original_session = original_session if not original_session is None else session

    kwarg_inputs = dict(protocol=protocol,
                        experiment=experiment, session=session,
                        code=code, original_session=original_session,
                        data_root=paths.data_root, events_root=paths.events_root, db_root=paths.db_root, **kwargs)

    kwarg_inputs['subject'] = subject

    if 'system_1' in groups and 'transfer' in groups:
        try:
            kwarg_inputs['sync_folder'], kwarg_inputs['sync_filename'] = \
                find_sync_file(code, experiment, original_session)
        except TransferError:
            logger.warn("******* Could not find syncs! Will likely fail soon!")

    if not new_experiment:
        new_experiment = experiment

    destination = os.path.join(paths.db_root,
                               'protocols', protocol,
                               'subjects', subject,
                               'experiments', new_experiment,
                               'sessions', str(session),
                               'behavioral')


    transferer= Transferer(cfg_file, (experiment,) + groups, destination, new_experiment=new_experiment,**kwarg_inputs)
    return transferer




def test_load_groups():
    groups = ['r1', 'transfer', 'ltp', 'system_1', 'system_2', 'system_3']

    import itertools

    combos = []
    for i in range(len(groups)):
        combos.extend(itertools.combinations(groups, i))

    for combo in combos:

        print '----- {} '.format(combo)

        transferer2 = Transferer('./transfer_inputs/behavioral_inputs.yml', combo, '/Users/iped/PycharmProjects/event_creation/tests/test_data/test_output',
                                 db_root=paths.db_root, protocol='r1', subject='R9999X', new_experiment='FR1', session=0,
                                 localization=0, montage_num=0, data_root=paths.data_root, experiment='FR1', code='R9999X',
                                 original_session=0, sync_folder='')

        print combo, [f.name for f in transferer2.transfer_config.valid_files]


def transfer_dict_match(a ,b):
    for k, v_a in a.items():
        if k not in b:
            print 'KEY MISSING: {}'.format(k)
            continue
        v_b = b[k]
        for kk, vv_a in v_a.items():
            if kk == 'checksum_filename_only' and not vv_a:
                continue
            if kk not in v_b :
                print 'Missing key {} {}'.format(k, kk)
                continue
            vv_b = v_b[kk]
            if vv_a != vv_b:
                print 'Mismatch {} {}: {} vs {}'.format(k, kk, vv_a, vv_b)


def xtest_transfer_files_sys2():
    transferer = Transferer('./behavioral_inputs.json', ('system_2', 'r1'), '../tests/test_output/test_transfer',
                            data_root='../tests/test_data',
                            db_root=paths.db_root,
                            subject='R1124J_1',
                            experiment='FR3',
                            session=1)
    transferer.transfer_with_rollback()


def test_transfer_files_sys3():
    transferer = Transferer('./transfer_inputs/behavioral_inputs.json', ('system_3', 'transfer', 'r1', 'PS'), '../tests/test_output/test_transfer',
                            data_root='../tests/test_input',
                            db_root=paths.db_root,
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

def xtest_transfer_files_sys3_2():
    transferer = Transferer('./transfer_inputs/behavioral_inputs.json', ('system_3', 'transfer', 'r1', 'FR'), '../tests/test_output/test_transfer',
                            data_root='../tests/test_input',
                            db_root=paths.db_root,
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
    test_load_groups()
