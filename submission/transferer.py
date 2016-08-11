import json
import os
import glob
import shutil
import re
import datetime


from parsers.base_log_parser import get_version_num
from parsers.fr_log_parser import FRSessionLogParser
from parsers.catfr_log_parser import CatFRSessionLogParser
from parsers.math_parser import MathSessionLogParser
from parsers.pal_log_parser import PALSessionLogParser

from alignment.system1 import System1Aligner
from alignment.system2 import System2Aligner

from readers.eeg_reader import get_eeg_reader

from viewers.view_recarray import to_json

from loggers import log

DATA_ROOT='/Volumes/rhino_mount/data/eeg'
DB_ROOT='../tests/test_output'

class UnTransferrableException(Exception):
    pass

class Transferer(object):

    CURRENT_NAME = 'current_source'

    def __init__(self, json_file, groups, destination, **kwargs):
        self.groups = groups
        self.destination_root = os.path.abspath(destination)
        self.destination_current = os.path.join(self.destination_root, self.CURRENT_NAME)
        self.label = datetime.datetime.now().strftime('%y%m%d.%H%M')
        self.kwargs = kwargs
        self.transferred_files = {}
        self.transfer_dict = self.load_groups(json.load(open(json_file)), groups)
        self.old_symlink = None

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

    def get_destination_path(self, info):
        if info['multiple']:
            destination_directory = info['destination']
        else:
            destination_directory = os.path.dirname(info['destination'])

        destination_path = os.path.join(self.destination_root, self.label, destination_directory)
        return destination_path

    @classmethod
    def get_origin_files(cls, info, **kwargs):
        origin_directory = info['origin_directory'].format(**kwargs)

        if info['type'] == 'link':
            origin_directory = os.path.relpath(os.path.realpath(origin_directory))

        origin_file_entry = info['origin_file']

        if not isinstance(origin_file_entry, list):
            origin_file_entry = [origin_file_entry]

        origin_files = []

        for origin_file in origin_file_entry:
            origin_filename = origin_file.format(**kwargs)
            origin_path = os.path.join(origin_directory, origin_filename)

            origin_files += glob.glob(origin_path)

        return origin_files

    def _transfer_files(self):
        if not os.path.exists(self.destination_root):
            os.makedirs(self.destination_root)

        log('Transferring into {}'.format(self.destination_root))

        for name, info in self.transfer_dict.items():
            log('Transferring {}'.format(name))

            origin_files = self.get_origin_files(info, **self.kwargs)

            if info['required'] and not origin_files:
                raise UnTransferrableException("Could not locate file {}".format(name))

            if (not info['multiple']) and len(origin_files) > 1:
                raise UnTransferrableException("multiple = {}, but {} files found".format(info['multiple'],
                                                                                          len(origin_files)))

            destination_path = self.get_destination_path(info)
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)

            this_files_transferred = []
            origin_file = None
            for origin_file in origin_files:
                if info['multiple']:
                    destination_file = os.path.join(destination_path, os.path.basename(origin_file))
                else:
                    destination_file = os.path.join(self.destination_root, self.label, info['destination'])

                if info['type'] == 'file':
                    shutil.copyfile(origin_file, destination_file)
                elif info['type'] == 'directory':
                    shutil.copytree(origin_file, destination_file)
                elif info['type'] == 'link':
                    os.symlink(os.path.relpath(origin_file, destination_path), destination_file)
                else:
                    raise UnTransferrableException("Type {} not known."+\
                                                   "Must be 'file' or 'directory'".format(info['type']))
                this_files_transferred.append(destination_file)

            self.transferred_files[name] = this_files_transferred if len(this_files_transferred) != 1 else \
                                           this_files_transferred[0]


        if os.path.islink(self.destination_current):
            self.old_symlink = os.path.relpath(os.path.realpath(self.destination_current), self.destination_root)
            os.unlink(self.destination_current)
        os.symlink(self.label, self.destination_current)
        return self.transferred_files

    def transfer_with_rollback(self):
            try:
                return self._transfer_files()
            except Exception as e:
                log('Exception encountered!')

                self.remove_transferred_files()
                raise

    def remove_transferred_files(self):
        new_transferred_files = {k:v for k,v in self.transferred_files.items()}
        for file in self.transferred_files:
            log('Removing Entry : {} '.format(file))
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
                            log('removing %s' % destination_file)
                            os.remove(destination_file)
                        elif info['type'] == 'directory':
                            log('removing %s' % destination_file)
                            shutil.rmtree(destination_file)
                else:
                    destination_file = os.path.join(self.destination_root, self.label, info['destination'])
                    log('removing %s' % destination_file)
                    if info['type'] == 'link':
                        os.unlink(destination_file)
                    else:
                        os.remove(destination_file)


            del new_transferred_files[file]
        self.transferred_files = new_transferred_files
        self.delete_if_empty(self.destination_root)
        if os.path.islink(self.destination_current):
            log('removing symlink: %s' % self.destination_current)
            os.unlink(self.destination_current)
            if self.old_symlink:
                os.symlink(self.old_symlink, self.destination_current)
        else:
            log('Could not find symlink %s' % self.destination_current)

    @classmethod
    def delete_if_empty(cls, path):
        try:
            for (subpath, _, _) in os.walk(path):
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
    sync_files = glob.glob(os.path.join(noreref_dir, '*.{exp}_{sess}.sync.txt'.format(exp=experiment, sess=session)))
    if len(sync_files) == 1:
        return noreref_dir, sync_files[0]
    else:
        raise UnTransferrableException("%s sync files found, expected 1" % len(sync_files))


def generate_ephys_transferer(subject, experiment, session, protocol='R1', groups=tuple(), **kwargs):
    json_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ephys_inputs.json')
    destination = os.path.join(DB_ROOT,
                               'protocols', protocol,
                               'subjects', subject,
                               'experiments', experiment,
                               'sessions', str(session),
                               'ephys')
    return Transferer(json_file, (experiment,) + groups, destination,
                      protocol=protocol,
                      subject=subject, experiment=experiment, session=session,
                      data_root=DATA_ROOT, db_root=DB_ROOT, **kwargs)


class SplitEEGTask(object):

    SPLIT_FILENAME = '{subject}_{experiment}_{session}_{time}'

    def __init__(self, subject, montage, experiment, session, **kwargs):
        self.name = 'Splitting {subj}: {exp}_{sess}'.format(subj=subject, exp=experiment, sess=session)
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.kwargs = kwargs

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def run(self, files, db_folder):
        raw_eegs = files['raw_eeg']
        if not isinstance(raw_eegs, list):
            raw_eegs = [raw_eegs]
        for raw_eeg in raw_eegs:
            reader = get_eeg_reader(raw_eeg,
                                    files['jacksheet'])
            split_eeg_filename = self.SPLIT_FILENAME.format(subject=self.subject,
                                                            experiment=self.experiment,
                                                            session=self.session,
                                                            time=reader.get_start_time_string())
            reader.split_data(self.pipeline.destination, split_eeg_filename)


def generate_session_transferer(subject, experiment, session, protocol='R1', groups=tuple(), **kwargs):
    groups = groups+(re.sub('\d', '', experiment),)
    json_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'behavioral_inputs.json')
    source_files = Transferer.load_groups(json.load(open(json_file)), groups)

    kwarg_inputs = dict(protocol=protocol,
                        subject=subject, experiment=experiment, session=session,
                        data_root=DATA_ROOT, db_root=DB_ROOT, **kwargs)

    session_log = Transferer.get_origin_files(source_files['session_log'], **kwarg_inputs)[0]
    is_sys2 = get_version_num(session_log) >= 2

    if is_sys2:
        groups += ('system_2', )
    else:
        groups += ('system_1', )
        kwarg_inputs['sync_folder'], kwarg_inputs['sync_filename'] = find_sync_file(subject, experiment, session)

    destination = os.path.join(DB_ROOT,
                               'protocols', protocol,
                               'subjects', subject,
                               'experiments', experiment,
                               'sessions', str(session),
                               'behavioral')

    transferer= Transferer(json_file, (experiment,) + groups, destination, **kwarg_inputs)
    return transferer, groups



class EventCreationTask(object):

    PARSERS = {
        'FR': FRSessionLogParser,
        'PAL': PALSessionLogParser,
        'catFR': CatFRSessionLogParser,
        'math': MathSessionLogParser
    }

    def __init__(self, subject, montage, experiment, session, is_sys2, event_label='task', parser_type=None, **kwargs):
        self.name = 'Event Creation for {subj}: {exp}_{sess}'.format(subj=subject, exp=experiment, sess=session)
        self.parser_type = parser_type or self.PARSERS[re.sub(r'\d', '', experiment)]
        self.subject = subject
        self.montage = montage
        self.experiment = experiment
        self.session = session
        self.is_sys2 = is_sys2
        self.kwargs = kwargs
        self.filename = '{label}_events.json'.format(label=event_label)
        self.pipeline = None

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def run(self, files, db_folder):
        parser = self.parser_type(self.subject, self.montage, files)
        unaligned_events = parser.parse()
        if self.is_sys2:
            aligner = System2Aligner(unaligned_events, files)
            events = aligner.align('SESS_START')
        else:
            aligner = System1Aligner(unaligned_events, files)
            events = aligner.align()
        with open(os.path.join(self.pipeline.destination, self.filename), 'w') as f:
            to_json(events, f)


class TransferPipeline(object):

    CURRENT_PROCESSED_DIRNAME = 'current_processed'

    def __init__(self, transferer, *pipeline_tasks):
        self.transferer = transferer
        self.pipeline_tasks = pipeline_tasks
        self.exports = {}
        self.label = '{}_processed'.format(self.transferer.label)
        self.destination_root = self.transferer.destination_root
        self.destination = os.path.join(self.destination_root, self.label)
        self.current_dir = os.path.join(self.destination_root, self.CURRENT_PROCESSED_DIRNAME)
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)
        for task in self.pipeline_tasks:
            task.set_pipeline(self)

    def run(self):
        transferred_files = self.transferer.transfer_with_rollback()
        pipeline_task = None
        try:
            for i, pipeline_task in enumerate(self.pipeline_tasks):

                log('Executing task {}: {}'.format(i+1, pipeline_task.name))
                pipeline_task.run(transferred_files, self.destination)
                log('Task {} finished successfully'.format(pipeline_task.name))

            if os.path.islink(self.current_dir):
                os.unlink(self.current_dir)
            os.symlink(self.label, self.current_dir)

        except:
            log('Task {} failed!!\nRolling back transfer'.format(pipeline_task.name if pipeline_task else
                                                                   'initialization'))
            self.transferer.remove_transferred_files()
            if os.path.exists(self.destination):
                log('Removing processed folder {}'.format(self.destination))
                shutil.rmtree(self.destination)
            raise

GROUPS = {
    'FR': ('verbal',)
}

def build_split_pipeline(subject, montage, experiment, session, protocol='R1', groups=tuple(), **kwargs):
    transferer = generate_ephys_transferer(subject, experiment, session, protocol, groups, **kwargs)
    task = SplitEEGTask(subject, montage, experiment, session, **kwargs)
    return TransferPipeline(transferer, task)

def build_events_pipeline(subject, montage, experiment, session, do_math=True, protocol='R1', groups=tuple(), **kwargs):
    exp_type = re.sub('\d','', experiment)
    if exp_type in GROUPS:
        groups += GROUPS[exp_type]

    transferer, groups = generate_session_transferer(subject, experiment, session, protocol, groups, **kwargs)
    tasks = [EventCreationTask(subject, montage, experiment, session, 'system_2' in groups)]
    if do_math:
        tasks.append(EventCreationTask(subject, montage, experiment, session, 'system_2' in groups, 'math', MathSessionLogParser))
    return TransferPipeline(transferer, *tasks)


def xtest_load_groups():
    from pprint import pprint
    transferer = Transferer('./behavioral_inputs.json', ('system_2', "TH"))
    pprint(transferer.transfer_dict)

def xtest_transfer_files():
    transferer = Transferer('./behavioral_inputs.json', ('system_2'), '../tests/test_output/test_transfer',
                            data_root='../tests/test_data',
                            db_root=DB_ROOT,
                            subject='R1124J_1',
                            experiment='FR3',
                            session=1)
    transferer.transfer_with_rollback()

def test_split_pipeline():
    pipeline = build_split_pipeline('R1083J', '0.0', 'FR1', 0)
    pipeline.run()

def test_events_pipeline():
    pipeline = build_events_pipeline('R1083J', '0.0', 'FR1', 0)
    pipeline.run()
