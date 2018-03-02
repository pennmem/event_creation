from __future__ import print_function

from submission.parsers.fr_log_parser import FRSessionLogParser
from submission.parsers.catfr_log_parser import CatFRSessionLogParser
from submission.parsers.fr_sys3_log_parser import FRSys3LogParser,catFRSys3LogParser
from submission.parsers.mat_converter import FRMatConverter,CatFRMatConverter
from submission.configuration import paths
import pandas as pd
import numpy as np



from ptsa.data.readers import JsonIndexReader,CMLEventReader
import os.path
import glob

import pytest
import conftest


beh_sess_root = os.path.join(paths.rhino_root, 'data', 'eeg', '{subject}', 'behavioral', '{experiment}', 'session_{session}')

 
session_log= os.path.join(  beh_sess_root, 'session.log')

 
session_sql= os.path.join(  beh_sess_root, 'session.sqlite')

 
ann_files= os.path.join(  beh_sess_root, '*.ann')

 
fr_wordpool=  os.path.join(os.path.dirname(  beh_sess_root), 'RAM_wordpool.txt')

catfr_wordpool = os.path.join(os.path.dirname(  beh_sess_root), 'CatFR_WORDS.txt')

reader= JsonIndexReader(os.path.join(paths.rhino_root, 'protocols', 'r1.json'))


fields_to_skip = ['eegfile','msoffset','eegoffset','stim_params','montage']

def make_files(wordpool,  **kwargs):
    if 'original_experiment' not in kwargs:
        kwargs['original_experiment'] = ''
    return {'session_log':  session_log.format(**kwargs),
            'session_sqlite':  session_sql.format(**kwargs),
            'wordpool':  wordpool.format(**kwargs),
            'matlab_events':os.path.join(paths.rhino_root,'data','events','{original_experiment}','{subject}_events.mat'
                                         ).format(**kwargs),
            'annotations':list(glob.glob(  ann_files.format(**kwargs)))}

def make_diff(old_events,new_events,fields):
    old_events,new_events = (pd.DataFrame.from_records([e for e in events],columns=events.dtype.names)[fields]
                             for events in [old_events,new_events])
    if old_events.shape != new_events.shape:
        diff= (old_events,new_events)
    else:
        diff=   (old_events.where(old_events!=new_events),new_events.where(old_events != new_events))

    diff= pd.concat(diff,axis=1,keys=['old','new'])
    diff[('old','type')] = old_events['type']
    diff[('new','type')] = new_events['type']
    for multi_col in diff.columns:
        if 'type' not in multi_col:
            if diff[multi_col].isnull().all():
                diff.drop(multi_col,axis=1,inplace=True)
    return diff


@pytest.fixture(params=[8,-2])
def old_fr_subject(  request):
    return [s for s in   reader.subjects(experiment='FR1',import_type='build') if not
    any([v>=3 for v  in   reader.aggregate_values('system_version',subject=s,experiment='FR1')])][request.param]

@pytest.fixture(params=[8,-2])
def new_fr_subject(  request):
    return   reader.subjects(experiment='FR1',system_version=3.1)[request.param]

@pytest.fixture(params=[0,-2])
def old_catfr_subject(  request):
    return [s for s in reader.subjects(experiment='catFR1',import_type='build') if not
    any([v >= 3 for v in reader.aggregate_values('system_version', subject=s)])][
        request.param]

@pytest.fixture(params=[0,-2])
def new_catfr_subject(  request):
    return   reader.subjects(experiment='catFR1',system_version=3.1)[request.param]

def test_across_fr_parsers(old_fr_subject, new_fr_subject, ):
    session =    reader.sessions(subject=old_fr_subject,experiment='FR1')[0]
    old_fr1_files =   make_files(fr_wordpool,
        subject=old_fr_subject,experiment='FR1',session=session
    )
    old_fr1_events = FRSessionLogParser('r1', old_fr_subject, '0', 'FR1', session, old_fr1_files).parse()
    new_fr1_files =   make_files(fr_wordpool,subject=new_fr_subject, experiment='FR1', session=session)
    new_fr1_events = FRSys3LogParser('r1', new_fr_subject, 0, 'FR1', session, new_fr1_files, primary_log='session_sqlite').parse()
    if conftest.from_test:
        assert old_fr1_events.dtype.names == new_fr1_events.dtype.names
        assert old_fr1_events['stim_params'].dtype.names == new_fr1_events['stim_params'].dtype.names
    np.concatenate([old_fr1_events,new_fr1_events])

def test_against_existing_fr_data(   old_fr_subject):
    session=   reader.sessions(subject=old_fr_subject,experiment='FR1')[0]
    old_events = CMLEventReader(filename=  reader.get_value('task_events', subject=old_fr_subject, experiment='FR1', session=session)
                                ).read()
    files =   make_files(fr_wordpool,subject=old_fr_subject, experiment='FR1', session=session)
    new_events = FRSessionLogParser('r1', old_fr_subject, '0', 'FR1', session, files).parse()
    assert all(f in new_events.dtype.names for f in old_events.dtype.names)
    comparison_fields = [f for f in old_events.dtype.names if f not in  fields_to_skip]
    if conftest.from_test:
        for field in comparison_fields:
                assert (old_events[field]==new_events[field]).all() , '%d mismatched values for field %s'%(
                    (old_events[field] != new_events[field]).sum(), field)
    else:
        return make_diff(old_events,new_events,comparison_fields)

def test_across_catfr_parsers(   old_catfr_subject, new_catfr_subject):
    session =    reader.sessions(subject=old_catfr_subject,experiment='catFR1')[0]

    old_catfr1_files =   make_files(catfr_wordpool,
        subject=old_catfr_subject,experiment='catFR1',session=session
    )
    old_catfr1_events = CatFRSessionLogParser('r1', old_catfr_subject, '0', 'catFR1', session, old_catfr1_files).parse()
    new_catfr1_files =   make_files(catfr_wordpool,subject=new_catfr_subject, experiment='catFR1', session=session)
    new_catfr1_events = catFRSys3LogParser('r1', new_catfr_subject, 0, 'catFR1', session, new_catfr1_files, primary_log='session_sqlite').parse()
    if conftest.from_test:
        assert old_catfr1_events.dtype.names == new_catfr1_events.dtype.names
        assert old_catfr1_events['stim_params'].dtype.names == new_catfr1_events['stim_params'].dtype.names
    np.concatenate([old_catfr1_events,new_catfr1_events])

def test_against_existing_catfr_data(   old_catfr_subject):
    session =    reader.sessions(subject=old_catfr_subject,experiment='catFR1')[0]

    old_events = CMLEventReader(filename=  reader.get_value('task_events', subject=old_catfr_subject, experiment='catFR1', session=session)
                                ).read()
    files =   make_files(catfr_wordpool,
        subject=old_catfr_subject, experiment='catFR1', session=session)
    new_events = CatFRSessionLogParser('r1', old_catfr_subject, '0', 'catFR1', session, files).parse()
    fields_to_compare = [f for f in old_events.dtype.names if f not in fields_to_skip]
    if conftest.from_test:
        for field in fields_to_compare:
            assert (old_events[field]==new_events[field]).all() , '%d mismatched values for field %s'%(
                (old_events[field] != new_events[field]).sum(), field)
    else:
         return make_diff(old_events,new_events,fields_to_compare)

@pytest.fixture(params=[3,8,10])
def matlab_fr_subject(request):
    event_file=  list(glob.glob(
        os.path.join(paths.rhino_root,'data','events','RAM_FR1','R1*_events.mat')
    ))[request.param]
    return event_file.split('/')[-1].split('_')[0]

@pytest.fixture(params=[3,8,10])
def matlab_catfr_subject(request):
    event_file=  list(glob.glob(
        os.path.join(paths.rhino_root,'data','events','RAM_CatFR1','R1*_events.mat')
    ))[request.param]
    return event_file.split('/')[-1].split('_')[0]



def test_matlab_fr_conversion(old_fr_subject,matlab_fr_subject):
    session=   reader.sessions(subject=old_fr_subject, experiment='FR1')[0]
    files = make_files(fr_wordpool,subject=old_fr_subject,experiment='FR1',session=session)
    parsed_events = FRSessionLogParser('r1', old_fr_subject, '0', 'FR1', session, files).parse()
    session=   reader.sessions(subject=matlab_fr_subject, experiment='FR1')[0]

    files =   make_files(fr_wordpool, subject=matlab_fr_subject, experiment='FR1', session=session, original_experiment='RAM_FR1')
    converted_events = FRMatConverter('r1', matlab_fr_subject, '0', 'FR1', session, session, files).convert()
    assert parsed_events.dtype == converted_events.dtype


def test_matlab_catfr_conversion(old_catfr_subject, matlab_catfr_subject):
    session=   reader.sessions(subject=old_catfr_subject, experiment='catFR1')[0]
    files = make_files(catfr_wordpool, subject=old_catfr_subject, experiment='catFR1', session=session)
    parsed_events = CatFRSessionLogParser('r1', old_catfr_subject, '0', 'catFR1', session, files).parse()
    session=   reader.sessions(subject=matlab_catfr_subject, experiment='catFR1')[0]

    files =   make_files(catfr_wordpool, subject=matlab_catfr_subject, experiment='catFR1', session=session, original_experiment='RAM_CatFR1')
    converted_events = CatFRMatConverter('r1', matlab_catfr_subject, '0', 'catFR1', session, session, files).convert()
    assert parsed_events.dtype == converted_events.dtype



if __name__ == '__main__':
    print('Running script')
    fr_subjects = ['R1061T','R1136N','R1166D','R1122E','R1236J','R1264P']
    print('Computing FR1 differences')
    for subject in fr_subjects:
        fr_diffs = test_against_existing_fr_data(subject)
        print(subject + ' diffs.shape: '+str(fr_diffs.shape))
        # fr_diffs.to_excel(excel_writer=writer,sheet_name='FR1_'+subject)
        fr_diffs.to_csv('FR1_'+subject+'_diffs.csv')
    print('Computing catFR1 differences')
    catfr_subjects = ['R1056M','R1192C','R1273D','R1189M','R1021D','R1288P']
    for subject in catfr_subjects:
        test_against_existing_catfr_data(subject).to_csv('catFR1_'+subject+'_diffs.csv')



