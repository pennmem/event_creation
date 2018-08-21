import json

import numpy as np
import pandas as pd

from event_creation.submission.quality.util import as_recarray
from ..log import logger


def with_time_field(function):
    import functools

    @functools.wraps(function)
    def wrapped(events, files):
        time_field = 'eegoffset' if 'FR6' in events[0]['experiment'] else 'mstime'
        return function(events,files,time_field)
    return wrapped


@as_recarray
def test_catfr_categories(events,files):
    """
    This function makes the following assertions about an event structure:
    - That all presented words not in the practice list have been assigned a category
    - That all recalled words, apart from extra-list intrusions, have been assigned a category
    :param events:
    :return:
    """
    # np.recarray -> bool
    rec_events = events[(events.type=='REC_WORD') & (events.list>0) & (events.intrusion>-1)]
    word_events = events[(events.type=='WORD') & (events.list>0)]
    assert (word_events.category != 'X').all(), 'Some word presentations missing categories'
    assert (rec_events.category != 'X').all() , 'Some recalled words missing categories'


@as_recarray
def test_session_length(events,files):
    """
    Asserts that there are no more than 26 lists in the event structure.
    Specifically, for a set of event types that should appear once per list,
    asserts that there are no more than 26 events of that type.
    :param events:
    :return:
    """
    listwise_event_types = ['REC_START','REC_END','TRIAL',] # list is incomplete
    for type_ in listwise_event_types:
        assert (events.type==type_).sum() <= 26 , 'Session contains more than 26 lists'

@as_recarray
def test_words_in_wordpool(events,files):
    """
    Asserts that all non-practice words are in the wordpool file
    :param events:
    :param files:
    :return:
    """
    words = events[(events.type=='WORD') & (events.list>0)].item_name
    wordpool_file = files.get('wordpool') or files.get('no_accent_wordpool')
    if wordpool_file is not None:
        with open(wordpool_file,'r') as wf:
            wordpool = [x.strip().split()[-1] for x in wf]
        assert np.in1d(words,wordpool).all() , 'Wordpool missing presented words'

@as_recarray
def test_serialpos_order(events,files):
    """
    Asserts that serial position increases uniformly across lists, always between 0 and 12
    :param events:
    :param files:
    :return:
    """
    words = pd.DataFrame.from_records([e for e in events[events.type=='WORD']],columns=events.dtype.names)
    assert ((words.groupby('list').serialpos.diff().dropna())==1).all(), 'Serial positions not increasing uniformly'
    assert (words['serialpos']<=12).all(), 'Serial Position > 12 found'
    assert (words['serialpos']>=0).all() , 'Negative serial position found'


@as_recarray
def test_words_per_list(events,files):
    """
    Asserts that each serialposition occurs once per list
    :param events:
    :return:
    """
    words = pd.DataFrame.from_records([e for e in events[events.type == 'WORD']], columns=events.dtype.names)
    assert (words.groupby('serialpos').apply(len) <= len(words.list.unique())).all(), 'Serial position repeated'
    assert (words.groupby('serialpos').apply(len) >= len(words.list.unique())).all() , 'List missing serial position'


@as_recarray
@with_time_field
def test_rec_word_position(events,files,time_field):
    """
    Asserts that all REC_WORD events are preceded by a REC_START event and followed by a REC_END event
    :param events:
    :return:
    """
    events = events.view(np.recarray)
    for lst in np.unique(events.list):
        rec_start = events[(events.list==lst) & (events.type=='REC_START')]
        rec_end = events[(events.list==lst) & (events.type=='REC_END')]
        rec_words = events[(events.list==lst) & (events.type=='REC_WORD')]
        if len(rec_start):
            assert (rec_words[time_field]>rec_start[time_field]).all(),'REC_WORD occurs before REC_START in list %s'%lst
        if len(rec_end):
            assert (rec_words[time_field] < rec_end[time_field]).all(), 'REC_WORD occurs after REC_END in list %s'%lst

@as_recarray
@with_time_field
def test_stim_on_position(events,files,time_field):
    """
    Asserts that all STIM_ON events are preceded by a TRIAL event
    :param events:
    :return:
    """
    logger.debug('Checking stim event locations')
    with open(files['experiment_config'][-1]) as config_file:
        config=json.load(config_file)
    stim_events = events[events.type=='STIM_ON']
    if len(stim_events):
        try:
            n_artifact_stims = config['experiment']['artifact_detection'][
                                   'artifact_detection_number_of_stims_per_channel'] * len(
                config['experiment']['experiment_specific_data']['stim_channels'])
        except KeyError:
            return

        trial_0 = events[(events.type=='TRIAL') | (events.type=='ENCODING_START')][0]
        n_early_stims = (stim_events[time_field]<=trial_0[time_field]).sum()
        assert n_early_stims<= n_artifact_stims, '%s unexpected stim events before experiment begins'%(n_early_stims-n_artifact_stims)

@as_recarray
def test_rec_bracket(events,files):
    events =events.view(np.recarray)
    for lst in np.unique(events.list):
        rec_start = events[(events.list==lst) & (events.type=='REC_START')]
        assert rec_start.any(), 'NO REC_START event for list %s'%lst
        rec_end = events[(events.list==lst) & (events.type=='REC_END')]
        assert rec_end.any(), 'No REC_END event for list %s'%lst


# def test_stim_on_position(events,files):
#     """
#     Asserts that all STIM_ON events are preceded by a TRIAL event
#     :param events:
#     :return:
#     """
#     stim_events = events[events.type=='STIM_ON']
#     trial_0 = events[events.type=='TRIAL'][0]
#     assert stim_events[time_field]>trial_0[time_field], ''
