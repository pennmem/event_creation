import numpy as np
import pandas as pd


"""
Data quality checks for events.
"""



def test_catfr_categories(events):
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


def test_session_length(events):
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


def test_serialpos_order(events):
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

def test_words_per_list(events):
    """
    Asserts that each serialposition occurs once per list
    :param events:
    :return:
    """
    words = pd.DataFrame.from_records([e for e in events[events.type == 'WORD']], columns=events.dtype.names)
    assert words.groupby('serialpos').apply(len) <= len(words.list.unique()), 'Serial position repeated'
    assert words.groupby('serialpos').apply(len) >= len(words.list.unique()), 'List missing serial position'