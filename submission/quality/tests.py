import numpy as np
import pandas as pd


"""
Data quality checks for events.
"""



def test_catfr_categories(events,files=None):
    """
    This function makes the following assertions about an event structure:
    - That all presented words not in the practice list have been assigned a category
    - That all recalled words, apart from extra-list intrusions, have been assigned a category
    :param events:
    :return:
    """
    # np.recarray -> bool
    rec_events = events[events.type=='REC_WORD']
    word_events = events[events.type=='WORD']
    assert (word_events[word_events.list>=0].category != 'X').all()
    assert (rec_events[rec_events.intrusion>-1].category != 'X').all()


def test_session_length(events,files=None):
    """
    Asserts that there are no more than 26 lists in the event structure.
    Specifically, for a set of event types that should appear once per list,
    asserts that there are no more than 26 events of that type.
    :param events:
    :return:
    """
    listwise_event_types = ['REC_START','REC_END','TRIAL','INSTRUCT_START','INSTRUCT_END',] # list is incomplete
    for type_ in listwise_event_types:
        assert (events.type==type_).sum() < 26


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
        assert np.in1d(words,wordpool).all()


def test_serialpos_order(events,files=None):
    """
    Asserts that serial position increases uniformly across lists, always between 0 and 12
    :param events:
    :param files:
    :return:
    """
    words = pd.DataFrame([e for e in events[events.type=='WORD']],columns=events.dtype.names)
    assert ((words.groupby('list').serialpos.diff().dropna())==1).all()
    assert (words['serialpos']<=12).all()
    assert (words['serialpos']>=0).all()

