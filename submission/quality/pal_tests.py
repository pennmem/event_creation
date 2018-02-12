import pandas as pd
import numpy as np

def test_session_length(events,files):
    """
    Asserts that there are no more than 26 lists in the event structure.
    Specifically, for a set of event types that should appear once per list,
    asserts that there are no more than 26 events of that type.
    :param events:
    :return:
    """
    listwise_event_types = ['TRIAL','INSTRUCT_START','INSTRUCT_END','ENCODING_START',
                            'RETRIEVAL_START','TEST_START'] # list is incomplete
    for type_ in listwise_event_types:
        assert (events.type==type_).sum() <= 26 , 'Session contains more than 26 lists'


def test_words_per_list(events,files):
    """
    Asserts that each serialposition occurs once per list
    :param events:
    :return:
    """

    for type_ in ['STUDY_PAIR','REC_EVENT']:
        for field in ['serialpos','probepos']:
            words = pd.DataFrame.from_records([e for e in events[events.type == type_]], columns=events.dtype.names)
            assert (words.groupby(field).apply(len) <= len(words.list.unique())).all(), '%s repeated for type %s'%(field,type_)
            assert (words.groupby(field).apply(len) >= len(words.list.unique())).all(), 'List missing %s for type %s'%(field,type_)