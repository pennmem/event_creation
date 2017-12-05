from .system3_log_parser import BaseSys3LogParser
from .fr_log_parser import FRSessionLogParser
from .base_log_parser import BaseLogParser, BaseSys3_1LogParser
from collections import defaultdict
import numpy as np
from ..quality.fr_tests import test_catfr_categories

def mark_beginning(suffix='START'):
    def with_beginning_marked(f):
        def new_f(parser, event_json):
            event = f(parser, event_json)
            try:
                if event_json['value']:
                    event.type = event.type + '_%s' % suffix
            except KeyError:
                pass
            return event
        return new_f
    return with_beginning_marked


def mark_end(suffix='END'):
    def with_beginning_marked(f):
        def new_f(parser, event_json):
            event = f(parser, event_json)
            try:
                if not event_json['value']:
                    event.type = event.type + '_%s' % suffix
            except KeyError:
                pass
            return event
        return new_f
    return with_beginning_marked


class FRSys3LogParser(FRSessionLogParser,BaseSys3_1LogParser):

    _STIM_FIELDS = BaseLogParser._STIM_FIELDS + (
        ('biomarker_value',-1,'float64'),
        ('id','','S64'),
        ('position','','S64')
    )

    _RECOG_FIELDS =  (
        ('recognized',-999,'int16'),
        ('rejected',-999,'int16'),
        ('recog_resp',-999,'int16'),
        ('recog_rt',-999,'int16'),
    )

    @staticmethod
    def persist_fields_during_stim(event):
        return FRSessionLogParser.persist_fields_during_stim(event)+('phase',)

    _ITEM_FIELD = 'word'
    _SERIAL_POS_FIELD = 'serialpos'
    _ONSET_FIELD = 'start_offset'
    _ID_FIELD = 'hashsum'
    _RESPONSE_FIELD = 'yes'

    def event_default(self, event_json):
        event = BaseSys3_1LogParser.event_default(self,event_json)
        event.list = self._list
        event.stim_list = self._stim_list
        return event

    def __init__(self, protocol, subject, montage, experiment, session, files,primary_log='session_log'):
        super(FRSys3LogParser,self).__init__(protocol, subject, montage, experiment, session, files,
                                        primary_log=primary_log,allow_unparsed_events=True)

        self._list = -999
        self._stim_list = False
        self._on = False
        self._recognition = False
        self._was_recognized = False
        self._phase = ''
        self._add_fields(*self._RECOG_FIELDS)

        self._type_to_new_event = defaultdict(
            lambda: self.event_default,
            WORD_START=self.event_word,
            WORD_END=self.event_word_off,
            TRIAL_START=self.event_trial,
            ENCODING_END=self.event_reset_serialpos,
            RETRIEVAL_START=self.event_recall_start,
            RETRIEVAL_END=self.event_recall_end,
            RECOGNITION_START=self._begin_recognition,
            KEYPRESS=self.event_recog,
        )
        self._add_type_to_modify_events(
            RETRIEVAL_START=self.modify_recalls,
            KEYPRESS=self.modify_recog,
        )

    def modify_recog(self, events):
        events = events.view(np.recarray)
        recog_off_event = events[-1]
        recog_word = recog_off_event.item_name
        word_mask = events.item_name == recog_word
        recog_event = events[word_mask & ((events.type == 'RECOG_TARGET') | (events.type == 'RECOG_LURE'))][0]
        rejected = not self._was_recognized if recog_event.type =='RECOG_LURE' else -999
        recognized = self._was_recognized if recog_event.type == 'RECOG_TARGET' else -999
        new_events = events[word_mask]
        new_events.recog_resp = self._was_recognized
        new_events.rejected = rejected
        new_events.recognized = recognized
        new_events.recog_rt = self._recog_endtime
        if (new_events.type == 'WORD').any():
            persist_fields = self.persist_fields_during_stim(new_events[0])
            for field in persist_fields:
                new_events[field] = new_events[new_events.type == 'WORD'][field].copy()
        events[word_mask] = new_events
        return events

    def modify_recalls(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        ann_outputs = self._parse_ann_file(str(self._list) if self._list > 0 else '0')
        for recall in ann_outputs:
            word = recall[-1]

            new_event = self._empty_event
            new_event.list = self._list
            new_event.stim_list = self._stim_list
            new_event.exp_version = self._version
            new_event.rectime = float(recall[0])
            new_event.mstime = rec_start_time + new_event.rectime
            new_event.msoffset = 20
            new_event.item_name = word
            new_event.item_num = recall[1]
            new_event.phase = self._phase

            # If vocalization
            if word == '<>' or word == 'V' or word == '!':
                new_event.type = 'REC_WORD_VV'
            else:
                new_event.type = 'REC_WORD'

            pres_mask = self.find_presentation(word, events)
            pres_list = np.unique(events[pres_mask].list)
            pres_mask = np.logical_and(pres_mask, events.list == self._list)

            # Correct recall or PLI
            if len(pres_list) >= 1:
                new_event.intrusion = self._list - max(pres_list)
                if new_event.intrusion == 0:
                    new_event.serialpos = np.unique(events[pres_mask].serialpos)
                    new_event.recalled = True
                    if not any(events.recalled[pres_mask]):
                        events.recalled[pres_mask] = True
                        events.rectime[pres_mask] = new_event.rectime
            else:  # XLI
                new_event.intrusion = -1

            events = np.append(events, new_event).view(np.recarray)

        return events

    def event_recog(self, event_json):
        self._was_recognized = event_json[self._RESPONSE_FIELD]
        return False

    def _begin_recognition(self, event_json):
        self._recognition = True
        return self.event_default(event_json)

    def event_reset_serialpos(self, split_line):
        return super(FRSys3LogParser, self).event_reset_serialpos(split_line)

    def event_recall_start(self, event_json):
        event = self.event_default(event_json)
        event.type = 'REC_START'
        return  event

    def event_recall_end(self, event_json):
        event = self.event_default(event_json)
        event.type = 'REC_END'
        return event

    def event_trial(self, event_json):
        list = event_json['listno']
        if list == 0:
            self._list = -1
        else:
            self._list = list
        self._stim_list = event_json[self._PHASE_TYPE_FIELD] in ['STIM', 'PS']
        self._phase = event_json[self._PHASE_TYPE_FIELD]
        event = self.event_default(event_json)
        return event

    def event_word(self, event_json):
        event = self.event_default(event_json)
        event.serialpos = event_json[self._SERIAL_POS_FIELD] + 1
        self._word = event_json[self._ITEM_FIELD]
        event = self.apply_word(event)
        if self._recognition:
            self._recog_pres_mstime = event_json[self._MSTIME_FIELD]
            if event_json[self._PHASE_TYPE_FIELD] == 'LURE':
                event.type = 'RECOG_LURE'
                event.stim_list = -999
            else:
                event.type = 'RECOG_TARGET'
        else:
            event.type = 'WORD'
        return event

    def event_word_off(self, event_json):
        event = self.event_default(event_json)
        event.serialpos = event_json[self._SERIAL_POS_FIELD] + 1
        self._word = event_json[self._ITEM_FIELD]
        event = self.apply_word(event)
        if self._recognition:
            event.type = 'RECOG_WORD_OFF'
            self._recog_endtime = event_json[self._MSTIME_FIELD] - self._recog_pres_mstime
        else:
            event.type = 'WORD_OFF'
        return event


class catFRSys3LogParser(FRSys3LogParser):
    _BASE_FIELDS = FRSys3LogParser._BASE_FIELDS + (
        ('category','X','S64'),
        ('category_num',-999,'int16')
    )

    _CATEGORY = 'category'
    _CATEGORY_NUM = 'category_num'

    def __init__(self, *args, **kwargs):
        super(catFRSys3LogParser, self).__init__(*args, **kwargs)

    def event_word(self, event_json):
        event = super(catFRSys3LogParser, self).event_word(event_json)
        event.category = event_json[self._CATEGORY]
        event.category_num = event_json[self._CATEGORY_NUM] if not(np.isnan(event_json[self._CATEGORY_NUM])) else -999
        return event

    def event_word_off(self, event_json):
        event = super(catFRSys3LogParser, self).event_word_off(event_json)
        event.category = event_json[self._CATEGORY]
        event.category_num = event_json[self._CATEGORY_NUM]
        return event

    def clean_events(self, events):
        """
        Final processing of events
        Here we add categories and category numbers to all the non-ELI recalls
        In theory this could be added to modify_recalls, but we might as well wait until we've finished
        and then do it all at once.
        :param events:
        :return:
        """
        events= super(catFRSys3LogParser, self).clean_events(events).view(np.recarray)
        is_recall = (events.type=='REC_WORD') & (events.intrusion != -1)
        rec_events = events[is_recall]
        categories = [events[(events.type=='WORD') & (events.item_name==r.item_name)].category[0] for r in rec_events]
        category_nums = [events[(events.type=='WORD') & (events.item_name == r.item_name)].category_num[0] for r in rec_events]
        rec_events['category']=categories
        rec_events['category_num']=category_nums
        events[is_recall] = rec_events
        return events

    @staticmethod
    def check_event_quality(events,files):
        FRSys3LogParser.check_event_quality(events,files)
        test_catfr_categories(events,files)


class RecognitionParser(BaseSys3LogParser):
    # Lures have phase-type "LURE"
    def __init__(self, *args, **kwargs):
        super(RecognitionParser, self).__init__(*args, **kwargs)

        self._type_to_new_event = defaultdict(lambda : self._event_skip)
        self._add_type_to_new_event(
            WORD = self.event_recog,
            RECOGNITION = self.begin_recognition
        )
        self._recognition = False

    def begin_recognition(self, event):
        self._recognition = True

    def event_recog(self, event_json):
        if self._recognition:
            event = self.event_default(event_json)
            event.type = 'RECOG_LURE' if event_json['phase_type']=='LURE' else 'RECOG_TARGET'
            event.response = event_json['value']






if __name__ == '__main__':
    files = {
        'session_log': '/Users/leond/Documents/R1111M/behavioral/FR5/session_0/session.log',
        'wordpool': '/Users/leond/Documents/R1111M/behavioral/FR5/RAM_wordpool.txt',
        'session_sql': '/Users/leond/Documents/R1111M/behavioral/FR5/session_0/session.sqlite'
    }

    frslp = FRSys3LogParser('r1', 'R1999X', 0.0, 'FR1', 0, files)
    events = frslp.parse()
