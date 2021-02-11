from .base_log_parser import BaseSys3_1LogParser
from .pal_log_parser import PALSessionLogParser
from collections import defaultdict
import warnings
import numpy as np


class PALSys3LogParser(PALSessionLogParser,BaseSys3_1LogParser):

    def __init__(self,protocol, subject, montage, experiment, session, files):

        super(PALSys3LogParser,self).__init__(protocol, subject, montage, experiment, session, files)
        self._allow_unparsed_events = True

        self._type_to_new_event = defaultdict(lambda :self.event_default,
                                              STUDY_PAIR_START=self.event_study_pair,
                                              STUDY_PAIR_END  = self.event_study_pair,
                                              PROBE_START = self.event_test_probe,
                                              PROBE_END = self.event_test_probe,
                                              REC_START = self.event_recall_start,
                                              REC_END = self.event_test_probe,
                                              TRIAL = self.event_trial,
                                              VOCALIZATION_START = self.event_vocalization,
                                              VOCALIZATION_END = self.event_vocalization,
                                              DISTRACT_START = self.event_distract_start,
                                              DISTRACT_END = self.event_distract_end,
                                              Logging = self._event_skip,
                                              SESSION_SKIPPED = self._event_skip,
                                              )

        self._type_to_modify_events = {
            'PROBE_START': self.modify_recalls,
            'REC_END': self.modify_test,
        }

        self._study_1 = None


        self._list = -999
        self._phase = ''
        self._probepos = -999

    @property
    def _stim_list(self):
        return self._phase in ['PS','STIM']

    @_stim_list.setter
    def _stim_list(self,val):
        pass

    @staticmethod
    def persist_fields_during_stim(event):
        fields = PALSessionLogParser.persist_fields_during_stim(event)
        if fields:
            fields += ('phase',)
        return fields

    ### NEW EVENT METHODS ###

    def event_default(self, event_json):
        event = BaseSys3_1LogParser.event_default(self,event_json)
        event.list = self._list
        event.stim_list  = self._stim_list
        return event


    def event_vocalization(self,event_json):
        event = self.event_default(event_json)
        event.mstime = int(event_json[self._MSTIME_FIELD])
        return event

    def event_recall_start(self,event_json):
        if self._probepos == -999:
            self._probepos = 1
        else:
            self._probepos +=1
        return self.event_test_probe(event_json)

    def event_study_pair(self, event_json):
        event=self.event_default(event_json)
        event.study_1=event_json['word1']
        event.study_2=event_json['word2']
        event.serialpos=event_json['serialpos']+1
        event.type='STUDY_PAIR_OFF' if str(event.type).endswith('END') else 'STUDY_PAIR'
        return event


    def event_test_probe(self, split_line):

        event = self.event_default(split_line)
        event.probe_word = split_line['probe']
        event.expecting_word = split_line['expecting']
        event.cue_direction  = split_line['direction']
        event.study_1 = event.expecting_word if split_line['direction'] else event.probe_word
        event.study_2  = event.probe_word if split_line['direction'] else event.expecting_word
        event.probepos = self._probepos
        event.serialpos = split_line['serialpos']+1
        self._serialpos = split_line['serialpos']+1
        self._probe_word = event.probe_word
        self._expecting_word = event.expecting_word
        self._cue_direction = event.cue_direction
        return event

    def event_trial(self,event_json):
        lst = event_json['listno']
        self._list = lst if lst >0 else -1
        self._phase = event_json[self._PHASE_TYPE_FIELD]
        self._probepos = -999
        return self.event_default(event_json)

    ### MODIFY EVENT METHODS ###



    def modify_test(self, events):
        this_pair = ((events.serialpos==self._serialpos) | (events.probepos==self._probepos)) & (events.list==self._list)
        new_events = events[this_pair]
        new_events['serialpos'] = self._serialpos
        new_events['probepos'] = self._probepos

        if self._study_1:
            new_events['study_1'] = self._study_1
            new_events['study_2'] = self._study_2
        events[this_pair] = new_events
        return events

    def modify_recalls(self, events):
        rec_start_event = events[-1]
        rec_start_time = float(rec_start_event.mstime)
        list_str = str(self._list) if self._list>0 else '0'
        ann_file = list_str + '_' + str(self._probepos-1)
        try:
            ann_outputs = self._parse_ann_file(ann_file)
        except IOError: # Will happen if no ann file
            warnings.warn("Ann file %s not parseable" % ann_file)
            return events
        response_spoken = False

        modify_events_mask = np.logical_and.reduce((events.serialpos == self._serialpos,
                                                    events.list == self._list,
                                                    events.type != 'REC_EVENT'))

        events.resp_pass[modify_events_mask] = 0
        events.correct[modify_events_mask] = 0

        for i, recall in enumerate(ann_outputs):
            word = recall[-1]
            new_event = self._empty_event
            new_event.list = self._list
            new_event.serialpos = self._serialpos
            new_event.probepos = self._probepos
            new_event.probe_word = self._probe_word
            new_event.expecting_word = self._expecting_word
            new_event.cue_direction = self._cue_direction
            if self._cue_direction == 0:
                self._study_1 = self._probe_word
                self._study_2 = self._expecting_word
            elif self._cue_direction == 1:
                self._study_1 = self._expecting_word
                self._study_2 = self._probe_word
            new_event.study_1 = self._study_1
            new_event.study_2 = self._study_2
            new_event.resp_word = word
            new_event.stim_list = self._stim_list
            new_event.RT = recall[0]
            new_event.mstime = rec_start_time + recall[0]
            new_event.msoffset = 20
            self._correct = 1 if self._expecting_word==word and self._probe_word != word else 0
            new_event.correct = self._correct
            new_event.resp_pass = 0
            new_event.intrusion = 0
            new_event.vocalization = 0

            modify_events_mask = ((events.serialpos == self._serialpos)
                                  & (events.list == self._list)
                                  & (events.type != 'REC_EVENT'))

            pres_mask = self.find_presentation(word, events)
            pres_list = np.unique(events[pres_mask].list)

            same_word = i > 0 and ann_outputs[i-1][-1] == word

            if word != '<>' and word != 'v' and word != '!':
                is_vocalization = False
                events.vocalization[modify_events_mask] = 0
                events.resp_word[modify_events_mask] = word
                events.correct[modify_events_mask] = self._correct
                events.resp_pass[modify_events_mask] = 0
                if not same_word:
                    events.RT[modify_events_mask] = new_event.RT

                response_spoken = True
            else:
                if not response_spoken:
                    events.correct[modify_events_mask] = 0
                is_vocalization = True

            new_event.type = 'REC_EVENT'

            if word == 'PASS':
                new_event.resp_pass = 1
                events.resp_pass[modify_events_mask] = 1
            elif not is_vocalization:
                events.resp_pass[modify_events_mask] = 0
            # if xli

            if self._correct == 0:
                if word == '<>' or word == 'v' or word == '!':
                    new_event.vocalization = 1
                    new_event.intrusion = 0
                    new_event.resp_pass = 0
                    if not response_spoken or new_event.resp_pass:
                        events.resp_pass[modify_events_mask] = 0
                elif word == 'PASS':
                    new_event.resp_pass = 1
                    new_event.intrusion = 0
                    events.resp_pass[modify_events_mask] = 1
                    events.intrusion[modify_events_mask] = 0
                elif recall[1] == -1:
                    new_event.intrusion = -1
                    events.intrusion[modify_events_mask] = -1
                else:  # correct recall or pli or xli from latter list
                    # correct recall or pli
                    if len(pres_list) == 1:
                        new_event.intrusion = self._list - pres_list
                        events.intrusion[modify_events_mask] = self._list - pres_list
                    else:  # xli
                        new_event.intrusion = -1
                        events.intrusion[modify_events_mask] = -1

            # No PAL2 experiments on system 3
            # if self._pal2_stim_serialpos == new_event.serialpos and \
            #         self._pal2_stim_list == new_event.list and \
            #         self._pal2_stim_is_retrieval:
            #     new_event.is_stim = True
            #     stim_on = new_event.mstime - self._pal2_stim_on_time < self.PAL2_STIM_DURATION
            #     self.set_event_stim_params(new_event, jacksheet=self._jacksheet, stim_on=stim_on, **self._pal2_stim_params)

            events = np.append(events, new_event).view(np.recarray)
        return events



if __name__ == '__main__':
    files = {
        'session_log':['/Users/leond/PAL5/R1293P/session_0/session.sqlite'],
        'annotations':['/Users/leond/PAL5/R1293P/session_0/0_0.ann']
    }
    parser = PALSys3LogParser('r1','R9999M',0.0,'PAL1',0,files)
    events =  parser.parse()
    events = parser.clean_events(events)
    pass
