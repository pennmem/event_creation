from .base_log_parser import BaseUnityLTPLogParser
import numpy as np

class VFFRSessionLogParser(BaseUnityLTPLogParser):

    @classmethod
    def _vffr_fields(cls):
        """
        Returns the templates for all LTPFR-specific fields (those not included under _BASE_FIELDS)
        :return:
        """
        return (
            ('trial', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('item_name', '', 'S16'),
            ('recalled', False, 'b1'),
            ('rectime', -999, 'int32'),
            ('intrusion', -999, 'int16'),
            ('eogArtifact', -1, 'int8')
        )

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(VFFRSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files)
        self._trial = 0
        self._serialpos = 0

        self._add_fields(*self._vffr_fields())
        self._add_type_to_new_event(
            end_message=self._event_skip,
            final_recall_prompt=self._event_skip,
            final_recall_start=self.event_ffr_start,  # Start of final free recall
            final_recall_stop=self.event_ffr_stop,  # End of final free recall
            microphone_test_begin=self._event_skip,
            microphone_test_confirmation=self._event_skip,
            microphone_test_end=self._event_skip,
            microphone_test_playing=self._event_skip,
            microphone_test_recording=self._event_skip,
            press_any_key_prompt=self._event_skip,
            recall_prompt=self._event_skip,
            recall_start=self.event_rec_start,  # Start of vocalization period
            recall_stop=self.event_rec_stop,  # End of vocalization period
            required_break_start=self.event_break_start,  # Start of break
            required_break_stop=self.event_break_stop,  # End of break
            restore_original_text_color=self._event_skip,
            stimulus=self._event_skip,
            stimulus_cleared=self.event_word_off,  # End of word presentation
            stimulus_display=self.event_word_on,  # Start of word presentation
            text_color_changed=self._event_skip,
            text_display_cleared=self._event_skip,
        )
        self._add_type_to_modify_events(
            final_recall_start=self.modify_ffr,  # Parse FFR annotations and add recall information to events
            recall_start=self.modify_voc  # Parse annotation for word vocalization and add a vocalization event
        )

    ###############
    #
    # EVENT CREATION FUNCTIONS
    #
    ###############

    def event_word_on(self, evdata):
        self._serialpos += 1
        event = self.event_default(evdata)
        event.type = 'WORD'
        event.item_name = evdata['displayed text']
        event.serialpos = self._serialpos
        return event

    def event_word_off(self, evdata):
        event = self.event_default(evdata)
        event.type = 'WORD_OFF'
        event.item_name = evdata['word']
        event.serialpos = self._serialpos
        return event

    def event_rec_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'REC_START'
        event.serialpos = self._serialpos
        return event

    def event_rec_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'REC_STOP'
        event.serialpos = self._serialpos
        return event

    def event_ffr_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'FFR_START'
        return event

    def event_ffr_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'FFR_STOP'
        return event

    def event_break_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_START'
        return event

    def event_break_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_STOP'
        return event

    ###############
    #
    # EVENT MODIFIER FUNCTIONS
    #
    ###############
    def modify_voc(self, events):
        return events

    def modify_ffr(self, events):
        return events

    def modify_recalls(self, events):
        if self._distractor != 0:
            events = self.modify_final_distractor_info(self._trial, events).view(np.recarray)

        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        # Get list of recalls from the .ann file for the current list; each recall is (rectime, item_num, item_name)
        ann_outputs = self._parse_ann_file(str(self._trial - 1))

        for recall in ann_outputs:
            word = recall[-1]
            new_event = self._empty_event
            new_event.trial = self._trial
            new_event.session = self._session
            new_event.rectime = int(round(float(recall[0])))
            new_event.mstime = rec_start_time + new_event.rectime
            new_event.msoffset = 20
            new_event.item_name = word
            new_event.item_num = int(recall[1])
            # Vocalizations are skipped in the .mat files for ltpFR2
            if word == '<>' or word == 'V' or word == '!':
                new_event.type = 'REC_WORD_VV'
            else:
                new_event.type = 'REC_WORD'

            # If XLI
            if recall[1] == -1:
                new_event.intrusion = -1
            else:  # Correct recall or PLI or XLI from later list
                # Determines which list the recalled word was from (gives [] if from a future list)
                pres_mask = self.find_presentation(new_event.item_num, events)
                pres_trial = np.unique(events[pres_mask].trial)

                # Correct recall or PLI
                if len(pres_trial) == 1:
                    # Determines how many lists back the recalled word was presented
                    new_event.intrusion = self._trial - pres_trial[0]
                    # Retrieve the recalled word's serial position in its list, as well as the distractor(s) used
                    new_event.serialpos = np.unique(events[pres_mask].serialpos)
                    new_event.begin_distractor = np.unique(events[pres_mask].begin_distractor)
                    new_event.final_distractor = np.unique(events[pres_mask].final_distractor)
                    # Correct recall if word is from the most recent list
                    if new_event.intrusion == 0:
                        # Retroactively log on the word pres event that the word was recalled
                        if not any(events[pres_mask].recalled):
                            events.recalled[pres_mask] = True
                    else:
                        events.intruded[pres_mask] = new_event.intrusion
                else:  # XLI from later list
                    new_event.intrusion = -1

            # Add recall event to events array
            events = np.append(events, new_event).view(np.recarray)

        return events

    @staticmethod
    def find_presentation(item_num, events):
        events = events.view(np.recarray)
        return np.logical_and(events.item_num == item_num, events.type == 'WORD')
