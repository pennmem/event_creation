import numpy as np
from . import dtypes
from .base_log_parser import BaseUnityLTPLogParser


class PrelimSessionLogParser(BaseUnityLTPLogParser):

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super(PrelimSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files)
        self._trial = -999
        self._serialpos = -999
        self.current_word = ''
        self.current_num = -999

        self._add_fields(*dtypes.vffr_fields)
        self._add_type_to_new_event(
            countdown=self.event_countdown,  # Pre-trial countdowns
            end_message=self.event_sess_end,  # End of session
            microphone_test_start=self.event_sess_start,  # Start of session (microphone test)
            recall_start=self.event_rec_start,  # Start of vocalization period
            recall_stop=self.event_rec_stop,  # End of vocalization period
            required_break_start=self.event_break_start,  # Start of mid-session break
            required_break_stop=self.event_break_stop,  # End of mid-session break
            stimulus=self.event_word_on,  # Start of word presentation
            stimulus_cleared=self.event_word_off,  # End of word presentation
        )
        self._add_type_to_modify_events(
            recall_start=self.modify_recalls,  # Parse annotations for the current trial and add them as recall events
            stimulus_cleared=self.modify_pres  # Mark word presentation events with how long the word was onscreen
        )

    ###############
    #
    # EVENT CREATION FUNCTIONS
    #
    ###############

    def update_trial(self, evdata):
        """
        At the start of each trial, update the trial number and reset the serial position.
        :param evdata: A dictionary containing the data from an event.
        :return: None (does not create an event)
        """
        self._trial = int(evdata['data']['displayed_text'][5:]) - 1
        self._serialpos = 0
        return None

    def event_countdown(self, evdata):
        """
        Creates an event for a single number of the pre-trial countdown.
        :param evdata: A dictionary containing the data from an event.
        :return: A COUNTDOWN event.
        """
        event = self.event_default(evdata)
        event.type = 'COUNTDOWN'
        event.item_name = evdata['data']['displayed text']
        return event

    def event_word_on(self, evdata):
        """
        Creates an event for the onset of a word presentation.
        :param evdata: A dictionary containing the data from an event.
        :return: A WORD event.
        """
        # Read the word that was presented
        self.current_word = evdata['data']['word'].strip()
        # Increment serial position
        self._serialpos += + 1

        # Build event
        event = self.event_default(evdata)
        event.type = 'WORD'
        event.item_name = self.current_word
        event.serialpos = self._serialpos
        if 'ltp word number' in evdata['data']:
            self.current_num = evdata['data']['ltp word number']
            event.item_num = self.current_num
        return event

    def event_word_off(self, evdata):
        """
        Creates an event for the offset of a word presentation.
        :param evdata: A dictionary containing the data from an event.
        :return: A WORD_OFF event.
        """
        event = self.event_default(evdata)
        event.type = 'WORD_OFF'
        event.item_name = evdata['data']['word'].strip()
        event.serialpos = self._serialpos
        if event.item_name == self.current_word:
            event.item_num = self.current_num
        return event

    def event_rec_start(self, evdata):
        """
        Creates an event for the start of a recall period.
        :param evdata: A dictionary containing the data from an event.
        :return: A REC_START event.
        """
        event = self.event_default(evdata)
        event.type = 'REC_START'
        event.item_name = self.current_word
        event.item_num = self.current_num
        event.serialpos = self._serialpos
        return event

    def event_rec_stop(self, evdata):
        """
        Creates an event for the end of a recall period.
        :param evdata: A dictionary containing the data from an event.
        :return: A REC_STOP event.
        """
        event = self.event_default(evdata)
        event.type = 'REC_STOP'
        event.item_name = self.current_word
        event.item_num = self.current_num
        event.serialpos = self._serialpos
        return event

    def event_break_start(self, evdata):
        """
        Creates an event for the start of a mid-session break.
        :param evdata: A dictionary containing the data from an event.
        :return: A BREAK_START event.
        """
        event = self.event_default(evdata)
        event.type = 'BREAK_START'
        return event

    def event_break_stop(self, evdata):
        """
        Creates an event for the end of a mid-session break.
        :param evdata: A dictionary containing the data from an event.
        :return: A BREAK_STOP event.
        """
        event = self.event_default(evdata)
        event.type = 'BREAK_STOP'
        return event

    def event_sess_start(self, evdata):
        """
        Creates an event to mark the start of a session.
        :param evdata: A dictionary containing the data from an event.
        :return: A SESS_START event.
        """
        event = self.event_default(evdata)
        event.type = 'SESS_START'
        return event

    def event_sess_end(self, evdata):
        """
        Creates an event to mark the end of a session.
        :param evdata: A dictionary containing the data from an event.
        :return: A SESS_END event.
        """
        event = self.event_default(evdata)
        event.type = 'SESS_END'
        return event

    ###############
    #
    # EVENT MODIFIER FUNCTIONS
    #
    ###############

    def modify_pres(self, events):
        """
        Determine for how long the most recently presented word was onscreen, then add the presentation duration to the
        onset and offset events.
        :param events: The recarray of all events in the session.
        :return: The modified recarray of events.
        """
        pres_dur = events[-1].mstime - events[-2].mstime
        events[-1].pres_dur = pres_dur
        events[-2].pres_dur = pres_dur
        return events

    def modify_recalls(self, events):
        """
        Parse the annotation file for the current trial's recalls and append REC_WORD events for each recalled word.
        :param events: The recarray of all events in the session.
        :return: The modified recarray of events.
        """
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
            new_event.item_name = word
            new_event.item_num = int(recall[1])
            new_event.type = 'REC_WORD_VV' if word == '<>' or word == 'V' or word == '!' else 'REC_WORD'

            # If XLI
            if recall[1] == -1:
                new_event.intrusion = -1
            else:  # Correct recall, PLI, or XLI word that appears in a later list
                pres_mask = self.find_presentation(new_event.item_num, events)
                pres_trial = np.unique(events[pres_mask].trial)

                # Correct recall or PLI
                if len(pres_trial) == 1:
                    # Determines how many lists back the recalled word was presented
                    new_event.intrusion = self._trial - pres_trial[0]
                    # Retrieve the recalled word's serial position in its list
                    new_event.serialpos = np.unique(events[pres_mask].serialpos)
                    # Correct recall if word is from the most recent list
                    if new_event.intrusion == 0:
                        # Retroactively log on the word pres event that the word was recalled
                        if not any(events[pres_mask].recalled):
                            events.recalled[pres_mask] = True
                    else:
                        # Mark on the presentation event that the word later intruded as a PLI
                        events.intruded[pres_mask] = new_event.intrusion
                elif len(pres_trial) == 0:  # XLI from later list
                    new_event.intrusion = -1
                else:
                    raise ValueError('The word "%s" was presented on multiple trials!' % word)

            # Add recall event to events array
            events = np.append(events, new_event).view(np.recarray)

        return events

    @staticmethod
    def find_presentation(item_num, events):
        """
        Identifies the presentation event for a specified item.
        :param item_num: The integer ID number of the recalled word.
        :param events: The recarray of all events in the session.
        :return: A boolean array marking the presentation event for the input item.
        """
        events = events.view(np.recarray)
        pres_mask = np.logical_and(events.item_num == item_num, events.type == 'WORD')
        return pres_mask
