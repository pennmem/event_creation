from .base_log_parser import BaseLogParser
import numpy as np
import pandas as pd
from . import dtypes
import json

class BaseElememLogParser(BaseLogParser):
    """
    Class for parsing Elemem / System4 event.log files. As a design choice, Elemem logs all messages
    received over network cable from Unity tasks (or other sources, in theory)

    This should obviate the need for alignment, as Elemem records these events in the mstime/unix time according
    to its own clock, which is the same clock as the EEG recording.
    """
    # Maximum length of stim params
    MAX_STIM_PARAMS = 1
    # basically initialize a base log parser but specify that the primary log is event.log
    def __init__(self, protocol, subject, montage, experiment, session, files, primary_log='event_log', include_stim_params=True):
        if primary_log not in files:
            primary_log = 'event_log'

        BaseLogParser.__init__(self, protocol, subject, montage, experiment, session, files, primary_log=primary_log,
                               allow_unparsed_events=True, include_stim_params=include_stim_params)
        self._phase = ''
        self._files = files
        self._trial = -999
        self._stim_list = True # base is non-behavioral stim, so always "stim list"
        self._experiment_config = self._set_experiment_config()
        self._include_stim_params = include_stim_params
        self._add_type_to_new_event(
            start=self.event_elemem_start,
            sham=self.event_sham,
            stimming=self.event_stimulation,
        )

    def _get_raw_event_type(self, event_json):
        return event_json['type'].lower()

    def parse(self):
        try:
            return super().parse()
        except Exception as exc:
            traceback.print_exc(exc)
            logger.warn('Encountered error in parsing %s session %s: \n %s: %s' % (self._subject, self._session,
                                                                                   str(type(exc)), exc.message))
            raise exc

    def _read_event_log(self, filename):
        """
        Read events from the event.log format consistent with unity json lines output
        (JSON strings separated by newline characters).

        :param str filename: The path to the session log you wish to parse.
        """
        # Read session log
        df = pd.read_json(filename, lines=True)
        # Create a list of dictionaries, where each dictionary is the information about one event
        events = [e.to_dict() for _, e in df.iterrows()]
        return events

    def _read_primary_log(self):
        if hasattr(self._primary_log, '__iter__'):
            self._primary_log = self._primary_log[0]
        evdata = self._read_event_log(self._primary_log)
        return evdata

    def _set_experiment_config(self):
        config_file = (self._files['experiment_config'][0] if isinstance(self._files['experiment_config'],list)
                       else self._files['experiment_config'])
        with open(config_file,'r') as ecf:
            self._experiment_config = json.load(ecf)

    def event_default(self, event_json):
        event = self._empty_event
        event.mstime = event_json['time']
        event.type = event_json['type']
        event.list = self._trial
        event.session = self._session
        event.phase = self._phase
        if self._include_stim_params:
            event.stim_list = self._stim_list
        return event

    def event_elemem_start(self, event_json):
        event = self.event_default(event_json)
        event.type = 'START'
        return event

    def event_sham(self, event_json):
        event = self.event_default(event_json)
        event.type = 'SHAM'
        return event

    def event_stimulation(self, evdata):
        event = self.event_default(evdata)
        event.type = "STIM"
        event.stim_params.amplitude = evdata["data"]["amplitude"]
        event.stim_params.stim_duration = evdata["data"]["duration"]
        event.stim_params.anode_number = evdata["data"]["electrode_neg"]
        event.stim_params.cathode_number = evdata["data"]["electrode_pos"]
        event.stim_params.anode_label = self._jacksheet[evdata["data"]["electrode_neg"]]
        event.stim_params.cathode_label = self._jacksheet[evdata["data"]["electrode_pos"]]
        event.stim_params.burst_freq = evdata["data"]["frequency"]
        event.stim_params._remove = False
        return event

# LC: EFRCourier Parser
class ElememEFRCourierParser(BaseElememLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        if experiemnt=='EFRCourierReadOnly':
            self._include_stim_params = False
        elif experiment=='EFRCourierOpenLoop':
            self._include_stim_params = True
        else:
            raise Exception(f"Necessity of stim fields unknown for experiment {experiment}")

        super().__init__(protocol, subject, montage, experiment, session, files, include_stim_params=self._include_stim_params)
        self._session = -999
        self._trial = 0
        self._serialpos = -999
        self.practice = True
        self.current_num = -999

	# LC: For EFRCourier, every list is stim list
        if self._include_stim_params:
            self._stim_list = True
        else:
            self._stim_list = False

        if("wordpool" in list(files.keys())):
            with open(files["wordpool"]) as f:
                self.wordpool = [line.rstrip() for line in f]
        else:
            raise Exception("wordpool not found in transferred files")


        self.phase= ''
        self.practice = False
        self.storeX = -999
        self.storeZ = -999
        self.presX = -999
        self.presZ = -999

        self.STORES = ['gym', 'pet store', 'barber shop', 'florist', 'craft shop', 'jewelry store', 'grocery store', 'music store', 'cafe', 'pharmacy', 'clothing store', 'pizzeria', 'dentist', 'toy store', 'hardware store', 'bakery', 'bike shop']

        self._add_fields(*dtypes.courier_fields)
        self._add_fields(*dtypes.efr_fields)

        self.add_type_to_new_event(
            VERSIONS=self.add_experiment_version,
            STORE_MAPPINGS=self.add_store_mappings,
            PLAYERTRANSFORM=self.add_player_transform,
            POINTING_BEGINS=self.add_pointing_begins,
            POINTING_FINISHED=self.add_pointing_finished,
            START_TOWN_LEARNING=self.event_town_learning_start,
            STOP_TOWN_LEARNING=self.event_town_learning_end,
            START_PRACTICE_DELIVERIES=self.event_practice_start,
            STOP_PRACTICE_DELIVERIES=self.event_practice_end,
            KEYPRESS=self.event_efr_mark,
            CONTINUOUS_POINTER=self.event_pointer_on,
            START_DELIVERIES=self.event_trial_start,
            STOP_DELIVERIES=self.event_trial_end,
            OBJECT_PRESENTATION_BEGINS=self.add_object_presentation_begins,

            OBJECT_RECALL_RECORDING_START=self.add_object_recall_recording_start,
            OBJECT_RECALL_RECORDING_STOP=self.add_object_recall_recording_stop,
            CUED_RECALL_RECORDING_START=self.add_cued_recall_recording_start,
            CUED_RECALL_RECORDING_STOP=self.add_cued_recall_recording_end,
            FINAL_STORE_RECALL_RECORDING_START=self.add_final_store_recall_recording_start,
            FINAL_STORE_RECALL_RECORDING_STOP=self.add_final_store_recall_recording_stop,
            FINAL_OBJECT_RECALL_RECORDING_START=self.add_final_object_recall_recording_start,
            FINAL_OBJECT_RECALL_RECORDING_STOP=self.add_final_object_recall_recording_stop,
            END_TEXT=self.event_sess_end 
            )
	
        self._add_type_to_modify_events(
            FINAL_OBJECT_RECALL_RECORDING_START=self.modify_free_recall,
            CUED_RECALL_RECORDING_START=self.modify_cued_rec,
            FINAL_STORE_RECALL_RECORDING_START=self.modify_store_recall,
            OBJECT_RECALL_RECORDING_START=self.modify_rec_start,
            )

    def _identify_intrusion(self, events, new_event):
        words = events[events['type'] == 'WORD']
        word_event = words[(words['item'] == new_event["item"])]
        
        if len(word_event) > 1:
            raise Exception("Repeat items not supported or expected. Please check your data.")

        elif len(word_event) == 0: #ELI
            new_event.intrusion  = -1

        elif word_event["trial"][0] == self._trial:
            new_event.intrusion = 0
            new_event.serialpos = words[words['item'] == new_event["item"]]['serialpos'][0]
            new_event.store  = words[words['item'] == new_event["item"]]['store'][0]
            new_event.storeX = words[words['item'] == new_event["item"]]['storeX'][0]
            new_event.storeZ = words[words['item'] == new_event["item"]]['storeZ'][0]

        elif word_event["trial"][0] >= 0: # PLI
            new_event.intrusion = self._trial - word_event["trial"][0]
            new_event.serialpos = words[words['item'] == new_event["item"]]['serialpos'][0]
            new_event.store  = words[words['item'] == new_event["item"]]['store'][0]
            new_event.storeX = words[words['item'] == new_event["item"]]['storeX'][0]
            new_event.storeZ = words[words['item'] == new_event["item"]]['storeZ'][0]
        else:
            raise Exception("Processing error, word event was not presented during experimental trial")

        return new_event

    ####################
    # Functions to add new events from a single line in the log
    ####################

    def add_experiment_version(self, evdata):
        # version numbers are either v4.x or v4.x.x depending on the era,
        # due to a lack of foresight. At this point, we only need to check
        # that the version is greater than 4.0
        minor_version = int(evdata["data"]["Experiment version"].split('.')[1])
        if minor_version == 0 \
           and self.subject.startswith('R'):
            self.old_syncs=True
        else:
            self.old_syncs=False
        return False

    def add_store_mappings(self, evdata):
        event = self.event_default(evdata)
        event.type = "store mappings"

        # Different versions have different store mappings
        try:
            event.mappings = {"from {k}".format(k=k): {"to store": evdata['data'][k], "storeX": evdata['data']["{} position X".format(k)] , "storeZ": evdata['data']["{} position Z".format(k)] } for k in self.STORES}
        except:
            try:
                event.mappings = {"from {k}".format(k=k): {"to store": evdata['data']['_'.join(k.split())], "storeX": evdata['data']["{} position X".format(k)] , "storeZ": evdata['data']["{} position Z".format(k)] } for k in self.STORES}
            except:
                print("mappings is null")
                event.mappings = {}
        return event 

    def add_player_transform(self, evdata):
        self.presX = evdata['data']['positionX']
        self.presZ = evdata['data']['positionZ']
        return False

    def add_pointing_begins(self, evdata):
        event = self.event_default(evdata)
        event.type = "pointing begins"
	
        event.store = evdata['data']['store']
        event.storeX = self.storeX
        event.storeZ = self.storeZ
        event.presX = self.presX
        event.presZ = self.presZ
        return event

    def add_pointing_finished(self, evdata):
        event = self.event_default(evdata)
        event.type = "pointing finished"
        event.correctPointingDirection = evdata['data']['correct direction (degrees)']
        event.submittedPointingDirection = evdata['data']['pointed direction (degrees)']
        event.storeX = self.storeX
        event.storeZ = self.storeZ
        event.presX = self.presX
        event.presZ = self.presZ
        return event

    def event_town_learning_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'TL_START'
        return event

    def event_town_learning_end(self, evdata):
        event = self.event_default(evdata)
        event.type = 'TL_END'
        return event

    def event_practice_start(self, evdata):
        self.practice = True
        self.trial = 0
        event = self.event_default(evdata)
        event.type = 'PRACTICE_DELIVERY_START'
        return event
    
    def event_practice_end(self, evdata):
        self.practice = True
        event = self.event_default(evdata)
        event.type = 'PRACTICE_DELIVERY_END'
        return event

    def event_efr_mark(self, evdata):
        event = self.event_default(evdata)
        event.type = 'EFR_MARK'
        event.efr_mark = evdata['data']['response']=='correct'
        return event

    def event_pointer_on(self, evdata):
        event = self.event_default(evdata)
        event.type = 'POINTER_ON'
        return event

    def event_trial_start(self, evdata):
        self.practice = False
        self._trial = evdata['data']['trial number']
        event = self.event_default(evdata)
        event.type = 'TRIAL_START'
        return event
    
    def event_trial_end(self, evdata):
        event = self.event_default(evdata)
        event.type = 'TRIAL_END'
        return event

    def add_object_presentation_begins(self, evdata):
        self._trial = evdata['data']['trial number']
        event = self.event_default(evdata)
        event.type = "WORD" if not self.practice else "PRACTICE_WORD"
        event.serialpos = evdata['data']["serial position"]

        event.store = '_'.join(evdata['data']['store name'].split(' '))
        event.intruded = 0
        event.recalled = 0
        event.finalrecalled = 0

        if isinstance(evdata['data']['store position'], six.string_types):
            evdata['data']['store position'] = [float(p) for p in evdata['data']['store position'][1:-1].split(',')]

        event.storeX = evdata['data']['store position'][0]
        self.storeX = event.storeX

        event.storeZ = evdata['data']['store position'][2]
        self.storeZ = event.storeZ

        event.presX = self.presX
        event.presZ = self.presZ

        event["item"] = evdata['data']['item name'].upper().rstrip('.1')

        return event

    def add_object_recall_recording_start(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_START"
        event.presX = self.presX
        event.presZ = self.presZ
        return event

    def add_object_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_STOP"
        return event

    def add_cued_recall_recording_start(self, evdata):
        event = self.event_default(evdata)

        if isinstance(evdata['data']['store position'], six.string_types):
            evdata['data']['store position'] = [float(p) for p in evdata['data']['store position'][1:-1].split(',')]

        event.storeX = evdata['data']['store position'][0]
        event.storeZ = evdata['data']['store position'][2]
        self.storeX = event.storeX
        self.storeZ = event.storeZ

        event.presX = self.presX
        event.presZ = self.presZ

        event.type = 'CUED_REC_CUE'
        event["item"] = evdata['data']['item'].rstrip('.1').upper()
        event.store = '_'.join(evdata['data']['store'].lower().split(' '))
        return event 

    def add_cued_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "CUED_REC_STOP"
        return event

    def add_final_store_recall_recording_start(self, evdata):
        event = self.event_default(evdata)
        event.type = "FSR_START"
        return event

    def add_final_store_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "FSR_STOP"
        return event

    def add_object_recall_recording_start(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_START"

        event.presX = self.presX
        event.presZ = self.presZ

        return event

    def add_final_object_recall_recording_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "FFR_STOP"
        return event

    def event_sess_end(self, evdata):
        event = self.event_default(evdata)
        event.type = 'SESS_END'
        return event

    ####################
    # Functions that modify the existing event structure
    # when a line in the log is encountered
    ####################

    def modify_rec_start(self, events):
            
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        if self.practice:
            # Practice parsing not implemented for Courier / NICLS
            return events
        else:
            ann_outputs = self._parse_ann_file(str(self._trial))
        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)

            # Create a new event for the recall
            evtype = 'REC_WORD_VV' if "<>" in new_event["item"] else 'REC_WORD'
            new_event.type = evtype
            new_event = self._identify_intrusion(events, new_event)

            if new_event.intrusion > 0:
                events.intruded[(events["type"] == 'WORD') & (events["item"] == new_event["item"])] = 1
            elif new_event.intrusion == 0:
                events.recalled[(events["type"] == 'WORD') & (events["item"] == new_event["item"])] = 1

            events = np.append(events, new_event).view(np.recarray) 

        return events

    def modify_cued_rec(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        store_name = rec_start_event.store

        try:
            ann_outputs = self._parse_ann_file(str(self._trial) + '-' + " ".join(store_name.lower().split('_')))

        except:
            ann_outputs = []
            print(("MISSING ANNOTATIONS FOR %s" % rec_start_event.store))

        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)
            # new_event = self._identify_intrusion(events, new_event)

            evtype = 'CUED_REC_WORD_VV' if "<>" in new_event["item"] else 'CUED_REC_WORD'
            new_event.type = evtype

            # TODO: this matches previous events, but without this CUED_REC events lack any annotation
            # if new_event.intrusion > 0:
            #     events[(events["type"] == 'WORD') & (events["item"] == new_event["item"])].intruded = 1
            # elif new_event.intrusion == 0:
            #     events[(events["type"] == 'WORD') & (events["item"] == new_event["item"])].recalled = 1

            new_event["intrusion"] = -999

            events = np.append(events, new_event).view(np.recarray) 

        return events

    def modify_store_recall(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        ann_outputs = self._parse_ann_file("store recall")

        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)
            new_event = self._identify_intrusion(events, new_event)

            evtype = 'SR_REC_WORD_VV' if "<>" in new_event["item"] else 'SR_REC_WORD'
            new_event.type = evtype
            new_event.trial = -999 # to match old events
            new_event.intrusion = 0 if new_event["item"] in ["_".join(s.upper().split()) for s in self.STORES] else -1

            events = np.append(events, new_event).view(np.recarray) 

        return events

    def modify_free_recall(self, events):
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime
        try:
            ann_outputs = self._parse_ann_file("final recall")
        except:
            ann_outputs = self._parse_ann_file("final free")

        words = events[events["type"] == 'WORD']

        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)

            # Create a new event for the recall
            evtype = 'FFR_REC_WORD_VV' if "<>" in new_event["item"] else 'FFR_REC_WORD'
            new_event.type = evtype
            new_event = self._identify_intrusion(events, new_event)
            new_event.trial = -999 # to match old events

            if new_event.intrusion >= 0:
                new_event.intrusion = 0
                events.finalrecalled[(events["type"] == "WORD") & (events["item"] == new_event["item"])] = 1
            
            events = np.append(events, new_event).view(np.recarray) 

        return events


class ElememRepFRParser(BaseElememLogParser):
    def __init__(self, protocol, subject, montage, experiment, session, files):
        if experiment=='RepFR2':
            self._include_stim_params = True
        elif experiment=='RepFR1':
            self._include_stim_params = False
        else:
            raise Exception(f"Necessity of stim fields unknown for experiment {experiment}")

        super().__init__(protocol, subject, montage, experiment, session, files,
                        include_stim_params=self._include_stim_params)
        self._session = -999
        self._trial = 0
        self._serialpos = -999
        self.practice = True
        self.current_num = -999
        if self._include_stim_params:
            self._stim_list = False

        if("wordpool" in list(files.keys())):
            with open(files["wordpool"]) as f:
                self.wordpool = [line.rstrip() for line in f]

        else:
            raise Exception("wordpool not found in transferred files")

        if protocol=='ltp':
            self._add_fields(*dtypes.ltp_fields)
            self.add_type_to_new_event(participant_break=self.event_break_start) # Mid session break
        self._add_fields(*dtypes.repFR_fields)
        self._add_type_to_new_event(
            session=self.event_sess_start,
            trial=self.event_trial_start,
            countdown=self.event_countdown,  # Pre-trial countdown video
            rest=self._event_skip, # skip orientation events
            recall=self.event_rec_start,  # Start of vocalization period
            trialend=self.event_rec_stop, # End of vocalization period
            word=self.event_word_on,  # Start of word presentation
            stimming=self.event_stimulation,
            exit=self.event_sess_end,
        )
        
        self._add_type_to_modify_events(
            recall=self.modify_recalls,
            session=self.modify_session,
        )

    ###############
    #
    # EVENT CREATION FUNCTIONS
    #
    ###############

    def event_sess_start(self, evdata):
        self._session = evdata['data']['session']
        event = self.event_default(evdata)
        event.type = 'SESS_START'
        return event
    
    # added SESS_START and SESS_END to satisfy lcf partitions
    def event_sess_end(self, evdata):
        # FIXME: no session # attribute in embedded data
        # like there is for session start above 
        event = self.event_default(evdata)
        event.type = 'SESS_END'
        return event

    def event_trial_start(self, evdata):
        if self._include_stim_params:
            self._stim_list = evdata['data']['stim']
        self._trial = evdata['data']['trial']
        event = self.event_default(evdata)
        self._phase = 'encoding'
        event.type = 'TRIAL_START'
        return event

    def event_countdown(self, evdata):
        event = self.event_default(evdata)
        event.type = 'COUNTDOWN'
        
        # reset serial position of presentation at start of list
        self._serialpos = 0

        return event
    
    def event_break_start(self, evdata):
        event = self.event_default(evdata)
        event.type = 'BREAK_START'
        return event

    def event_word_on(self, evdata):

        # Build event
        event = self.event_default(evdata)
        event.type = "WORD"
        
        event.item_name = evdata['data']['word'].strip()
        # FIXME: someone made a redundant "word stimulus" instead of 
        # "word stimulus info". This will make duplicate events. Not my (JR) fault. 
        event.serialpos = evdata['data']['serialpos']
        event["list"] = self._trial
        event.is_stim = evdata['data']['stim']
        
        #FIXME: RepFR wordpool has "word" as the first item. Is this ok? 
        event.item_num = self.wordpool.index(str(event.item_name)) + 1

        return event

    def event_word_off(self, evdata):
        event = self.event_default(evdata)
        event.type = "WORD_OFF"
        event.serialpos = self._serialpos
        event["list"] = self._trial


        # increment serial position after word is processed 
        # to order words correctly
        self._serialpos += 1

        return event


    def event_rec_start(self, evdata):
        self._phase = 'retrieval'
        event = self.event_default(evdata)
        event.type = "REC_START"
        event["list"] = self._trial

        return event

    def event_rec_stop(self, evdata):
        event = self.event_default(evdata)
        event.type = "REC_END"
        event["list"] = self._trial


        # increment list index at end of each list
        self._trial += 1

        return event

    ###############
    #
    # EVENT MODIFIER FUNCTIONS
    #
    ###############

    def modify_session(self, events):
        """
        Applies session number to all previous events.
        """
        events.session = self._session
        return events

    def modify_recalls(self, events):
        # Access the REC_START event for this recall period to determine the timestamp when the recording began
        rec_start_event = events[-1]
        rec_start_time = rec_start_event.mstime

        # Get list of recalls from the .ann file for the current list, formatted as (rectime, item_num, item_name)
        try:
            ann_outputs = self._parse_ann_file(str(self._trial))
        # final free recall is not names with integer 26, but internally is 
        # considered trial 26
        except:
            ann_outputs = self._parse_ann_file("ffr")
        
        for recall in ann_outputs:

            # Get the vocalized word from the annotation
            word = recall[-1]
            new_event = self._empty_event

            new_event["list"] = self._trial
            new_event.session = self._session
            new_event.rectime = int(round(float(recall[0])))
            new_event.mstime = rec_start_time + new_event.rectime
            new_event.msoffset = 20
            new_event.item_name = word
            new_event.item_num = int(recall[1])
            new_event.phase = 'retrieval'

            # Create a new event for the recall
            evtype = 'REC_WORD_VV' if word == '<>' else 'REC_WORD'
            new_event.type = evtype

            # If XLI
            if recall[1] == -1:
                new_event.intrusion = -1
            else:  # Correct recall or PLI or XLI from later list
                # Determines which list the recalled word was from (gives [] if from a future list)
                pres_mask = (events.item_num == new_event.item_num) & (events.type == 'WORD')
                pres_trial = np.unique(events["list"][pres_mask])


                # Correct recall or PLI
                if len(pres_trial) == 1:
                    # Determines how many lists back the recalled word was presented

                    num_lists_back = self._trial - pres_trial[0]
                    # Retrieve the recalled word's serial position in its list, as well as the distractor(s) used
                    # Correct recall if word is from the most recent list
                    if num_lists_back == 0:
                        new_event.intrusion = 0

                        # Retroactively log on the word pres event that the word was recalled
                        if not any(events.recalled[pres_mask]):
                            events.recalled[pres_mask] = True
                    else:
                        new_event.intrusion = num_lists_back
                        events.intruded[pres_mask] = num_lists_back

                else:  # XLI from later list
                    new_event.intrusion = -1

            # Append the new event
            events = np.append(events, new_event).view(np.recarray)
            events = self.modify_repeats(events)


        return events

    def modify_repeats(self, events):
        # FIXME: this could be a lot more readable
        repeat_mask = [True if ev.item_num in events[:i][events[:i]["type"] =="WORD"].item_num else False for i, ev in enumerate(events)]
        events.is_repeat[repeat_mask] = True

        for w in np.unique(events[events["type"] == "WORD"]["item_name"]):
            events.repeats[((events["type"] == "WORD") | (events["type"] == "REC_WORD")) & (events["item_name"] == w)] = len(events[(events["item_name"] == w) & (events["type"] == "WORD")])


        return events

    @staticmethod
    def find_presentation(item_num, events):
        events = events.view(np.recarray)
        return np.logical_and(events.item_num == item_num, events.type == 'WORD')
