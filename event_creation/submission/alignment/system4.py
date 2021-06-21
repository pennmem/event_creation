import os
import mne
import glob
import numpy as np
import pandas as pd
import json
from ..log import logger

class System4Offset:
    def __init__(self, events, files, eeg_dir):
        eeg_sources = json.load(open(files['eeg_sources']))
        if len(eeg_sources) != 1:
            raise AlignmentError('Cannot align EEG with %d sources' % len(eeg_sources))
        self.eeg_file_stem = list(eeg_sources.keys())[0]
        self.eeg_log = files['event_log'][0]
        self.eeg_file = glob.glob(os.path.join(eeg_dir, '*.edf'))[0]
        self.eeg = mne.io.read_raw_edf(self.eeg_file, preload=True)
        self.ev_ms = events.view(np.recarray).mstime
        self.events = events.view(np.recarray)
    
    @staticmethod
    def extract_eegstart(logfile):
        """
        Extract timing of EEGSTART event from a session log for system 4.

        :param logfile: The filepath for the event.log jsonl file
        :return: the mstime at which the eeg file began recording
        """
        # Read session log
        df = pd.read_json(logfile, lines=True)
        # Get the mstime of eeg start
        eeg_start_ms = df[df.type == 'EEGSTART'].time.iloc[0]
        return eeg_start_ms
    
    def align(self):
        # Skip alignment if there are no events or no sync pulse logs
        if self.events.shape == ():
            logger.warn('Skipping alignment due to there being no events')
            return self.events

        logger.debug('Aligning...')

        # get the sample rate and length of recording for the current file
        self.num_samples = self.eeg.n_times
        self.sample_rate = self.eeg.info['sfreq']
        
        # get eeg start time
        self.eeg_start_ms = self.extract_eegstart(self.eeg_log)
        
        #import pdb; pdb.set_trace()
        # Calculate the eeg offset for each event
        logger.debug('Calculating EEG offsets...')
        try:
            eeg_offsets = np.round((self.ev_ms - self.eeg_start_ms) * self.sample_rate / 1000.).astype(int)
            
            logger.debug('Done.')

            # Add eeg offset and eeg file information to the events
            logger.debug('Adding EEG file and offset information to events structure...')

            oob = 0  # Counts the number of events that are out of bounds of the start and end sync pulses
            for i in range(self.events.shape[0]):
                if 0 <= eeg_offsets[i] <= self.num_samples:
                    self.events[i].eegoffset = eeg_offsets[i]
                    self.events[i].eegfile = self.eeg_file_stem
                else:
                    oob += 1
            logger.debug('Done.')

            if oob > 0:
                logger.warn(str(oob) + ' events are out of bounds of the EEG files.')

        except ValueError:
            logger.warn('Unable to align events with EEG data!')

        return self.events



class System4Aligner:
    """
    Used for aligning the EEG data from the ephys computer with the task events from the behavioral computer.
    """
    ALIGNMENT_WINDOW = 100  # Tries to align this many sync pulses
    ALIGNMENT_THRESH = 10  # This many milliseconds may differ between sync pulse times during matching

    def __init__(self, events, files, eeg_dir):
        """
        Constructor for the System4 (Elemem) aligner.

        :param events: The events structure to be aligned (np.recarray).
        :param behav_log: The filepath to the logfile with behavioral data, assumed to be .jsonl format
        :param eeg_dir: The path to the session's eeg directory.

        DATA FIELDS:
        behav_log: the logfile with behavioral data, assumed to be .jsonl format 
        eeg_log: The event.log file created by system 4, containing heartbeats according the ephys computer's clock
        eeg: A dictionary matching the basename of each EEG recording to its data (designed for cases with multiple 
        recordings from a single session).
        num_samples: The number of EEG samples in the current EEG recording.
        sample_rate: The sample rate of the current EEG recording.
        pulses: A numpy array containing the indices of EEG samples that contain sync pulses.
        ephys_ms: The mstimes of the sync pulses received by the ephys computer.
        behav_ms: The mstimes of the sync pulses sent by the behavioral computer.
        ev_ms: The mstimes of all task events.
        events: The events structure for the experimental session.
        """
        self.behav_log = files['session_log']
        eeg_sources = json.load(open(files['eeg_sources']))
        if len(eeg_sources) != 1:
            raise AlignmentError('Cannot align EEG with %d sources' % len(eeg_sources))
        self.eeg_file_stem = list(eeg_sources.keys())[0]
        self.eeg_dir = eeg_dir  # Path to current_processed ephys files
        # Get list of the ephys computer's EEG recordings, then get a list of their basenames, and create a Raw object
        # for each
        # FIXME: probably does not need to be iterable for system 4. Ask Ryan.
        self.eeg_files = glob.glob(os.path.join(eeg_dir, '*.edf'))
        self.eeg_log = files['event_log'][0]
        self.eeg = {}
        for f in self.eeg_files:
            basename = os.path.basename(f)
            self.eeg[basename] = mne.io.read_raw_edf(f, preload=True)

        self.num_samples = None
        self.sample_rate = None
        self.ephys_ms = None
        self.behav_ms = None
        self.is_unity = self.behav_log.endswith('.jsonl')
        self.ev_ms = events.view(np.recarray).mstime
        self.events = events.view(np.recarray)

    def align(self):
        """
        Aligns the times at which sync pulses were sent by the behavioral computer with the times at which they were
        received by the ephys computer. This enables conversions to be made between the mstimes of behavioral events
        and the indices of EEG data samples that were taken at the same time. Behavioral sync pulse times come from
        the eeg.eeglog and eeg.eeglog.up files, located in the main session directory. Ephys sync pulses are identified
        from event channels within the EEG recording. A linear regression is run on the behavioral and ephys sync pulse 
        times in order to calculate the EEG sample number that corresponds to each behavioral event. The events 
        structure is then updated with this information.

        :return: The updated events structure, now filled with eegfile and eegoffset information.
        """
        # Skip alignment if there are no events or no sync pulse logs
        if self.events.shape == () or len(self.eeg_files) == 0:
            logger.warn('Skipping alignment due to there being no events or no EEG parameter info.')
            return self.events

        logger.debug('Aligning...')

        # Get the behavioral heartbeat data
        if self.is_unity:
            self.behav_ms = self.extract_heartbeats_unity(self.behav_log)
        else:
            logger.warn('No session.jsonl logfile could be found. Unable to align behavioral and EEG data.')
            return self.events

        if not isinstance(self.behav_ms, np.ndarray) or len(self.behav_ms) < 2:
            logger.warn('No heartbeats were found in the session log. Unable to align behavioral and EEG data.')
            return self.events

        # Align each EEG file
        for basename in self.eeg:
            logger.debug('Calculating alignment for recording, ' + basename)

            # Reset ephys sync pulse info and get the sample rate and length of recording for the current file
            self.num_samples = self.eeg[basename].n_times
            self.sample_rate = self.eeg[basename].info['sfreq']
            
            # Grab logged start time for the edf file
            self.eeg_start_ms = self.extract_eegstart(self.eeg_log)

            self.ephys_ms = self.extract_heartbeats_eventlog(self.eeg_log)
            if not isinstance(self.ephys_ms, np.ndarray) or len(self.ephys_ms) < 2:
                logger.warn('No heartbeats were found in the event.log file. Unable to align behavioral and EEG data.')
                return self.events

            # Calculate the eeg offset for each event
            logger.debug('Calculating EEG offsets...')
            try:
                eeg_offsets, s_ind, e_ind = times_to_offsets(self.behav_ms, self.ephys_ms, self.ev_ms, self.eeg_start_ms, self.sample_rate,
                                                             window=self.ALIGNMENT_WINDOW, thresh_ms=self.ALIGNMENT_THRESH)
                logger.debug('Done.')

                # Add eeg offset and eeg file information to the events
                logger.debug('Adding EEG file and offset information to events structure...')

                # FIXME: point to split eeg files? Should really implement an EDFReader in cmlreaders
                eegfile_name = self.eeg_file_stem 

                oob = 0  # Counts the number of events that are out of bounds of the start and end sync pulses
                for i in range(self.events.shape[0]):
                    if 0 <= eeg_offsets[i] <= self.num_samples:
                        self.events[i].eegoffset = eeg_offsets[i]
                        self.events[i].eegfile = eegfile_name
                    else:
                        oob += 1
                logger.debug('Done.')

                if oob > 0:
                    logger.warn(str(oob) + ' events are out of bounds of the EEG files.')

            except ValueError:
                logger.warn('Unable to align events with EEG data!')

        return self.events

    @staticmethod
    def extract_heartbeats_eventlog(logfile):
        """
        Extract timing of heartbeat events from a session log for system 4.

        :param logfile: The filepath for the event.log jsonl file
        :return: 1-D numpy array containing the mstimes for all heartbeats
        """
        # Read session log
        df = pd.read_json(logfile, lines=True)
        # Get the times when all heartbeats were sent
        heartbeats = np.array(df[df.type == 'HEARTBEAT'].time)
        # Convert pulse times to integers before returning
        heartbeats = heartbeats.astype(int)
        return heartbeats

    @staticmethod
    def extract_heartbeats_unity(logfile):
        """
        Extract timing of heartbeat events from a UnityEPL task session log for system 4.

        :param logfile: The filepath for the session log .jsonl file
        :return: 1-D numpy array containing the mstimes for all heartbeats
        """
        strip_empty_lines(logfile) 
        # Read session log
        df = pd.read_json(logfile, lines=True)
        # Get the times when all heartbeats were sent
        # We want only the received elemem network messages containing heartbeats.
        messages = df[df.type=='network'].data.apply(lambda data: data.get('message') if
                                                     data.get('sent')=='True' else None).dropna()
        # C# syntax not understood so replace false/true with False/True
        messages = messages.astype(str).str.replace('false', 'False').replace('true', 'True') 
        # "message" field is a string, not a dict, so tell pandas it's a string then eval as dict.
        # .apply(pd.Series) explodes dictionary fields as columns
        messages = messages.astype(str).apply(eval).dropna().apply(pd.Series)
        heartbeats = messages[messages.type=='HEARTBEAT'].time.values
        # Convert pulse times to integers before returning
        heartbeats = heartbeats.astype(int)
        return heartbeats

    @staticmethod
    def extract_eegstart(logfile):
        """
        Extract timing of heartbeat events from a session log for system 4.

        :param logfile: The filepath for the event.log jsonl file
        :return: the mstime at which the eeg file began recording
        """
        # Read session log
        df = pd.read_json(logfile, lines=True)
        # Get the mstime of eeg start
        eeg_start_ms = df[df.type == 'EEGSTART'].time.iloc[0]
        return eeg_start_ms

def strip_empty_lines(logfile):
    """
    pandas read_json(line=True) fails if the file has empty lines. 
    This function checks for empty lines and removes them if they exist
    """
    with open(logfile,'r') as file:
        lines = file.readlines()
    newlines = []
    for line in lines:
        if not line.strip():
            continue
        else:
            newlines.append(line)
    if len(lines)==len(newlines):
        import sys
        return
    with open(logfile, 'w') as file:
        file.writelines(newlines)

def times_to_offsets(behav_ms, ephys_ms, ev_ms, eeg_start_ms, samplerate, window=100, thresh_ms=10):
    """
    Slightly modified version of the times_to_offsets_old function in PTSA's alignment systems. Aligns the sync pulses
    sent by the behavioral computer with those received by the ephys computer in order to find the start and end window
    of the experiment. It then runs a regression on the behavioral and ephys sync pulse times within the starting and
    ending windows in order to calculate which EEG sample number corresponds to each task event's mstime.

    Alignment example: Suppose the behavioral computer sent the first sync pulse, then another 920 ms later, then
    another 892 ms after that, followed by another sync pulse 1011 ms later. In order to identify which sync pulses
    received by the ephys computer correspond to those sent by the behavioral computer, we would look for a set of 4
    pulses in the ephys computer's data that approximately follows the same 920 ms, 892 ms, 1011 ms pattern of spacing.
    We would then take this set of pulses as our start window. (Assuming our window parameter was 4, rather than 100)

    :param behav_ms: The mstimes for all sync pulses sent by the behavioral computer.
    :param ephys_ms: The mstimes for all sync pulses received by the ephys computer.
    :param ev_ms: The mstimes for all task events.
    :param samplerate: The sample rate of the EEG recording (typically 500).
    :param window: The number of sync pulses to match between the behavioral and ephys computers' logs at the beginning
    and end of the recording.
    :param thresh_ms: The magnitude of discrepancy permitted when matching behavioral and ephys sync pulses.

    :return offsets: A numpy array containing the EEG sample number corresponding to the onset of each task event.
    :return s_ind: The index of the EEG sample that matches the beginning of the behavioral pulse syncs.
    :return e_ind: The index of the EEG sample that matches the end of the behavioral pulse syncs.
    """
    s_ind = None
    e_ind = None

    # Determine which range of samples in the ephys computer's heartbeat log matches the behavioral computer's heartbeat timings
    # Determine which ephys sync pulses correspond to the beginning behavioral sync pulses
    for i in range(len(ephys_ms) - window):
        s_ind = match_sequence(np.diff(ephys_ms[i:i + window]), np.diff(behav_ms), thresh_ms)
        if s_ind is not None:
            start_ephys_vals = ephys_ms[i:i + window]
            start_behav_vals = behav_ms[s_ind:s_ind + window]
            break
    if s_ind is None:
        raise ValueError("Unable to find a start window.")

    # Determine which ephys sync pulses correspond with the ending behavioral sync pulses
    for i in range(len(ephys_ms) - window):
        e_ind = match_sequence(np.diff(ephys_ms[::-1][i:i + window]), np.diff(behav_ms[::-1]), thresh_ms)
        if e_ind is not None:
            e_ind = len(behav_ms) - e_ind - window
            i = len(ephys_ms) - i - window
            end_ephys_vals = ephys_ms[i:i + window]
            end_behav_vals = behav_ms[e_ind:e_ind + window]
            break
    if e_ind is None:
        raise ValueError("Unable to find an end window.")

    # Perform a regression on the corresponding behavioral and ephys sync pulse times to enable a conversion between
    # event mstimes and EEG offsets.
    x = np.r_[start_behav_vals, end_behav_vals]
    y = np.r_[start_ephys_vals, end_ephys_vals]
    m, c = np.polyfit(x, y, 1)
    
    # FIXME: replace y[0] with the actual eeg start time in ms 
    # Use the regression to convert task event mstimes to EEG offsets
    offsets = np.round((m*ev_ms + c - eeg_start_ms)*samplerate/1000.).astype(int)
    # eeg_start = np.round(eeg_start_ms*samplerate/1000.)

    return offsets, s_ind, e_ind


def match_sequence(needle, haystack, maxdiff):
    """
    Look for a matching subsequence in a long sequence.
    """
    nlen = len(needle)
    found = False
    for i in range(len(haystack)-nlen):
        if np.abs(haystack[i:i+nlen] - needle).max() < maxdiff:
            found = True
            break
    if not found:
        i = None
    return i
