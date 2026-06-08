import os
import sys
import mne
import glob
import numpy as np
import pandas as pd
import json
from ..log import logger
from ..exc import AlignmentError

# Reuse the robust HEARTBEAT-correction fit logic and the non-heartbeat
# message-matching helpers from the heartbeat_correction git submodule, which
# lives at <repo root>/heartbeat_correction (3 dirs up from this file's
# alignment/ directory). Added to sys.path so its flat modules are importable.
_HB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'heartbeat_correction'))
if _HB_DIR not in sys.path:
    sys.path.insert(0, _HB_DIR)
from fix_heartbeats_sys4 import prepare_merged_heartbeats, fit_correction, correct_event_times
from check_nonheart import score_session

# Per-experiment non-heartbeat alignment messages. One entry per experiment.
# Each value is a LIST of (task_type, host_type) pairs; all listed types are
# pooled into one correction fit so multiple message types can be used
# together. Placeholders for now; the real types per experiment will be filled
# in later, and this map will eventually be relocated for modularity.
NONHB_EVENT_MAP = {
    # FR family (free recall) — WORD is the primary (dense) anchor. Task laptop
    # and Elemem host use the same string for every shared behavioral event, so
    # all pairs are (X, X). Host-only control/stim events (START, CONFIGURE,
    # CONNECTED, READY, EEGSTART, STIMMING, etc.) are excluded.
    'IFR1':    [('WORD', 'WORD'), ('ORIENT', 'ORIENT'), ('ENCODING', 'ENCODING'), ('COUNTDOWN', 'COUNTDOWN'), ('DISTRACT', 'DISTRACT'), ('RETRIEVAL', 'RETRIEVAL'), ('TRIAL', 'TRIAL')],
    'IFR6':    [('WORD', 'WORD'), ('ORIENT', 'ORIENT'), ('ENCODING', 'ENCODING'), ('COUNTDOWN', 'COUNTDOWN'), ('DISTRACT', 'DISTRACT'), ('RETRIEVAL', 'RETRIEVAL'), ('TRIAL', 'TRIAL')],
    'ICatFR1': [('WORD', 'WORD'), ('ORIENT', 'ORIENT'), ('ENCODING', 'ENCODING'), ('COUNTDOWN', 'COUNTDOWN'), ('DISTRACT', 'DISTRACT'), ('RETRIEVAL', 'RETRIEVAL'), ('TRIAL', 'TRIAL')],
    'ICatFR6': [('WORD', 'WORD'), ('ORIENT', 'ORIENT'), ('ENCODING', 'ENCODING'), ('COUNTDOWN', 'COUNTDOWN'), ('DISTRACT', 'DISTRACT'), ('RETRIEVAL', 'RETRIEVAL'), ('TRIAL', 'TRIAL')],
    'catFR1':  [('WORD', 'WORD'), ('ORIENT', 'ORIENT'), ('ENCODING', 'ENCODING'), ('COUNTDOWN', 'COUNTDOWN'), ('DISTRACT', 'DISTRACT'), ('RETRIEVAL', 'RETRIEVAL'), ('TRIAL', 'TRIAL'), ('MATH', 'MATH')],
    # RepFR family — recall state is named RECALL (not RETRIEVAL); inter-stim is
    # ISI (not ORIENT).
    'RepFR1':  [('WORD', 'WORD'), ('ISI', 'ISI'), ('RECALL', 'RECALL'), ('COUNTDOWN', 'COUNTDOWN'), ('TRIAL', 'TRIAL'), ('TRIALEND', 'TRIALEND'), ('SESSION', 'SESSION')],
    'RepFR2':  [('WORD', 'WORD'), ('ISI', 'ISI'), ('RECALL', 'RECALL'), ('COUNTDOWN', 'COUNTDOWN'), ('TRIAL', 'TRIAL'), ('TRIALEND', 'TRIALEND'), ('SESSION', 'SESSION'), ('READY', 'READY')],
    # EFRCourier (spatial delivery) — no WORD; OBJECT_PRESENTATION_BEGINS is the
    # item-onset anchor.
    'EFRCourierOpenLoop': [('OBJECT_PRESENTATION_BEGINS', 'OBJECT_PRESENTATION_BEGINS'), ('ORIENT', 'ORIENT'), ('ENCODING', 'ENCODING'), ('RETRIEVAL', 'RETRIEVAL'), ('TRIAL', 'TRIAL'), ('OBJECT_RECALL_RECORDING_START', 'OBJECT_RECALL_RECORDING_START'), ('CUED_RECALL_RECORDING_START', 'CUED_RECALL_RECORDING_START')],
    'EFRCourierReadOnly': [('OBJECT_PRESENTATION_BEGINS', 'OBJECT_PRESENTATION_BEGINS'), ('ORIENT', 'ORIENT'), ('ENCODING', 'ENCODING'), ('RETRIEVAL', 'RETRIEVAL'), ('TRIAL', 'TRIAL'), ('OBJECT_RECALL_RECORDING_START', 'OBJECT_RECALL_RECORDING_START'), ('CUED_RECALL_RECORDING_START', 'CUED_RECALL_RECORDING_START')],
    # CPS (closed-loop) — behavioral anchors present on both clocks.
    'CPS':     [('ENCODING', 'ENCODING'), ('TRIAL', 'TRIAL'), ('WAITING', 'WAITING'), ('VOCALIZATION', 'VOCALIZATION')],
    # OPS — stim-only parameter search; no task behavioral message stream, so no
    # non-heartbeat anchors.
    'OPS':     [],
}

class System4Offset:
    def __init__(self, events, files, eeg_dir):
        eeg_sources = json.load(open(files['eeg_sources']))
        if len(eeg_sources) != 1:
            raise AlignmentError('Cannot align EEG with %d sources' % len(eeg_sources))
        self.eeg_file_stem = list(eeg_sources.keys())[0]
        self.eeg_log = files['event_log'][0]
        self.eeg_file = sorted(glob.glob(os.path.join(eeg_dir, '*.edf')), key=os.path.getmtime, reverse=True)[0]
        self.eeg = mne.io.read_raw_edf(self.eeg_file, preload=True)
        self.ev_ms = events.view(np.recarray).mstime
        self.events = events.view(np.recarray)
        #logger.debug("Event fields = {}".format(events.view(np.recarray).dtype.names))
        #logger.debug("eeg_dir = {}, eeg_file = {}".format(eeg_dir, self.eeg_file))
    
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
            logger.error('Skipping alignment due to there being no events')
            return self.events

        logger.debug('Aligning...')

        # get the sample rate and length of recording for the current file
        self.num_samples = self.eeg.n_times
        self.sample_rate = self.eeg.info['sfreq']
        
        # get eeg start time
        self.eeg_start_ms = self.extract_eegstart(self.eeg_log)
        
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
                logger.error(str(oob) + ' events are out of bounds of the EEG files.')

        except ValueError:
            logger.error('Unable to align events with EEG data!')

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
            logger.error('Skipping alignment due to there being no events or no EEG parameter info.')
            return self.events

        logger.debug('Aligning...')

        # Get the behavioral heartbeat data
        if self.is_unity:
            self.behav_ms = self.extract_heartbeats_unity(self.behav_log)
        else:
            logger.error('No session.jsonl logfile could be found. Unable to align behavioral and EEG data.')
            return self.events

        if not isinstance(self.behav_ms, np.ndarray) or len(self.behav_ms) < 2:
            logger.error('No heartbeats were found in the session log. Unable to align behavioral and EEG data.')
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
                logger.error('No heartbeats were found in the event.log file. Unable to align behavioral and EEG data.')
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
                    logger.error(str(oob) + ' events are out of bounds of the EEG files.')

            except ValueError:
                logger.error('Unable to align events with EEG data!')

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

class System4AlignerHB:
    """
    Aligns System 4 (Elemem) EEG data to behavioral task events using the
    HEARTBEAT messages exchanged once per second between the task laptop and the
    host PC (elemem). Unlike the legacy ``System4Aligner`` (sync-pulse pattern
    matching), this reuses the robust correction pipeline from the
    ``heartbeat_correction`` submodule: it fits ``time_host = slope*time_task +
    offset`` with RANSAC (outlier rejection + network-latency adjustment + fit
    validation). This is a *correction pass* over already-aligned events: it
    rewrites both ``mstime`` (full slope+offset) and ``eegoffset`` (slope-only
    drift) via the submodule's ``correct_event_times``, mirroring
    ``fix_heartbeats_for_session``.
    """

    def __init__(self, events, files, eeg_dir):
        """
        :param events: The events structure to be aligned (np.recarray).
        :param files: dict of pipeline file paths; uses 'session_log' (task
            laptop session.jsonl), 'event_log' (host PC event.log), and
            'eeg_sources'.
        :param eeg_dir: The path to the session's eeg directory.
        """
        self.behav_log = files['session_log']
        eeg_sources = json.load(open(files['eeg_sources']))
        if len(eeg_sources) != 1:
            raise AlignmentError('Cannot align EEG with %d sources' % len(eeg_sources))
        self.eeg_file_stem = list(eeg_sources.keys())[0]
        self.eeg_dir = eeg_dir
        self.eeg_files = glob.glob(os.path.join(eeg_dir, '*.edf'))
        self.eeg_log = files['event_log'][0]
        self.eeg = {}
        for f in self.eeg_files:
            basename = os.path.basename(f)
            self.eeg[basename] = mne.io.read_raw_edf(f, preload=True)

        self.num_samples = None
        self.sample_rate = None
        self.eeg_start_ms = None
        self.ev_ms = events.view(np.recarray).mstime
        self.events = events.view(np.recarray)

    def align(self):
        """
        Fit the task->host clock correction from HEARTBEATs and apply it to the
        events' ``mstime`` and ``eegoffset``.

        :return: The corrected events structure.
        """
        # Skip alignment if there are no events or no EEG
        if self.events.shape == () or len(self.eeg_files) == 0:
            logger.error('Skipping alignment due to there being no events or no EEG parameter info.')
            return self.events

        logger.debug('Aligning (heartbeats)...')

        # Build the merged task/host heartbeat dataframe and fit the correction
        try:
            task_df = read_heartbeats_from_path(self.behav_log, load_host_pc=False)
            host_df = read_heartbeats_from_path(self.eeg_log, load_host_pc=True)
            merged = prepare_merged_heartbeats(pd.concat([task_df, host_df]))
            res = fit_correction(merged, ignore_errors=True)
            slope = res['slope']
            offset = res['offset']
        except Exception as e:
            logger.error('Unable to compute heartbeat correction (%s: %s). '
                         'Skipping alignment.' % (type(e).__name__, e))
            return self.events

        logger.debug('Heartbeat fit: slope=%s, offset=%s' % (slope, offset))

        self.events = _correct_events(self.events, slope, offset)
        return self.events


class System4AlignerNonHB:
    """
    Aligns System 4 (Elemem) EEG data to behavioral task events using
    *non-heartbeat* messages (e.g. WORD onsets) that appear in both the task
    laptop log and the host PC event.log. The message type(s) to align on are
    looked up per-experiment in ``NONHB_EVENT_MAP``. Matched task/host
    timestamps are pooled across all configured message types and a single
    RANSAC correction ``time_host = slope*time_task + offset`` is fit, then
    applied to ``mstime`` and ``eegoffset`` exactly as in ``System4AlignerHB``.
    """

    MIN_MATCHES = 3      # minimum pooled (task, host) pairs needed to fit
    RANSAC_INLIER_MS = 20.0   # residual threshold (ms); matches check_nonheart

    def __init__(self, events, files, eeg_dir, message_types=None):
        """
        :param events: The events structure to be aligned (np.recarray).
        :param files: dict of pipeline file paths (see System4AlignerHB).
        :param eeg_dir: The path to the session's eeg directory.
        :param message_types: Optional override; a list of (task_type,
            host_type) pairs to use instead of the experiment's default entry
            in NONHB_EVENT_MAP. Useful to restrict to a subset.
        """
        self.behav_log = files['session_log']
        eeg_sources = json.load(open(files['eeg_sources']))
        if len(eeg_sources) != 1:
            raise AlignmentError('Cannot align EEG with %d sources' % len(eeg_sources))
        self.eeg_file_stem = list(eeg_sources.keys())[0]
        self.eeg_dir = eeg_dir
        self.eeg_files = glob.glob(os.path.join(eeg_dir, '*.edf'))
        self.eeg_log = files['event_log'][0]
        self.eeg = {}
        for f in self.eeg_files:
            basename = os.path.basename(f)
            self.eeg[basename] = mne.io.read_raw_edf(f, preload=True)

        self.num_samples = None
        self.sample_rate = None
        self.eeg_start_ms = None
        self.ev_ms = events.view(np.recarray).mstime
        self.events = events.view(np.recarray)

        # Determine the experiment so we can look up the message types to match.
        rec = events.view(np.recarray)
        self.experiment = str(rec.experiment[0]) if 'experiment' in rec.dtype.names and rec.shape != () else None
        self.message_types = message_types

    def _resolve_message_types(self):
        """Return the list of (task_type, host_type) pairs to align on."""
        if self.message_types is not None:
            return self.message_types
        if self.experiment is None or self.experiment not in NONHB_EVENT_MAP:
            raise AlignmentError(
                'No non-heartbeat message types configured for experiment %r' % self.experiment)
        return NONHB_EVENT_MAP[self.experiment]

    def align(self):
        """
        Fit the task->host clock correction from non-heartbeat messages and
        apply it to the events' ``mstime`` and ``eegoffset``.

        :return: The corrected events structure.
        """
        if self.events.shape == () or len(self.eeg_files) == 0:
            logger.error('Skipping alignment due to there being no events or no EEG parameter info.')
            return self.events

        logger.debug('Aligning (non-heartbeat messages)...')

        try:
            pairs = self._resolve_message_types()
            # score_session reads both file paths directly and strips heartbeats
            _, _, _, task_by_type, host_by_type, _ = score_session(
                self.behav_log, self.eeg_log, include_heartbeats=False)

            # Pool matched task/host timestamps across all configured message types
            task_pooled, host_pooled = [], []
            for task_type, host_type in pairs:
                t_times = sorted(task_by_type.get(task_type.upper(), []))
                h_times = sorted(host_by_type.get(host_type.upper(), []))
                n = min(len(t_times), len(h_times))
                if n == 0:
                    continue
                task_pooled.extend(t_times[:n])
                host_pooled.extend(h_times[:n])

            if len(task_pooled) < self.MIN_MATCHES:
                logger.error('Too few matched non-heartbeat messages (%d < %d) to align. '
                             'Skipping alignment.' % (len(task_pooled), self.MIN_MATCHES))
                return self.events

            slope, offset = self._fit(np.array(task_pooled, dtype=float),
                                      np.array(host_pooled, dtype=float))
        except Exception as e:
            logger.error('Unable to compute non-heartbeat correction (%s: %s). '
                         'Skipping alignment.' % (type(e).__name__, e))
            return self.events

        logger.debug('Non-heartbeat fit: slope=%s, offset=%s' % (slope, offset))

        self.events = _correct_events(self.events, slope, offset)
        return self.events

    def _fit(self, task_ms, host_ms):
        """RANSAC-fit host_time = slope*task_time + offset on pooled points."""
        from sklearn.linear_model import LinearRegression, RANSACRegressor
        ransac = RANSACRegressor(estimator=LinearRegression(),
                                 residual_threshold=self.RANSAC_INLIER_MS,
                                 random_state=0)
        ransac.fit(task_ms.reshape(-1, 1), host_ms)
        slope = ransac.estimator_.coef_[0]
        offset = ransac.estimator_.intercept_
        return slope, offset


def read_heartbeats_from_path(path, load_host_pc=False, drop_network_test=True):
    """
    Read HEARTBEAT/HEARTBEAT_OK records from a single log file and return a
    DataFrame with the columns ``prepare_merged_heartbeats`` expects
    (``count``, ``time``, ``latency``, ``session``, ``hardware_system``).

    This mirrors the post-file-read parsing in the submodule's ``get_heart``
    but reads from an explicit path (the task laptop ``session.jsonl`` when
    ``load_host_pc`` is False, or the host PC ``event.log`` when True) rather
    than relocating logs via the CMLReader data index.

    :param path: Path to the log file.
    :param load_host_pc: True for the host PC event.log, False for the task log.
    :param drop_network_test: Drop the initial network-test heartbeats (count<=20).
    """
    log = []
    with open(path, 'r') as fr:
        for line in fr:
            try:
                log.append(json.loads(line))
            except Exception:
                continue
    heart_beat = pd.DataFrame(log)
    heart_beat['session'] = 0  # single-session: satisfies downstream assertions

    if load_host_pc:
        heart_beat = heart_beat[heart_beat.type.isin(['HEARTBEAT', 'HEARTBEAT_OK'])]
        heart_beat['count'] = heart_beat.data.apply(lambda x: _hb_field(x, 'count'))
    else:
        heart_beat['message'] = heart_beat.data.apply(lambda x: _hb_field(x, 'message'))
        heart_beat.dropna(subset=['message'], inplace=True)
        heart_beat['type'] = heart_beat.message.apply(lambda x: _hb_field(x, 'type'))
        heart_beat['data'] = heart_beat.message.apply(lambda x: _hb_field(x, 'data'))
        heart_beat = heart_beat[heart_beat.type.isin(['HEARTBEAT', 'HEARTBEAT_OK'])]
        heart_beat['count'] = heart_beat.data.apply(lambda x: _hb_field(x, 'count'))
    if len(heart_beat) == 0:
        raise ValueError('No HEARTBEAT / HEARTBEAT_OK events logged in %s!' % path)

    if drop_network_test:
        heart_beat = heart_beat[heart_beat['count'] > 20]

    # Pair HEARTBEAT (sent) with HEARTBEAT_OK (acknowledged) on count to get latency
    bpm_sent = heart_beat[heart_beat.type == 'HEARTBEAT'].set_index('count')
    bpm_done = heart_beat[heart_beat.type == 'HEARTBEAT_OK'].set_index('count')
    bpm_err = bpm_done.time.astype(float) - bpm_sent.time.astype(float)

    hardware_system = 'host_pc' if load_host_pc else 'task_laptop'
    heart_beat = heart_beat.query('type == "HEARTBEAT"')
    if 'message' in heart_beat.columns:
        heart_beat.drop('message', axis=1, inplace=True)
    heart_beat.set_index('count', inplace=True, drop=False)
    heart_beat.loc[:, ['latency']] = bpm_err
    heart_beat.loc[:, ['hardware_system']] = hardware_system
    return heart_beat


def _hb_field(obj, key):
    """obj[key] tolerant of obj being a JSON-encoded string or non-dict."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def _correct_events(events, slope, offset):
    """
    Apply a fitted ``time_host = slope*time_task + offset`` correction to both
    time fields of an events recarray, mirroring the submodule's
    ``fix_heartbeats_for_session``:

    - ``mstime``   — an absolute task-laptop timestamp, so the full slope+offset
      correction applies.
    - ``eegoffset`` — a sample index whose origin is the EEG file start (already
      in the host-PC reference), so only the slope (sample-rate drift) applies;
      it is rounded back to its original integer dtype.

    Reuses ``correct_event_times`` (which does ``events.copy()`` + column
    assignment, working on the numpy recarray) and returns the corrected
    recarray. Shared by System4AlignerHB and System4AlignerNonHB.
    """
    logger.debug('Applying clock correction to mstime and eegoffset...')
    corrected = correct_event_times(events, offset, slope, time_col='mstime')
    eegoffset_dtype = events['eegoffset'].dtype
    corrected = correct_event_times(corrected, 0, slope, time_col='eegoffset')
    corrected['eegoffset'] = np.round(corrected['eegoffset']).astype(eegoffset_dtype)
    logger.debug('Done.')
    return corrected.view(np.recarray)


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
