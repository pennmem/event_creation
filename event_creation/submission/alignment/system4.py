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
from fix_heartbeats_sys4 import correct_event_times
from check_nonheart import score_session, read_jsonl, task_type, task_time

# HEARTBEAT is treated as just another message type for fitting; this is the
# (task_type, host_type) pair the gather loop uses for heartbeat points.
HEARTBEAT_PAIRS = [('HEARTBEAT', 'HEARTBEAT')]

# Per-experiment non-heartbeat alignment messages. One entry per experiment.
# Each value is a LIST of (task_type, host_type) pairs; all listed types are
# pooled into one correction fit so multiple message types can be used
# together. Placeholders for now; the real types per experiment will be filled
# in later, and this map will eventually be relocated for modularity.
NONHB_EVENT_MAP = {
    # UnityEPL FR family (IFR/ICatFR) — the task laptop logs human-readable
    # event names that DIFFER from the Elemem host's short names, so pairs are
    # asymmetric (task_name, host_name). WORD STIMULUS is the dense anchor
    # (~one per presented word, hundreds per session); the rest are per-trial.
    # ENCODING/COUNTDOWN are omitted: on the task laptop they live inside STATE
    # transitions (not decoded here) and never matched as standalone types.
    # Host-only control/stim events (START, CONFIGURE, READY, STIMMING, …) are
    # excluded by _elemem_originated_for(). NB: these sessions log NO heartbeats
    # on the task laptop, so the non-heartbeat sweep is the only fit path.
    'IFR1':    [('WORD STIMULUS', 'WORD'), ('ORIENTATION STIMULUS', 'ORIENT'), ('DISPLAY DISTRACTOR FIXATION CROSS', 'DISTRACT'), ('DISPLAY RECALL TEXT', 'RETRIEVAL'), ('TRIAL', 'TRIAL')],
    'IFR6':    [('WORD STIMULUS', 'WORD'), ('ORIENTATION STIMULUS', 'ORIENT'), ('DISPLAY DISTRACTOR FIXATION CROSS', 'DISTRACT'), ('DISPLAY RECALL TEXT', 'RETRIEVAL'), ('TRIAL', 'TRIAL')],
    'ICatFR1': [('WORD STIMULUS', 'WORD'), ('ORIENTATION STIMULUS', 'ORIENT'), ('DISPLAY DISTRACTOR FIXATION CROSS', 'DISTRACT'), ('DISPLAY RECALL TEXT', 'RETRIEVAL'), ('TRIAL', 'TRIAL')],
    'ICatFR6': [('WORD STIMULUS', 'WORD'), ('ORIENTATION STIMULUS', 'ORIENT'), ('DISPLAY DISTRACTOR FIXATION CROSS', 'DISTRACT'), ('DISPLAY RECALL TEXT', 'RETRIEVAL'), ('TRIAL', 'TRIAL')],
    # FR/catFR family — task laptop and Elemem host use the same string for every
    # shared behavioral event, so all pairs are (X, X). Host-only control/stim
    # events (START, CONFIGURE, CONNECTED, READY, EEGSTART, STIMMING, …) excluded.
    'catFR1':  [('WORD', 'WORD'), ('ORIENT', 'ORIENT'), ('ENCODING', 'ENCODING'), ('COUNTDOWN', 'COUNTDOWN'), ('DISTRACT', 'DISTRACT'), ('RETRIEVAL', 'RETRIEVAL'), ('TRIAL', 'TRIAL'), ('MATH', 'MATH')],
    # RepFR family — recall state is named RECALL (not RETRIEVAL); inter-stim is
    # ISI (not ORIENT).
    'RepFR1':  [('WORD', 'WORD'), ('ISI', 'ISI'), ('RECALL', 'RECALL'), ('COUNTDOWN', 'COUNTDOWN'), ('TRIAL', 'TRIAL'), ('TRIALEND', 'TRIALEND'), ('SESSION', 'SESSION')],
    'RepFR2':  [('WORD', 'WORD'), ('ISI', 'ISI'), ('RECALL', 'RECALL'), ('COUNTDOWN', 'COUNTDOWN'), ('TRIAL', 'TRIAL'), ('TRIALEND', 'TRIALEND'), ('SESSION', 'SESSION')],
    # EFRCourier (spatial delivery, UnityEPL) — the task laptop logs human-readable
    # SPACE-separated names while the Elemem host uses UNDERSCORE names, so pairs are
    # asymmetric (task_name, host_name). The task laptop logs NO heartbeats, so the
    # non-heartbeat sweep is the only fit path. Anchors are discrete one-shot events
    # that match ~1 ms between clocks (verified); PLAYERTRANSFORM is deliberately
    # excluded -- it streams with ~50 ms task/host jitter and is not a tight anchor.
    # The host-only state names (ORIENT/ENCODING/RETRIEVAL/TRIAL) have no task-laptop
    # counterpart and are dropped.
    'EFRCourierOpenLoop': [('OBJECT PRESENTATION BEGINS', 'OBJECT_PRESENTATION_BEGINS'), ('OBJECT RECALL RECORDING START', 'OBJECT_RECALL_RECORDING_START'), ('POINTING BEGINS', 'POINTING_BEGINS'), ('POINTING FINISHED', 'POINTING_FINISHED'), ('POINTER MESSAGE CLEARED', 'POINTER_MESSAGE_CLEARED'), ('AUDIO PRESENTATION FINISHED', 'AUDIO_PRESENTATION_FINISHED')],
    'EFRCourierReadOnly': [('OBJECT PRESENTATION BEGINS', 'OBJECT_PRESENTATION_BEGINS'), ('OBJECT RECALL RECORDING START', 'OBJECT_RECALL_RECORDING_START'), ('POINTING BEGINS', 'POINTING_BEGINS'), ('POINTING FINISHED', 'POINTING_FINISHED'), ('POINTER MESSAGE CLEARED', 'POINTER_MESSAGE_CLEARED'), ('AUDIO PRESENTATION FINISHED', 'AUDIO_PRESENTATION_FINISHED')],
    # CPS (closed-loop) — behavioral anchors present on both clocks.
    'CPS':     [('ENCODING', 'ENCODING'), ('TRIAL', 'TRIAL'), ('WAITING', 'WAITING'), ('VOCALIZATION', 'VOCALIZATION')],
    # OPS — stim-only parameter search; no task behavioral message stream, so no
    # non-heartbeat anchors.
    'OPS':     [],
}

# Control/handshake messages Elemem (host PC) emits every session regardless of
# experiment. HEARTBEAT is intentionally absent — it is the alignment ground
# truth (sent by the task laptop, logged by both clocks); only HEARTBEAT_OK is
# host-originated.
_COMMON_ELEMEM_ORIGINATED = {
    'START', 'EXIT', 'ELEMEM', 'EEGSTART', 'READY', 'WAITING',
    'CONNECTED', 'CONNECTED_OK', 'CONFIGURE', 'CONFIGURE_OK',
    'HEARTBEAT_OK', 'NETWORK', 'VERSIONS', 'EXPERIMENTCONFIG',
    'SYSTEMCONFIG', 'STIMNETMSG', 'STIMSELECT',
}

# Per-experiment Elemem-originated (host-only) event types: stim delivery,
# closed-loop classifier decisions, and stim config. No task-laptop counterpart
# exists, so these must never enter the clock-correction fit. Merged with
# _COMMON_ELEMEM_ORIGINATED by _elemem_originated_for(). Keys mirror
# NONHB_EVENT_MAP. Values verified against real event.log files.
ELEMEM_ORIGINATED = {
    'CPS': {
        'STIM', 'STIMMING', 'SHAM', 'NORMALIZE', 'NORMALIZATION_STATS',
        'ZEROED_ARTIFACT_CHANNELS', 'UPDATE', 'CONFIG_STIM', 'CCLSTARTSTIM',
        'PS_METADATA',
        'STIM_CLASSIFY', 'SHAM_CLASSIFY', 'NOSTIM_CLASSIFY',
        'STIM_DECISION', 'SHAM_DECISION', 'NOSTIM_DECISION',
        'CLASSIFY_STIM_CPS', 'CLASSIFY_SHAM_CPS', 'CLASSIFY_NOSTIM_CPS',
    },
    'EFRCourierOpenLoop': {'STIM', 'STIMMING'},
    'EFRCourierReadOnly': set(),
    'OPS': {'STIM', 'STIMMING', 'SHAM', 'CONFIG_STIM'},
    # FR-family: IFR/ICatFR/catFR1 are non-stim. RepFR can be stim-enabled
    # (RepFR2 event.log emits STIM/STIMMING); these are host-only and must never
    # be fit anchors.
    'IFR1': set(), 'IFR6': set(), 'ICatFR1': set(), 'ICatFR6': set(),
    'catFR1': set(),
    'RepFR1': {'STIM', 'STIMMING', 'SHAM'}, 'RepFR2': {'STIM', 'STIMMING', 'SHAM'},
}


def _elemem_originated_for(experiment):
    """Set of Elemem-originated (host-only) type strings to exclude from the fit
    for `experiment` = the always-present control/handshake base plus any
    experiment-specific stim/closed-loop types. All upper-cased to match the
    normalized type keys returned by score_session."""
    types = set(_COMMON_ELEMEM_ORIGINATED) | set(ELEMEM_ORIGINATED.get(experiment, set()))
    return {t.upper() for t in types}


def _assert_no_elemem_pairs(experiment, pairs):
    """Fail fast if any configured fit anchor is an Elemem-originated type."""
    deny = _elemem_originated_for(experiment)
    bad = {tt for tt, _ in pairs if tt.upper() in deny} | \
          {ht for _, ht in pairs if ht.upper() in deny}
    if bad:
        raise AlignmentError(
            'Elemem-originated type(s) %s configured as fit anchors for '
            'experiment %r; remove them from NONHB_EVENT_MAP.'
            % (sorted(bad), experiment))

class System4Offset:
    def __init__(self, events, files, eeg_dir):
        eeg_sources = json.load(open(files['eeg_sources']))
        if len(eeg_sources) != 1:
            raise AlignmentError('Cannot align EEG with %d sources' % len(eeg_sources))
        self.eeg_file_stem = list(eeg_sources.keys())[0]
        self.eeg_log = files['event_log'][0]
        self.eeg_file = sorted(glob.glob(os.path.join(eeg_dir, '*.edf')), key=os.path.getmtime, reverse=True)[0]
        self.eeg = mne.io.read_raw_edf(self.eeg_file, preload=False)  # header only (n_times/sfreq)
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
            self.eeg[basename] = mne.io.read_raw_edf(f, preload=False)  # header only (n_times/sfreq)

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

class System4AlignerCorrection:
    """
    Correction aligner for System 4 (Elemem) sessions. Fits the task->host clock
    correction ``time_host = slope*time_task + offset`` (shared RANSAC ``_fit``)
    and applies it as a *correction pass* over already-aligned events, rewriting
    both ``mstime`` (full slope+offset) and ``eegoffset`` (slope-only drift) via
    the submodule's ``correct_event_times``, mirroring ``fix_heartbeats_for_session``.

    HEARTBEAT is treated as just another message type, so the fit points are
    gathered the same way regardless of source: ``score_session`` returns matched
    task/host timestamps per type, and we walk a list of ``(task_type, host_type)``
    pairs pooling *all* matched points across every listed type. The pairs to walk
    are selected by ``source``:

    - ``'heartbeat'``    — only the HEARTBEAT messages exchanged once per second
      between the task laptop and the host PC (elemem).
    - ``'nonheartbeat'`` — all non-heartbeat messages (e.g. WORD onsets) for the
      experiment, looked up in ``NONHB_EVENT_MAP``.
    - ``'auto'`` (default) — all messages: heartbeats plus the experiment's
      non-heartbeat messages, pooled together for one fit.
    """

    TARGET_MATCHES = 20  # warn if fewer than this many matched points are pooled
    VALID_SOURCES = ('auto', 'heartbeat', 'nonheartbeat')

    def __init__(self, events, files, eeg_dir, source='auto'):
        """
        :param events: The events structure to be aligned (np.recarray).
        :param files: dict of pipeline file paths; uses 'session_log' (task
            laptop session.jsonl), 'event_log' (host PC event.log), and
            'eeg_sources'.
        :param eeg_dir: The path to the session's eeg directory.
        :param source: 'auto' | 'heartbeat' | 'nonheartbeat' (see class docstring).
        """
        if source not in self.VALID_SOURCES:
            raise AlignmentError('Invalid source %r; expected one of %s'
                                 % (source, self.VALID_SOURCES))
        self.source = source
        self.behav_log = files['session_log']
        eeg_sources = json.load(open(files['eeg_sources']))
        if len(eeg_sources) != 1:
            raise AlignmentError('Cannot align EEG with %d sources' % len(eeg_sources))
        self.eeg_file_stem = list(eeg_sources.keys())[0]
        self.eeg_dir = eeg_dir
        self.eeg_files = glob.glob(os.path.join(eeg_dir, '*.edf'))
        self.eeg_log = files['event_log'][0]
        # Single source (eeg_sources asserted len==1 above); align() uses self.eeg as one
        # MNE Raw (self.eeg.n_times / .info), mirroring System4Offset. Pick the most recent
        # .edf. None when there is no EEG; align() returns early in that case.
        # preload=False: we only read header metadata (n_times, sfreq) -- never the signal --
        # so do NOT load the samples. High-rate sessions (e.g. 30 kHz, ~70M samples) would be
        # tens of GB and OOM the worker on preload=True.
        self.eeg = (mne.io.read_raw_edf(
            sorted(self.eeg_files, key=os.path.getmtime, reverse=True)[0], preload=False)
            if self.eeg_files else None)

        self.num_samples = None
        self.sample_rate = None
        self.eeg_start_ms = None
        self.ev_ms = events.view(np.recarray).mstime
        self.events = events.view(np.recarray)

        # Determine the experiment so we can look up the non-HB message types.
        rec = events.view(np.recarray)
        self.experiment = str(rec.experiment[0]) if 'experiment' in rec.dtype.names and rec.shape != () else None

    def _resolve_pairs(self):
        """Return the ordered list of (task_type, host_type) pairs to gather fit
        points from, based on ``self.source``. Heartbeats come first under
        'auto'."""
        nonhb = NONHB_EVENT_MAP.get(self.experiment)
        if self.source in ('auto', 'nonheartbeat') and nonhb is None:
            raise AlignmentError(
                'No non-heartbeat message types configured for experiment %r' % self.experiment)
        if self.source == 'heartbeat':
            pairs = list(HEARTBEAT_PAIRS)
        elif self.source == 'nonheartbeat':
            pairs = list(nonhb)
        else:  # 'auto'
            pairs = list(HEARTBEAT_PAIRS) + list(nonhb)
        _assert_no_elemem_pairs(self.experiment, pairs)
        return pairs

    def _gather(self, pairs):
        """
        Walk ``pairs``, pooling *all* matched task/host timestamps from
        ``score_session`` across every listed message type.

        :return: (task_ms array, host_ms array).
        """
        # include_heartbeats=True so HEARTBEAT is available as an ordinary type.
        _, _, _, task_by_type, host_by_type, _ = score_session(
            self.behav_log, self.eeg_log, include_heartbeats=True)

        # Defense-in-depth: never pool Elemem-originated (host-only) types into
        # the fit, even if one slips into the configured pair list.
        deny = _elemem_originated_for(self.experiment)

        task_pooled, host_pooled = [], []
        for task_type, host_type in pairs:
            if task_type.upper() in deny or host_type.upper() in deny:
                continue
            t_times = sorted(task_by_type.get(task_type.upper(), []))
            h_times = sorted(host_by_type.get(host_type.upper(), []))
            n = min(len(t_times), len(h_times))
            if n == 0:
                continue
            task_pooled.extend(t_times[:n])
            host_pooled.extend(h_times[:n])

        return (np.array(task_pooled, dtype=float),
                np.array(host_pooled, dtype=float))

    def _fit_relative(self, task_ms, host_ms, residual_threshold):
        """
        Fit ``th_rel = slope*tt_rel + b`` on RELATIVE timestamps, anchoring the
        task side at its first matched heartbeat (``Tt0``) and the host side at
        ``eeg_start_ms``. Doing the regression in relative space keeps the
        ~9 h task<->host clock skew and the ~1.6e12 absolute magnitude out of
        the fit, so ``slope``/residuals are well conditioned (sub-ms RANSAC
        thresholds are meaningful again).

        :return: (slope, b, Tt0, n_inliers).
        """
        task_ms = np.asarray(task_ms, dtype=float)
        host_ms = np.asarray(host_ms, dtype=float)
        Tt0 = float(task_ms[0])
        tt_rel = task_ms - Tt0
        th_rel = host_ms - self.eeg_start_ms
        # _ransac_fit(X, Y) fits Y = slope*X + intercept, i.e. th_rel = slope*tt_rel + b.
        slope, b, n_inliers = _ransac_fit(tt_rel, th_rel,
                                          residual_threshold=residual_threshold)
        return slope, b, Tt0, n_inliers

    def _fit_heartbeats_filtered(self):
        """
        Heartbeat-only step (filter then fit): keep only heartbeats whose
        task-laptop one-way latency is < 1 ms, require >= TARGET_MATCHES of
        them, then RANSAC-fit (on relative times) with a 1 ms residual threshold.

        :return: (slope, b, Tt0) or None on failure (after warning).
        """
        task_ms, host_ms = _heartbeat_points_filtered(self.behav_log, self.eeg_log)
        if len(task_ms) < self.TARGET_MATCHES:
            logger.warn('Heartbeat step: only %d low-latency (<1 ms) heartbeats '
                           'matched (target %d); failed to get enough messages in '
                           'this step.' % (len(task_ms), self.TARGET_MATCHES))
            return None

        slope, b, Tt0, n_inliers = self._fit_relative(task_ms, host_ms,
                                                      residual_threshold=1.0)
        if n_inliers < self.TARGET_MATCHES:
            logger.warn('Heartbeat step: 1 ms RANSAC produced only %d inliers '
                           '(target %d); failed to get enough messages in this step.'
                           % (n_inliers, self.TARGET_MATCHES))
            return None

        logger.debug('Heartbeat fit (relative): slope=%s, b=%s, Tt0=%s (%d inliers)'
                     % (slope, b, Tt0, n_inliers))
        return slope, b, Tt0

    def _fit_sweep(self, pairs):
        """
        All-messages fallback: pool all matched points across ``pairs`` and
        sweep the RANSAC residual threshold over 1..5 ms, accepting the first
        threshold yielding >= TARGET_MATCHES inliers. Warns at each iteration
        that falls short.

        :return: (slope, b, Tt0) or None on failure.
        """
        task_ms, host_ms = self._gather(pairs)
        if len(task_ms) < 2:
            logger.warn('Sweep step: too few matched messages (%d) to fit; '
                           'failed to get enough messages in this step.'
                           % len(task_ms))
            return None

        for thresh in range(1, 6):
            slope, b, Tt0, n_inliers = self._fit_relative(
                task_ms, host_ms, residual_threshold=float(thresh))
            if n_inliers >= self.TARGET_MATCHES:
                logger.debug('Sweep fit (relative) at threshold %d ms: slope=%s, '
                             'b=%s, Tt0=%s (%d inliers)'
                             % (thresh, slope, b, Tt0, n_inliers))
                return slope, b, Tt0
            logger.warn('Sweep step: threshold %d ms produced only %d inliers '
                           '(target %d); failed to get enough messages at threshold '
                           '%d ms.' % (thresh, n_inliers, self.TARGET_MATCHES, thresh))
        return None

    def align(self):
        """
        Fit the task->host clock correction from the selected message source and
        apply it to the events' ``mstime`` and ``eegoffset``.

        :return: The corrected events structure.
        """
        
        # Skip alignment if there are no events or no EEG
        if self.events.shape == () or len(self.eeg_files) == 0:
            logger.error('Skipping alignment due to there being no events or no EEG parameter info.')
            return self.events

        logger.debug('Aligning (source=%s)...' % self.source)
         # get the sample rate and length of recording for the current file
        self.num_samples = self.eeg.n_times
        self.sample_rate = self.eeg.info['sfreq']
        
        # get eeg start time
        self.eeg_start_ms = self.extract_eegstart(self.eeg_log)

        fit = None
        if self.source == 'heartbeat':
            fit = self._fit_heartbeats_filtered()
        elif self.source == 'nonheartbeat':
            fit = self._fit_sweep(self._resolve_pairs())
        else:  # 'auto'
            fit = self._fit_heartbeats_filtered()
            if fit is None:
                logger.warn('Heartbeat-only fit failed; falling back to the '
                               'all-messages RANSAC threshold sweep.')
                fit = self._fit_sweep(self._resolve_pairs())

        if fit is None:
            logger.error('All alignment correction steps failed for source=%s; '
                         'unable to fit clock correction.' % self.source)
            raise AlignmentError('System4AlignerCorrection failed to fit a clock '
                                 'correction for source=%s' % self.source)

        slope, b, Tt0 = fit
        logger.debug('Fit: slope=%s, b=%s, Tt0=%s' % (slope, b, Tt0))

        self.events = self._correct_events(self.events, slope, b, Tt0)
        return self.events

    def _correct_events(self, events, slope, b, Tt0):
        """
        Apply the relative-time heartbeat correction ``th_rel = slope*tt_rel + b``
        (anchored at ``Tt0`` on the task side and ``eeg_start_ms`` on the host
        side) to the events recarray.

        The event ``mstime`` is already on the host/EEG clock (same clock as
        ``eeg_start_ms``), so we work in host-relative time and the constant ~9 h
        task<->host skew never enters. ``mstime`` and ``eegoffset`` are computed
        INDEPENDENTLY from the continuous fit (mstime is not round-tripped through
        the integer eegoffset, which would inject ~0.25 ms quantization and drop
        the drift from inter-event spacing):

        - ``mstime``    — one formula for every event: the continuous inverse fit
          onto the task clock, ``Tt0 + (mstime - eeg_start_ms - b)/slope``. No
          quantization; inter-event spacing scales by ``1/slope``, following the fit.
        - ``eegoffset`` — task (behavioral) events are slope-fitted
          (``round(slope*(mstime - eeg_start_ms)*sr/1000)``); host (STIM/Elemem-
          originated) events keep the plain host sample (no slope), since they are
          timestamped natively on the host/EEG clock.

        ``*_uncorrected`` columns keep the originals: ``mstime`` untouched and
        ``eegoffset`` computed directly from the uncorrected ``mstime``.
        """
        logger.debug('Applying relative-time clock correction to mstime and eegoffset...')
        M = np.asarray(events["mstime"], dtype=float)
        events["mstime_uncorrected"] = events["mstime"]
        events["eegoffset_uncorrected"] = self._calc_eegoffset(M)

        corrected = events.copy()

        # mstime: continuous inverse fit onto the task clock, computed directly from
        # the fitted line (NOT from the rounded eegoffset). th_rel = slope*tt_rel + b
        # with th_rel = M - eeg_start_ms inverts to tt = Tt0 + (M - eeg_start_ms - b)/slope.
        # Applied to every event; inter-event spacing scales by 1/slope per the fit.
        corrected['mstime'] = Tt0 + (M - self.eeg_start_ms - b) / slope

        # eegoffset: task (behavioral) events are slope-fitted; the constant
        # task<->host skew cancels because we stay relative to eeg_start_ms.
        corrected['eegoffset'] = np.round(
            slope * (M - self.eeg_start_ms) * self.sample_rate / 1000.).astype(int)

        # STIM / Elemem-originated events are timestamped natively on the host (EEG)
        # clock, so their eegoffset is the plain host sample (NOT slope-fitted); their
        # mstime keeps the inverse-fit value, same as every other event.
        locked = self._locked_mask(corrected)
        if locked.any():
            logger.debug('Taking %d host-clock (STIM/Elemem-originated) eegoffsets direct'
                         % int(locked.sum()))
            corrected['eegoffset'][locked] = corrected['eegoffset_uncorrected'][locked]
        return corrected.view(np.recarray)

    def _locked_mask(self, events):
        """Boolean mask of events that must NOT be clock-corrected: STIM events and
        Elemem-originated (host-clock) types for this experiment."""
        deny = {t.upper() for t in _elemem_originated_for(self.experiment)}
        deny |= {'STIM', 'STIM_ON', 'STIM_OFF', 'STIMMING'}
        types_u = np.array([str(t).upper() for t in events['type']])
        return np.isin(types_u, list(deny))

    def _calc_eegoffset(self, mstime):
        return np.round((mstime - self.eeg_start_ms) * self.sample_rate / 1000.).astype(int)


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



def _get_field(d, key):
    """Safely pull ``key`` from a dict-like log payload, else None."""
    return d.get(key) if isinstance(d, dict) else None


def _heartbeat_points_filtered(task_log, host_log, max_one_way_ms=1.0):
    """
    Gather matched task/host HEARTBEAT timestamps for the clock-correction fit,
    keeping only heartbeats whose task-laptop round-trip is tight.

    The task laptop logs both the HEARTBEAT it sends and the HEARTBEAT_OK reply
    from the host; matched by ``count``, the one-way latency is
    ``(time_HEARTBEAT_OK - time_HEARTBEAT) / 2``. Heartbeats with one-way
    latency >= ``max_one_way_ms`` (default 1 ms) are dropped. The host PC's
    event.log provides the corresponding host-clock HEARTBEAT time.

    :return: (task_ms, host_ms) float arrays, matched by ``count`` and ordered
        by count, for heartbeats passing the latency filter and present on both
        clocks.
    """
    # Task laptop: collect inner HEARTBEAT / HEARTBEAT_OK times keyed by count.
    task_hb, task_ok = {}, {}
    for rec in read_jsonl(task_log):
        t = task_type(rec)
        if t not in ('HEARTBEAT', 'HEARTBEAT_OK'):
            continue
        msg = _get_field(rec.get('data'), 'message')
        count = _get_field(_get_field(msg, 'data'), 'count')
        ts = task_time(rec)
        if count is None or ts is None:
            continue
        (task_hb if t == 'HEARTBEAT' else task_ok)[count] = float(ts)

    # Host PC: collect HEARTBEAT times keyed by count (host clock).
    host_hb = {}
    for rec in read_jsonl(host_log):
        if rec.get('type') != 'HEARTBEAT':
            continue
        count = _get_field(rec.get('data'), 'count')
        ts = rec.get('time')
        if count is None or ts is None:
            continue
        host_hb[count] = float(ts)

    task_ms, host_ms = [], []
    for count in sorted(set(task_hb) & set(task_ok) & set(host_hb)):
        one_way = (task_ok[count] - task_hb[count]) / 2.0
        if one_way >= max_one_way_ms:
            continue
        task_ms.append(task_hb[count])
        host_ms.append(host_hb[count])

    return np.array(task_ms, dtype=float), np.array(host_ms, dtype=float)


def _ransac_fit(task_ms, host_ms, residual_threshold):
    """
    RANSAC-fit ``host_time = slope*task_time + offset`` on paired timestamp
    arrays and return ``(slope, offset, n_inliers)``.

    :param residual_threshold: RANSAC inlier threshold in ms. Heartbeats hit a
        sub-ms network floor so a tight (1 ms) threshold is appropriate; the
        all-messages fallback sweeps this from 1 to 5 ms.
    """
    from sklearn.linear_model import LinearRegression, RANSACRegressor
    task_ms = np.asarray(task_ms, dtype=float)
    host_ms = np.asarray(host_ms, dtype=float)
    ransac = RANSACRegressor(estimator=LinearRegression(),
                             residual_threshold=residual_threshold,
                             random_state=0)
    ransac.fit(task_ms.reshape(-1, 1), host_ms)
    n_inliers = int(np.sum(ransac.inlier_mask_))
    return ransac.estimator_.coef_[0], ransac.estimator_.intercept_, n_inliers


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
