"""
System 4 (Elemem) parser for PAL1 and IPAL1.

What it reads
-------------
Elemem's ``event.log`` (JSON lines). Elemem logs every task message with its own receive
time in ``time`` (Unix ms). The PAL1/IPAL1 task also puts the event's own task-clock time
in ``data.task_time_ms``, the list in ``data.trial`` (0 = practice), and sends a HEARTBEAT
with ``data.task_time_ms`` every second. The message vocabulary is docs/ELEMEM_MESSAGES.md
in the task repository.

What it writes
--------------
The legacy PAL event schema (see PALSessionLogParser._pal_fields): STUDY_PAIR, TEST_PROBE,
REC_EVENT, ... with list, serialpos, probepos, study_1, study_2, cue_direction, probe_word,
expecting_word, resp_word, correct, intrusion, resp_pass, vocalization, RT. Practice is
list -1 (the wire sends trial 0) and positions are 1-based (the wire sends 0-based), as in
the System 3 PAL parser.

Timing
------
``mstime`` must be on Elemem's clock, since System4Offset computes
``eegoffset = (mstime - EEGSTART.time) * sfreq / 1000``. Elemem's ``time`` on a message is
the moment it arrived, which lags the screen by the network and by the task's writer
queue. This parser instead maps each event's ``task_time_ms`` onto Elemem's clock with a
linear fit of HEARTBEAT pairs (Elemem receive time vs. task send time), fitted to the
lower envelope so the one-way network latency is removed. Do NOT use
``SESSION.task_epoch_unix_ms + task_time_ms``: that is the task laptop's wall clock, not
the clock the EEG is recorded on. With too few heartbeats the parser falls back to an
offset-only fit, then to Elemem's receive time.

Restarts
--------
A list cut short and run again (LIST_RESTARTED), or a repeated practice
(PRACTICE_REPEATED), sends the list's events twice. Every message after a TRIAL belongs
to that list's attempt until the next TRIAL; for each list only the LAST attempt is kept.
The task moves the earlier attempt's recordings to interrupted/<time>/, so the .ann files
in the session folder belong to the attempt that is kept. A list whose kept attempt never
reached TRIALEND is kept and reported (``self.incomplete_lists``).

Scoring (legacy rules from pal_log_parser.py, with three deliberate differences)
---------------------------------------------------------------------------------
* correct = the response equals the expected word and is not the probe word.
* resp_word / correct / resp_pass on the pair's events come from the LAST
  non-vocalization annotation (legacy behaviour). RT is legacy too: the onset of the
  last annotation whose word differs from the annotation before it.
* vocalizations: '<>' (TotalRecall's intrusion-sound string) and 'V', 'VV', '!'.
  PASS: the word 'PASS'.
Differences from legacy:
  1. Intrusion class does not use the TotalRecall word number (-1 = not in the loaded
     wordpool). It looks the word up among the presented pairs: same list, wrong pair -> 0;
     an earlier list -> lists back (> 0); a later list or never presented -> -1. Legacy
     trusted the number, which depends on which wordpool file the annotator loaded.
  2. The pair-level ``intrusion`` follows the final response (legacy kept the value
     from any earlier wrong word even if the final answer was correct).
  3. The legacy find_presentation passes a list of arrays to np.any without an axis,
     which reduces to one scalar; here the presentation lookup is restricted to
     STUDY_PAIR events as intended.
RT is measured from the probe's own onset:
RT = REC_START.task_time_ms + annotation_ms - TEST_PROBE.task_time_ms.
"""

import codecs
import json
import unicodedata
import warnings
from collections import OrderedDict, defaultdict

import numpy as np

from .base_log_parser import BaseLogParser
from .elemem_parsers import BaseElememLogParser
from ..exc import NoAnnotationError
from ..log import logger


# Quality tests (BaseLogParser.check_event_quality runs these; failures are collected as
# messages in pipeline.importer.tests, they do not stop the import).

def test_one_pair_per_position(events, files):
    pairs = events[(events.type == 'STUDY_PAIR') & (events.list > 0)]
    keys = list(zip(pairs.list.tolist(), pairs.serialpos.tolist()))
    assert len(keys) == len(set(keys)), 'a (list, serialpos) has more than one STUDY_PAIR'


def test_one_probe_per_slot(events, files):
    probes = events[(events.type == 'TEST_PROBE') & (events.list > 0)]
    keys = list(zip(probes.list.tolist(), probes.probepos.tolist()))
    assert len(keys) == len(set(keys)), 'a (list, probepos) has more than one TEST_PROBE'


def test_every_probe_scored(events, files):
    probes = events[(events.type == 'TEST_PROBE') & (events.list > 0)]
    assert (probes.correct != -999).all(), '%d probes have no score (missing .ann?)' % (
        (probes.correct == -999).sum())


def test_session_length(events, files):
    assert (events.type == 'TRIAL').sum() <= 26, 'Session contains more than 26 lists'


class ElememPALLogParser(BaseElememLogParser):
    """Builds PAL1 / IPAL1 events from a System 4 event.log. See the module docstring."""

    # Raise if a real list's probe has no .ann file. Practice probes only warn.
    STRICT_ANNOTATIONS = True
    # Fewer heartbeats than this: offset-only clock fit.
    MIN_HEARTBEATS_FOR_SLOPE = 30
    # Iterative inlier fit of the clock: residual band around the median (ms).
    CLOCK_INLIER_MS = 3.0
    # A mapped event time later than Elemem's receive time by more than this (ms) means
    # the fit is wrong; reported and the receive time is used instead.
    CLOCK_MAX_LEAD_MS = 5.0
    # Name practice events PRACTICE_PAIR / PRACTICE_PROBE (System 1/2 style) instead of
    # STUDY_PAIR / TEST_PROBE with list -1 (System 3 style).
    PRACTICE_TYPE_NAMES = False
    # Drop a list whose last attempt in this event.log never reached TRIALEND. Needed
    # when a session is split over two Elemem folders and each is imported on its own:
    # the first folder ends mid-list, and that list's .ann files belong to the rerun.
    DROP_INCOMPLETE_LISTS = False

    VOCALIZATIONS = frozenset(['<>', 'V', 'VV', '<VV>', '!'])
    PASS_WORDS = frozenset(['PASS'])

    # Wire type (lower case) -> event type.
    TYPE_NAMES = {
        'start': 'SESS_START',            # Elemem's answer to READY
        'trial': 'TRIAL',
        'countdown': 'COUNTDOWN_START',
        'countdown_end': 'COUNTDOWN_END',
        'encoding': 'ENCODING_START',
        'encoding_end': 'ENCODING_END',
        'study_pair': 'STUDY_PAIR',
        'pair_off': 'STUDY_PAIR_OFF',
        'recall': 'RECALL_START',
        'test_probe': 'TEST_PROBE',
        'probe_off': 'PROBE_END',
        'rec_start': 'REC_START',
        'rec_end': 'REC_END',
        'trialend': 'RECALL_END',
        'exit': 'SESS_END',
        'list_restarted': 'LIST_RESTARTED',
        'practice_repeated': 'PRACTICE_REPEATED',
    }
    ORIENT_NAMES = {'pair': 'STUDY_ORIENT', 'probe': 'TEST_ORIENT', 'recall': 'RECALL_ORIENT'}
    # DISTRACT data.type -> (start, end). PAL1 sends no type (arithmetic).
    DISTRACT_NAMES = {'fixation': ('FIXATION_START', 'FIXATION_END'),
                      None: ('MATH_START', 'MATH_END')}

    # Messages that are not part of any list attempt (never dropped by the restart rule).
    # The restart markers are kept too (they are logged before the rerun's TRIAL, so
    # they would otherwise fall inside the attempt being dropped).
    _SESSION_LEVEL = frozenset(['ELEMEM', 'EEGSTART', 'CONNECTED', 'CONNECTED_OK',
                                'CONFIGURE', 'CONFIGURE_OK', 'READY', 'START', 'SESSION',
                                'HEARTBEAT', 'HEARTBEAT_OK', 'EXIT', 'LIST_RESTARTED',
                                'PRACTICE_REPEATED'])

    @classmethod
    def _pal_fields(cls):
        # The legacy PAL fields, with strings widened to U32 and RT to int32.
        return (
            ('list', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('probepos', -999, 'int16'),
            ('study_1', '', 'U32'),
            ('study_2', '', 'U32'),
            ('cue_direction', -999, 'int16'),
            ('probe_word', '', 'U32'),
            ('expecting_word', '', 'U32'),
            ('resp_word', '', 'U32'),
            ('correct', -999, 'int16'),
            ('intrusion', -999, 'int16'),
            ('resp_pass', 0, 'int16'),
            ('vocalization', -999, 'int16'),
            ('RT', -999, 'int32'),
            ('exp_version', '', 'U32'),
            ('stim_type', '', 'U16'),
            ('stim_list', False, 'b1'),
            ('is_stim', False, 'b1'),
        )

    _TESTS = [test_session_length, test_one_pair_per_position, test_one_probe_per_slot,
              test_every_probe_scored]

    def __init__(self, protocol, subject, montage, experiment, session, files):
        # BaseLogParser.__init__ calls self._read_primary_log(), which does the reading,
        # the clock fit and the restart pruning and sets attributes; nothing set below
        # may overwrite them.
        BaseElememLogParser.__init__(self, protocol, subject, montage, experiment, session,
                                     files, primary_log='event_log', include_stim_params=False)
        self._add_fields(*self._pal_fields())

        self._list = -999
        self._phase = ''
        self._orient_scope = None
        self._distract_kind = None
        self._version = ''

        # (list, serialpos) -> (word1, word2); (list, probepos) -> probe record
        self._pairs = {}
        self._probes = OrderedDict()

        # Replace the base handlers: this parser registers its own 'start'.
        self._type_to_new_event = {}
        self._type_to_modify_events = {}
        self._add_type_to_new_event(
            start=self.event_simple,
            configure=self.event_configure,
            session=self.event_session,
            trial=self.event_trial,
            countdown=self.event_simple,
            countdown_end=self.event_simple,
            encoding=self.event_simple,
            encoding_end=self.event_simple,
            orient=self.event_orient,
            orient_off=self.event_orient_off,
            study_pair=self.event_study_pair,
            pair_off=self.event_pair_off,
            distract=self.event_distract,
            distract_end=self.event_distract_end,
            recall=self.event_simple,
            test_probe=self.event_test_probe,
            probe_off=self.event_probe_marker,
            rec_start=self.event_rec_start,
            rec_end=self.event_probe_marker,
            trialend=self.event_simple,
            list_restarted=self.event_simple,
            practice_repeated=self.event_simple,
            exit=self.event_simple,
            math=self._event_skip,     # MathElememLogParser makes math_events
            heartbeat=self._event_skip,
        )

    def _read_primary_log(self):
        log = self._primary_log
        if isinstance(log, (list, tuple)):
            # BaseElememLogParser._read_primary_log uses hasattr(x, '__iter__'), which is
            # also true of a str and would take its first character.
            log = log[0]
        self._primary_log = log
        messages = self._read_jsonl(log)
        self.clock_fits = self._fit_clocks(messages)
        messages = self._drop_superseded_attempts(messages)
        return messages

    @staticmethod
    def _read_jsonl(filename):
        out = []
        with codecs.open(filename, encoding='utf-8') as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    warnings.warn('event.log line %d is not JSON; skipped' % n)
                    continue
                if not isinstance(msg, dict) or 'type' not in msg:
                    continue
                if not isinstance(msg.get('data'), dict):
                    msg['data'] = {}
                msg['_line'] = n
                out.append(msg)
        return out

    def _fit_clocks(self, messages):
        """Map task_time_ms onto Elemem's clock, one fit per task connection.

        The task restarts its stopwatch when it is relaunched, so each CONNECTED starts
        a new segment with its own fit. Within a segment, HEARTBEAT pairs
        (task send time, Elemem receive time) are fitted with a line; the fit is moved
        down to the 5th percentile of the inlier residuals, i.e. to the fastest
        deliveries, which removes the one-way latency.
        """
        segments = []
        current = []
        for m in messages:
            # Elemem's own lines before the first CONNECTED join the first segment
            if m['type'] == 'CONNECTED' and any(x['type'] == 'CONNECTED' for x in current):
                segments.append(current)
                current = []
            current.append(m)
        if current:
            segments.append(current)

        fits = []
        for seg in segments:
            hb = [(float(m['data']['task_time_ms']), float(m['time'])) for m in seg
                  if m['type'] == 'HEARTBEAT' and 'task_time_ms' in m['data'] and 'time' in m]
            anyt = [(float(m['data']['task_time_ms']), float(m['time'])) for m in seg
                    if 'task_time_ms' in m['data'] and 'time' in m]
            fit = self._fit_line(hb, anyt)
            fits.append(fit)
            for m in seg:
                tt = m['data'].get('task_time_ms')
                if fit['method'] != 'receive_time' and tt is not None:
                    mapped = fit['offset'] + fit['slope'] * float(tt)
                    lead = mapped - float(m['time'])
                    if lead > self.CLOCK_MAX_LEAD_MS:
                        fit['n_lead_violations'] += 1
                        m['_mstime'] = float(m['time'])
                    else:
                        m['_mstime'] = mapped
                    m['_receive_lag_ms'] = -lead
                else:
                    m['_mstime'] = float(m['time'])
        return fits

    def _fit_line(self, hb, anyt):
        fit = dict(method='receive_time', slope=1.0, offset=0.0, n=0, n_inliers=0,
                   rms_ms=np.nan, n_lead_violations=0)
        if len(hb) >= self.MIN_HEARTBEATS_FOR_SLOPE:
            x = np.array([p[0] for p in hb])
            y = np.array([p[1] for p in hb])
            keep = np.ones(len(x), dtype=bool)
            for _ in range(5):
                slope, offset = np.polyfit(x[keep], y[keep], 1)
                resid = y - (slope * x + offset)
                band = max(self.CLOCK_INLIER_MS, 3 * np.median(np.abs(
                    resid[keep] - np.median(resid[keep]))))
                new_keep = np.abs(resid - np.median(resid[keep])) <= band
                if new_keep.sum() < 10 or (new_keep == keep).all():
                    break
                keep = new_keep
            slope, offset = np.polyfit(x[keep], y[keep], 1)
            resid = y[keep] - (slope * x[keep] + offset)
            offset += np.percentile(resid, 5)
            fit.update(method='heartbeat_fit', slope=float(slope), offset=float(offset),
                       n=len(x), n_inliers=int(keep.sum()),
                       rms_ms=float(np.sqrt(np.mean(resid ** 2))))
        elif anyt:
            d = np.array([p[1] - p[0] for p in anyt])
            fit.update(method='offset_only', offset=float(np.percentile(d, 5)), n=len(d))
        return fit

    def _drop_superseded_attempts(self, messages):
        """Keep, for each list, only the messages of its last attempt."""
        attempts = defaultdict(list)       # trial -> [attempt index, ...]
        owner = []                         # per message: attempt index or None
        attempt_trial = []
        current = None
        for m in messages:
            t = m['type']
            if t == 'TRIAL':
                current = len(attempt_trial)
                trial = int(m['data'].get('trial', -999))
                attempt_trial.append(trial)
                attempts[trial].append(current)
            if t in self._SESSION_LEVEL:
                owner.append(None)
            else:
                owner.append(current)
            if t in ('EXIT', 'CONNECTED'):
                current = None

        dropped = set()
        for trial, idxs in attempts.items():
            dropped.update(idxs[:-1])
        self.dropped_attempts = sorted(
            (attempt_trial[i], i) for i in dropped)

        # A kept attempt that never reached TRIALEND
        ended = set(owner[i] for i, m in enumerate(messages) if m['type'] == 'TRIALEND')
        self.incomplete_lists = sorted(attempt_trial[i] for trial, idxs in attempts.items()
                                       for i in idxs[-1:] if i not in ended)
        for trial in self.incomplete_lists:
            warnings.warn('list %d never reached TRIALEND (session stopped mid-list?)%s'
                          % (trial, '; dropped' if self.DROP_INCOMPLETE_LISTS else ''))
        if self.DROP_INCOMPLETE_LISTS:
            dropped.update(i for trial, idxs in attempts.items()
                           for i in idxs[-1:] if i not in ended)

        return [m for m, o in zip(messages, owner) if o is None or o not in dropped]

    @staticmethod
    def _norm(word):
        """Upper case, accents stripped: the form annotations are written in."""
        word = unicodedata.normalize('NFKD', str(word or ''))
        word = ''.join(c for c in word if not unicodedata.combining(c))
        return word.strip().upper()

    def event_default(self, msg):
        event = self._empty_event
        event.mstime = int(round(msg.get('_mstime', msg['time'])))
        event.type = self.TYPE_NAMES.get(msg['type'].lower(), msg['type'])
        event.list = self._list
        event.session = self._session
        event.phase = self._phase
        event.exp_version = self._version
        return event

    def event_simple(self, msg):
        return self.event_default(msg)

    def event_configure(self, msg):
        d = msg['data']
        if d.get('experiment') and d['experiment'] != self._experiment:
            warnings.warn('event.log CONFIGURE experiment %s != %s' % (d['experiment'], self._experiment))
        if d.get('subject') and d['subject'] != self._subject:
            warnings.warn('event.log CONFIGURE subject %s != %s' % (d['subject'], self._subject))
        return False

    def event_session(self, msg):
        s = msg['data'].get('session')
        if s is not None and int(s) != int(self._session):
            # Not fatal: the database session number need not equal the task's index
            # (e.g. a second montage). Recorded so a mismatch is visible.
            logger.info('task SESSION %s, importing as session %s' % (s, self._session))
        return False

    def event_trial(self, msg):
        trial = int(msg['data']['trial'])
        self._list = -1 if trial == 0 else trial
        self._phase = 'PRACTICE' if trial == 0 else 'ENCODING'
        event = self.event_default(msg)
        event.stim_list = bool(msg['data'].get('stim', False))
        return event

    def event_orient(self, msg):
        d = msg['data']
        self._orient_scope = d.get('scope')
        event = self.event_default(msg)
        event.type = self.ORIENT_NAMES.get(self._orient_scope, 'ORIENT')
        if self._orient_scope == 'pair' and 'serialpos' in d:
            event.serialpos = int(d['serialpos']) + 1
        elif self._orient_scope == 'probe' and 'slot' in d:
            event.probepos = int(d['slot']) + 1
        return event

    def event_orient_off(self, msg):
        event = self.event_default(msg)
        event.type = self.ORIENT_NAMES.get(self._orient_scope, 'ORIENT') + '_OFF'
        return event

    def event_study_pair(self, msg):
        d = msg['data']
        sp = int(d['serialpos']) + 1
        w1, w2 = self._norm(d['word1']), self._norm(d['word2'])
        self._pairs[(self._list, sp)] = (w1, w2)
        event = self.event_default(msg)
        if self._list == -1 and self.PRACTICE_TYPE_NAMES:
            event.type = 'PRACTICE_PAIR'
        event.serialpos = sp
        event.study_1, event.study_2 = w1, w2
        event.correct = 0
        return event

    def event_pair_off(self, msg):
        sp = int(msg['data']['serialpos']) + 1
        event = self.event_default(msg)
        if self._list == -1 and self.PRACTICE_TYPE_NAMES:
            event.type = 'PRACTICE_PAIR_OFF'
        event.serialpos = sp
        event.study_1, event.study_2 = self._pairs.get((self._list, sp), ('', ''))
        return event

    def event_distract(self, msg):
        self._distract_kind = msg['data'].get('type')
        names = self.DISTRACT_NAMES.get(self._distract_kind, self.DISTRACT_NAMES[None])
        self._phase = 'PRACTICE' if self._list == -1 else 'DISTRACT'
        event = self.event_default(msg)
        event.type = names[0]
        return event

    def event_distract_end(self, msg):
        kind = msg['data'].get('type', self._distract_kind)
        names = self.DISTRACT_NAMES.get(kind, self.DISTRACT_NAMES[None])
        event = self.event_default(msg)
        event.type = names[1]
        if self._list != -1:
            self._phase = 'RETRIEVAL'
        return event

    def event_test_probe(self, msg):
        d = msg['data']
        pp = int(d['probepos']) + 1
        sp = int(d['serialpos']) + 1
        probe, expecting = self._norm(d['probe']), self._norm(d['expecting'])
        direction = int(d['direction'])
        study_1, study_2 = (expecting, probe) if direction == 1 else (probe, expecting)
        studied = self._pairs.get((self._list, sp))
        if studied is not None and studied != (study_1, study_2):
            warnings.warn('list %d probe %d: TEST_PROBE words %s do not match STUDY_PAIR %s'
                          % (self._list, pp, (study_1, study_2), studied))
        # update, not replace: a REC_START logged first must not be lost
        rec = self._probes.setdefault((self._list, pp), dict(stem=None, rec_task_ms=None,
                                                             rec_mstime=None))
        rec.update(list=self._list, probepos=pp, serialpos=sp, probe=probe,
                   expecting=expecting, direction=direction, study_1=study_1,
                   study_2=study_2, probe_task_ms=d.get('task_time_ms'),
                   probe_mstime=msg.get('_mstime'))
        event = self.event_default(msg)
        if self._list == -1 and self.PRACTICE_TYPE_NAMES:
            event.type = 'PRACTICE_PROBE'
        event.probepos, event.serialpos = pp, sp
        event.probe_word, event.expecting_word = probe, expecting
        event.cue_direction = direction
        event.study_1, event.study_2 = study_1, study_2
        event.correct = 0
        return event

    def event_rec_start(self, msg):
        d = msg['data']
        pp = int(d.get('probepos', -1)) + 1
        stem = d.get('file') or ''
        listno = 0 if self._list == -1 else self._list
        if stem and stem != '%d_%d' % (listno, pp - 1):
            warnings.warn('REC_START file %s is not %d_%d' % (stem, listno, pp - 1))
        rec = self._probes.setdefault((self._list, pp), dict(
            list=self._list, probepos=pp, serialpos=-999, probe='', expecting='',
            direction=-999, study_1='', study_2='', probe_task_ms=None, probe_mstime=None))
        rec.update(rec_task_ms=d.get('task_time_ms'), rec_mstime=msg.get('_mstime'),
                   stem=stem or '%d_%d' % (listno, pp - 1),
                   rec_receive_time=float(msg['time']))
        event = self.event_default(msg)
        event.probepos = pp
        return event

    def event_probe_marker(self, msg):
        event = self.event_default(msg)
        if 'probepos' in msg['data']:
            event.probepos = int(msg['data']['probepos']) + 1
        return event

    def parse(self):
        # BaseLogParser.parse, not BaseElememLogParser.parse: the latter's error handler
        # (traceback.print_exc(exc), an unimported `logger`, `exc.message`) fails itself,
        # so it replaces any parse error with an unrelated one.
        events = BaseLogParser.parse(self)
        if events.ndim == 0 or len(events) == 0:
            return events
        events = events.view(np.recarray)
        self._backfill(events)
        events = self._score_all(events)
        return events

    def _backfill(self, events):
        # Study orient: the words of the pair it precedes.
        for (lst, sp), (w1, w2) in self._pairs.items():
            m = (events.list == lst) & (events.serialpos == sp) & \
                (events.type == self.ORIENT_NAMES['pair'])
            events.study_1[m] = w1
            events.study_2[m] = w2
        # Probe-level fields onto every event of that probe and of its pair (as the
        # legacy modify_test does).
        for (lst, pp), p in self._probes.items():
            if p['serialpos'] == -999:
                # REC_START with no TEST_PROBE: no pair to back-fill (reported when scoring)
                continue
            by_slot = (events.list == lst) & (events.probepos == pp)
            events.serialpos[by_slot] = p['serialpos']
            pair = (events.list == lst) & (events.serialpos == p['serialpos'])
            for mask in (by_slot, pair):
                events.probepos[mask] = pp
                events.probe_word[mask] = p['probe']
                events.expecting_word[mask] = p['expecting']
                events.cue_direction[mask] = p['direction']
                events.study_1[mask] = p['study_1']
                events.study_2[mask] = p['study_2']

    def _presented_lists(self, word):
        return sorted(set(lst for (lst, _), pair in self._pairs.items() if word in pair))

    def _intrusion(self, word, lst):
        """0 same list (wrong pair), >0 lists back, -1 later list or never presented.

        Practice (list -1) counts as list 0 when counting lists back.
        """
        here = 0 if lst == -1 else lst
        earlier = [0 if l == -1 else l for l in self._presented_lists(word)]
        earlier = [l for l in earlier if l <= here]
        if not earlier:
            return -1
        return here - max(earlier)

    def _map_time(self, rec, offset_ms):
        """Elemem-clock time of a point offset_ms after the recording started."""
        if rec.get('rec_task_ms') is not None and rec.get('rec_mstime') is not None:
            # The fitted map is linear with slope ~1: shift the mapped REC_START time.
            return rec['rec_mstime'] + offset_ms
        return rec['rec_receive_time'] + offset_ms

    def _score_all(self, events):
        new_events = []
        for (lst, pp), p in self._probes.items():
            if p.get('stem') is None:
                warnings.warn('list %d probe %d has no REC_START; not scored' % (lst, pp))
                continue
            if p['serialpos'] == -999:
                warnings.warn('list %d probe %d has no TEST_PROBE; not scored' % (lst, pp))
                continue
            try:
                anns = self._parse_ann_file(p['stem'])
            except NoAnnotationError:
                if lst == -1 or not self.STRICT_ANNOTATIONS:
                    warnings.warn('no %s.ann; list %d probe %d left unscored' % (p['stem'], lst, pp))
                    pair = (events.list == lst) & (events.serialpos == p['serialpos'])
                    events.correct[pair] = -999
                    continue
                raise

            pair = (events.list == lst) & (events.serialpos == p['serialpos']) & \
                   (events.type != 'REC_EVENT')
            events.correct[pair] = 0
            events.resp_pass[pair] = 0

            prev_word = None
            for rectime, _wordno, raw in anns:
                word = self._norm(raw)
                voc = word in self.VOCALIZATIONS
                is_pass = word in self.PASS_WORDS
                correct = int(word == p['expecting'] and word != p['probe'])
                if p.get('probe_task_ms') is not None and p.get('rec_task_ms') is not None:
                    rt = int(round(float(p['rec_task_ms']) + rectime - float(p['probe_task_ms'])))
                else:
                    rt = int(round(rectime))

                ev = self._empty_event
                ev.type = 'REC_EVENT'
                ev.mstime = int(round(self._map_time(p, rectime)))
                ev.msoffset = 20
                ev.list = lst
                ev.session = self._session
                ev.phase = 'PRACTICE' if lst == -1 else 'RETRIEVAL'
                ev.serialpos, ev.probepos = p['serialpos'], pp
                ev.probe_word, ev.expecting_word = p['probe'], p['expecting']
                ev.cue_direction = p['direction']
                ev.study_1, ev.study_2 = p['study_1'], p['study_2']
                ev.resp_word = word
                ev.RT = rt
                ev.correct = correct
                ev.resp_pass = int(is_pass)
                ev.vocalization = int(voc)
                ev.intrusion = 0 if (voc or is_pass or correct) else self._intrusion(word, lst)
                new_events.append(ev)

                if not voc:
                    events.vocalization[pair] = 0
                    events.resp_word[pair] = word
                    events.correct[pair] = correct
                    events.resp_pass[pair] = int(is_pass)
                    events.intrusion[pair] = ev.intrusion
                    if word != prev_word:
                        events.RT[pair] = rt
                prev_word = word

        if new_events:
            events = np.concatenate([events] + [np.atleast_1d(e) for e in new_events]
                                    ).view(np.recarray)
        return events
