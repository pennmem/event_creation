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

import warnings

import numpy as np

from .elemem_parsers import BaseElememLogParser
# The quality tests are re-exported for code that imported them from here.
from .pal_base import (PALParserMixin, test_session_length, test_one_pair_per_position,
                       test_one_probe_per_slot, test_every_probe_scored)
from ..log import logger


class ElememPALLogParser(PALParserMixin, BaseElememLogParser):
    """Builds PAL1 / IPAL1 events from a System 4 event.log. See the module docstring.

    The message handling, restart rule and scoring are PALParserMixin's (pal_base.py),
    shared with the System 1 task-log parser; this class reads event.log and maps the
    task clock onto Elemem's.
    """

    # Fewer heartbeats than this: offset-only clock fit.
    MIN_HEARTBEATS_FOR_SLOPE = 30
    # Iterative inlier fit of the clock: residual band around the median (ms).
    CLOCK_INLIER_MS = 3.0
    # A mapped event time later than Elemem's receive time by more than this (ms) means
    # the fit is wrong; reported and the receive time is used instead.
    CLOCK_MAX_LEAD_MS = 5.0

    def __init__(self, protocol, subject, montage, experiment, session, files):
        # BaseLogParser.__init__ calls self._read_primary_log(), which does the reading,
        # the clock fit and the restart pruning and sets attributes; nothing set below
        # may overwrite them.
        BaseElememLogParser.__init__(self, protocol, subject, montage, experiment, session,
                                     files, primary_log='event_log', include_stim_params=False)
        self._init_pal()
        self._add_type_to_new_event(
            configure=self.event_configure,
            session=self.event_session,
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
