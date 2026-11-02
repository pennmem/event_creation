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
``eegoffset = (mstime - EEGSTART.time) * sfreq / 1000``. As in BaseElememLogParser
(FR, catFR, ...), ``mstime`` is Elemem's ``time`` on the message: the moment it arrived.
That lags the screen by the task's send queue and the network. ``task_time_ms`` is used
only for differences on the task's own clock (RT). Do NOT use
``SESSION.task_epoch_unix_ms + task_time_ms`` as ``mstime``: that is the task laptop's
wall clock, not the clock the EEG is recorded on.

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

from .elemem_parsers import BaseElememLogParser
# The quality tests are re-exported for code that imported them from here.
from .pal_base import (PALParserMixin, test_session_length, test_one_pair_per_position,
                       test_one_probe_per_slot, test_every_probe_scored)
from ..log import logger


class ElememPALLogParser(PALParserMixin, BaseElememLogParser):
    """Builds PAL1 / IPAL1 events from a System 4 event.log. See the module docstring.

    The message handling, restart rule and scoring are PALParserMixin's (pal_base.py),
    shared with the System 1 task-log parser; this class reads event.log and gives every
    message Elemem's receive time.
    """

    def __init__(self, protocol, subject, montage, experiment, session, files):
        # BaseLogParser.__init__ calls self._read_primary_log(), which does the reading
        # and the restart pruning and sets attributes; nothing set below may overwrite
        # them.
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
        for m in messages:
            m['_mstime'] = float(m['time'])
        messages = self._drop_superseded_attempts(messages)
        return messages

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
