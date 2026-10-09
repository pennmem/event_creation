"""
System 1 parser for PAL1 and IPAL1: events from the task's own ``pal_events.jsonl``.

Under System 1 (a sync box, no host PC; Colorado's set-up) there is no Elemem event.log:
the only record of the session is the task's log, ``pal_events.jsonl`` in the session
folder (format: pal_task_log.py). This parser builds from it exactly the events that the
System 4 parser (ElememPALLogParser) builds from event.log, with the same types and
fields: practice is list -1, positions are 1-based, the last attempt of a list is kept
after LIST_RESTARTED / PRACTICE_REPEATED, IPAL1's retention interval is FIXATION_START /
FIXATION_END and PAL1's MATH_START / MATH_END, and every probe is scored from its
``<list>_<slot>.ann`` with REC_EVENTs made from the annotation lines. It does so by turning
the task's lines into the wire messages the task would have sent Elemem
(``pal_task_log.to_wire_messages``) and running the shared PALParserMixin (pal_base.py);
see elemem_pal_parser.py for the scoring rules.

Timing
------
``mstime`` is the event's time on the task computer: the launch's ``LAUNCH.epochUnixMs``
plus the line's ``time`` (the task's stopwatch, which restarts at each launch). It is NOT
on the EEG clock: TaskLogSystem1Aligner (alignment/system1_tasklog.py) maps it to EEG
samples with a separate fit for each launch, from the ``syncPulse`` lines of the same log
and the pulses extracted from the clinical EEG. The launch an event belongs to is not
stored in the events (the field set is System 4's); the aligner recovers it from mstime,
since each launch's absolute times start at its own epoch. ``self.launches`` lists the
launches the parser saw.

Differences from the System 4 output
------------------------------------
* mstime is on the task clock (before alignment), not Elemem's;
* SESS_START comes from each launch's SESSION_RESUME line, where System 4 has one per
  connection to Elemem; SESS_END from SESSION_END (Elemem: EXIT);
* the task's own REC_EVENT lines (its on-line scoring) are ignored, as Elemem never sees
  them; REC_EVENTs come from the .ann files, as on System 4.

Files
-----
``files['session_log']``: pal_events.jsonl; ``files['annotations']``: the .ann files.
"""

import warnings

from .base_log_parser import BaseLogParser
from .pal_base import PALParserMixin
from .pal_task_log import read_task_log, to_wire_messages


class PALTaskLogParser(PALParserMixin, BaseLogParser):
    """PAL1 / IPAL1 events from pal_events.jsonl. See the module docstring."""

    def __init__(self, protocol, subject, montage, experiment, session, files):
        BaseLogParser.__init__(self, protocol, subject, montage, experiment, session, files,
                               primary_log='session_log', allow_unparsed_events=True,
                               include_stim_params=False)
        self._init_pal()

    def _read_primary_log(self):
        log = self._primary_log
        if isinstance(log, (list, tuple)):
            log = log[0]
        self._primary_log = log
        launches = read_task_log(log)
        self.launches = [dict(launch=l['launch'], epoch_unix_ms=l['epoch_unix_ms'],
                              epoch_source=l['epoch_source'], first_abs=l['first_abs'],
                              last_abs=l['last_abs'], n_lines=len(l['messages']))
                         for l in launches]
        messages = to_wire_messages(launches)
        if not any(m['type'] == 'TRIAL' for m in messages):
            warnings.warn('%s has no TRIAL lines' % log)
        return self._drop_superseded_attempts(messages)


def is_task_log(filename):
    """True if a session_log is the PAL task's pal_events.jsonl (not a PyEPL session.log)."""
    if isinstance(filename, (list, tuple)):
        filename = filename[0] if filename else ''
    return str(filename).endswith('.jsonl')


def PALSystem1Parser(protocol, subject, montage, experiment, session, files):
    """System 1 PAL1: the task-log parser for pal_events.jsonl, else the PyEPL parser.

    The same indirection as MathLogParser / PSLogParser: PAL1 sessions recorded with the
    PyEPL task (session.log) keep PALSessionLogParser; sessions run with the UnityEPL task
    transfer pal_events.jsonl as session_log (group 'pal_task_log', see
    pipelines.determine_groups) and get PALTaskLogParser.
    """
    if is_task_log(files.get('session_log', '')):
        return PALTaskLogParser(protocol, subject, montage, experiment, session, files)
    from .pal_log_parser import PALSessionLogParser
    return PALSessionLogParser(protocol, subject, montage, experiment, session, files)
