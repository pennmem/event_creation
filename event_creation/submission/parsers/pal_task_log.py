"""
Reading the PAL1/IPAL1 task's own log, ``pal_events.jsonl``.

The task (pennmem PAL1_UnityEPL) writes one JSON object per line::

    {"type": "TEST_PROBE", "time": 123456, "unixMs": 1759780123456, "data": {...}}

* ``time`` is milliseconds on a stopwatch that restarts at every launch of the task;
* every launch begins with ``LAUNCH {"epochUnixMs", "utc"}`` (the file is opened for
  append, so a resumed session holds several launches); ``epochUnixMs + time`` is the
  event's time on the task computer's wall clock, and ``unixMs`` is the same sum;
* with a sync box (System 1) every pulse is logged as
  ``syncPulse {"index", "targetMs", "sendDoneMs", "intervalMs", "launch"}``, where ``time``
  is the stopwatch just before the pulse was sent and ``launch`` the 1-based LAUNCH count.

Both the System 1 parser (PALTaskLogParser) and the System 1 aligner
(TaskLogSystem1Aligner) read the file through ``read_task_log`` so they agree on what a
launch is and on the absolute time of every line: ``_abs = epochUnixMs + time`` of the
launch the line belongs to. The parser uses ``_abs`` as ``mstime``; the aligner fits each
launch's pulses separately and finds an event's launch from its ``mstime``.

``to_wire_messages`` turns the task's event names into the System 4 wire vocabulary that
the task sends Elemem (the task's ``ElememReporter.Map``: TRIAL{listno} -> TRIAL{trial},
TEST_PROBE{slot, expected} -> TEST_PROBE{probepos, expecting}, RECALL_END -> TRIALEND ...),
so the System 1 and System 4 parsers share every handler.
"""

import codecs
import json
import warnings

LAUNCH = 'LAUNCH'
SYNC_PULSE = 'syncPulse'


def read_task_log(filename):
    """Read pal_events.jsonl into launches.

    :return: a list of dicts, one per launch, in file order::

        {'launch': 1-based LAUNCH count (0 for lines before any LAUNCH),
         'epoch_unix_ms': the launch's stopwatch zero on the wall clock,
         'epoch_source': 'LAUNCH', 'unixMs' or 'none' (times are then relative),
         'messages': [{'type', 'time', 'data', '_abs', '_launch', '_line'}, ...],
         'first_abs', 'last_abs': the range of _abs in the launch}
    """
    if isinstance(filename, (list, tuple)):
        filename = filename[0]
    launches = []
    current = None
    n_launch = 0
    with codecs.open(filename, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                warnings.warn('pal_events.jsonl line %d is not JSON; skipped' % n)
                continue
            if not isinstance(msg, dict) or 'type' not in msg or 'time' not in msg:
                continue
            if not isinstance(msg.get('data'), dict):
                msg['data'] = {}
            msg['_line'] = n
            if msg['type'] == LAUNCH:
                n_launch += 1
                epoch = msg['data'].get('epochUnixMs')
                source = 'LAUNCH'
                if epoch is None and msg.get('unixMs') is not None:
                    epoch, source = float(msg['unixMs']) - float(msg['time']), 'unixMs'
                current = dict(launch=n_launch, epoch_unix_ms=epoch, epoch_source=source,
                               messages=[])
                launches.append(current)
            elif current is None:
                # Lines before any LAUNCH (an older build, or a log made by a test runner)
                current = dict(launch=0, epoch_unix_ms=None, epoch_source='none', messages=[])
                launches.append(current)
            if current['epoch_unix_ms'] is None and msg.get('unixMs') is not None:
                current['epoch_unix_ms'] = float(msg['unixMs']) - float(msg['time'])
                current['epoch_source'] = 'unixMs'
            current['messages'].append(msg)

    for launch in launches:
        if launch['epoch_unix_ms'] is None:
            warnings.warn('launch %d of %s has no LAUNCH epoch or unixMs; its times are '
                          'relative to the launch' % (launch['launch'], filename))
            launch['epoch_unix_ms'] = 0.0
        epoch = float(launch['epoch_unix_ms'])
        for msg in launch['messages']:
            msg['_abs'] = epoch + float(msg['time'])
            msg['_launch'] = launch['launch']
        times = [m['_abs'] for m in launch['messages']]
        launch['first_abs'], launch['last_abs'] = min(times), max(times)
    return launches


def _int(data, key, default=None):
    v = data.get(key)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _slot_from_stem(stem):
    """'8_3' is list 8, slot 3; -1 if it is not in that form."""
    stem = stem or ''
    _, _, slot = stem.rpartition('_')
    try:
        return int(slot) if stem.count('_') else -1
    except ValueError:
        return -1


def _copy(src, dst, *keys):
    for k in keys:
        if k in src:
            dst[k] = src[k]


def to_wire_messages(launches):
    """Task-log lines -> System 4 wire messages (what ElememReporter.Map sends Elemem).

    Each message is ``{'type', 'data', 'time', '_mstime', '_launch', '_line'}`` with
    ``time = _mstime = epochUnixMs + time``. Besides the reporter's mapping, three session
    markers are translated so the restart rule and the session events match System 4:
    LAUNCH -> CONNECTED (a new connection: no list attempt is open), SESSION_RESUME ->
    START (SESS_START; logged once per launch, where Elemem answers each connection's
    READY), SESSION_END -> EXIT (SESS_END). Lines the reporter does not forward (PRESENT,
    HEARTBEAT_RTT, OPERATOR_PROMPT, MIC_*, the task's own REC_EVENT, syncPulse ...) are
    dropped.
    """
    out = []
    trial = 0
    for launch in launches:
        for m in launch['messages']:
            t, data = m['type'], m['data']
            d = {}
            if t == 'TRIAL':
                trial = _int(data, 'listno', trial)
                wire = 'TRIAL'
                d.update(trial=trial, stim=False, phase_type=data.get('type', ''))
            elif t == 'COUNTDOWN_START':
                wire = 'COUNTDOWN'
            elif t == 'COUNTDOWN_END':
                wire = 'COUNTDOWN_END'
            elif t == 'ENCODING_START':
                wire = 'ENCODING'
            elif t == 'ENCODING_END':
                wire = 'ENCODING_END'
            elif t == 'ORIENT':
                wire = 'ORIENT'
                _copy(data, d, 'scope', 'text', 'serialpos', 'slot')
            elif t == 'ORIENT_OFF':
                wire = 'ORIENT_OFF'
            elif t == 'STUDY_PAIR':
                wire = 'STUDY_PAIR'
                d.update(serialpos=_int(data, 'serialpos'), word1=data.get('word1', ''),
                         word2=data.get('word2', ''), stim=False)
            elif t == 'PAIR_OFF':
                wire = 'PAIR_OFF'
                d.update(serialpos=_int(data, 'serialpos'))
            elif t == 'DISTRACT_START':
                wire = 'DISTRACT'
                _copy(data, d, 'type', 'min_duration_ms', 'duration_ms', 'practice')
            elif t == 'MATH':
                if data.get('response') in (None, ''):
                    continue    # not forwarded: the math parser reads responses as integers
                wire = 'MATH'
                correct = data.get('correct')
                if isinstance(correct, str):
                    correct = correct.lower() == 'true'
                d.update(problem=data.get('problem', ''), response=str(data.get('response')),
                         response_time_ms=_int(data, 'response_time_ms', 0),
                         correct='True' if correct else 'False')
            elif t == 'DISTRACT_END':
                wire = 'DISTRACT_END'
                _copy(data, d, 'type', 'problems', 'correct')
            elif t == 'RECALL_START':
                wire = 'RECALL'
                d.update(duration=0)
            elif t == 'TEST_PROBE':
                wire = 'TEST_PROBE'
                d.update(probepos=_int(data, 'slot'), serialpos=_int(data, 'serialpos'),
                         probe=data.get('probe', ''), expecting=data.get('expected', ''),
                         direction=_int(data, 'direction'), stim=False)
            elif t == 'PROBE_OFF':
                wire = 'PROBE_OFF'
                d.update(probepos=_int(data, 'slot'))
            elif t in ('REC_START', 'REC_END'):
                wire = t
                d.update(file=data.get('file', ''), probepos=_slot_from_stem(data.get('file')))
            elif t == 'RECALL_END':
                wire = 'TRIALEND'
            elif t in ('LIST_RESTARTED', 'PRACTICE_REPEATED'):
                wire = t
            elif t == LAUNCH:
                wire = 'CONNECTED'
                d.update(launch=launch['launch'])
            elif t == 'SESSION_RESUME':
                wire = 'START'
            elif t == 'SESSION_END':
                wire = 'EXIT'
            else:
                continue
            if wire not in ('CONNECTED', 'START', 'EXIT'):
                if 'trial' not in d:
                    d['trial'] = _int(data, 'listno', trial)
                d['task_time_ms'] = float(m['time'])
            out.append(dict(type=wire, data=d, time=m['_abs'], _mstime=m['_abs'],
                            _launch=m['_launch'], _line=m['_line']))
    return out
