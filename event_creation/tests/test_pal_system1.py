"""
Tests for PAL1/IPAL1 on System 1 (sync box, no host PC): the task-log parser
(PALTaskLogParser), the sync-pulse extractor, the per-launch aligner
(TaskLogSystem1Aligner), the PAL1 task-log math parser, and the routing/registry.

The session is synthetic. One timeline of task events is written three ways:

* pal_events.jsonl as the task writes it: two launches (LAUNCH lines, the stopwatch back
  at 0), practice, PRACTICE_REPEATED and practice again, list 1, list 2 cut off mid-probe
  by a quit, then after the relaunch LIST_RESTARTED, list 2, list 3 and SESSION_END; a
  syncPulse line for every pulse (first 1000 ms after the launch, then every 800-1200 ms),
  plus lines the parser must ignore (PRESENT, HEARTBEAT_RTT, OPERATOR_PROMPT, the task's
  own REC_EVENT, SYNCBOX_*);
* an Elemem event.log of the same session (what the task would have sent on System 4),
  for comparing the two parsers;
* one or two EDF files (pyedflib) whose sync channels carry 20 ms pulses at the task's
  pulse times mapped through a known clock: a per-launch offset, 35 ppm drift and 0.5 ms
  jitter, after a 0.5 Hz high-pass, with 8 "Test Syncbox" pulses before the session that
  the task log does not have.
"""
import datetime
import glob
import json
import os
import random
import tempfile
import warnings

import numpy as np
import pytest

from ..submission.configuration import paths

# log.py creates <db_root>/protocols when it is first imported
if not os.path.isdir(os.path.join(paths.db_root, 'protocols')):
    paths.set('db_root', tempfile.mkdtemp())

from ..submission.exc import AlignmentError
from ..submission.alignment.system1_tasklog import TaskLogSystem1Aligner
from ..submission.parsers.elemem_pal_parser import ElememPALLogParser
from ..submission.parsers.pal_tasklog_parser import PALTaskLogParser, PALSystem1Parser
from ..submission.parsers.math_parser import MathLogParser, MathPALTaskLogParser
from ..submission.readers.sync_pulse_extractor import extract_sync_pulses
from .test_elemem_pal_parser import (EXPECTED, SCENARIOS, ANN_HEADER, PRACTICE_WORDS, WORDS)

pyedflib = pytest.importorskip('pyedflib')

SUBJECT = 'R1999A'
DRIFT = 35e-6           # EEG clock runs 35 ppm fast against the task's stopwatch
JITTER_MS = 0.5         # SD of the pulse edge on the EEG
SAMPLE_RATE = 1000
EEG_START = 2.0e6       # EEG ms of the first sample of the first file (arbitrary)
EPOCHS = (1759780000000, None)   # wall-clock zero of each launch's stopwatch
SYNC_LABELS = ('C129', 'C130')


class Timeline(object):
    """Task events on each launch's stopwatch, and the EEG time of every stopwatch ms."""

    def __init__(self, experiment, seed):
        self.experiment = experiment
        self.rng = random.Random(seed)
        self.launches = []      # dicts: epoch, eeg0, lines, end
        self.t = 0.0

    def launch(self, epoch, eeg0):
        self.launches.append(dict(epoch=epoch, eeg0=eeg0, lines=[], end=0.0))
        self.t = 0.0
        self.log('LAUNCH', {'epochUnixMs': epoch, 'utc': 'x'})
        self.log('SESSION_RESUME', {'decision': 'New', 'sessionIndex': 0, 'nextListIndex': 0})
        self.log('SYNCBOX_OPENED', {'helper': '127.0.0.1:8903'})
        self.advance(1500)

    def eeg_ms(self, launch, t):
        L = self.launches[launch]
        return L['eeg0'] + (1 + DRIFT) * t

    def log(self, type_, data=None, t=None):
        L = self.launches[-1]
        t = int(self.t if t is None else t)
        L['lines'].append(dict(type=type_, time=t, data=data or {}))

    def advance(self, ms):
        self.t += ms
        self.launches[-1]['end'] = self.t

    def noise(self):
        self.log('PRESENT', {'frame': self.rng.randint(0, 9999)})
        self.log('HEARTBEAT_RTT', {'count': 1, 'rttMs': 1.2})

    def run_list(self, L, pairs, order, stop_after_slot=None):
        rng = self.rng
        self.log('OPERATOR_PROMPT', {'prompt': 'list_gate', 'listno': L, 'key': 'Return'})
        self.advance(500)
        self.log('TRIAL', {'listno': L, 'type': 'PRACTICE' if L == 0 else 'NON-STIM', 'session': 0})
        self.log('COUNTDOWN_START', {'listno': L})
        self.advance(3000)
        self.log('COUNTDOWN_END', {'listno': L})
        self.log('ENCODING_START', {'listno': L})
        for sp, (w1, w2) in enumerate(pairs):
            self.log('ORIENT', {'scope': 'pair', 'text': '', 'serialpos': sp})
            self.advance(250)
            self.log('ORIENT_OFF')
            self.advance(rng.uniform(500, 750))
            self.log('STUDY_PAIR', {'listno': L, 'serialpos': sp, 'word1': w1, 'word2': w2,
                                    'type': 'PRACTICE' if L == 0 else 'NON-STIM'})
            self.noise()
            self.advance(4000)
            self.log('PAIR_OFF', {'serialpos': sp})
            self.advance(1000)
        self.log('ENCODING_END', {'listno': L})
        if self.experiment == 'IPAL1':
            self.log('DISTRACT_START', {'type': 'fixation', 'text': '+', 'duration_ms': 10000,
                                        'listno': L})
            self.advance(10000)
            self.log('DISTRACT_END', {'type': 'fixation', 'elapsed_ms': 10000, 'listno': L})
        else:
            self.log('DISTRACT_START', {'min_duration_ms': 20000, 'practice': L == 0, 'listno': L})
            for _ in range(4):
                a, b, c = [rng.randint(1, 9) for _ in range(3)]
                rt = rng.randint(1500, 5000)
                self.advance(rt)
                self.log('MATH', {'listno': L, 'problem': '%d + %d + %d = ' % (a, b, c),
                                  'response': str(a + b + c), 'response_time_ms': rt,
                                  'correct': True})
            self.log('DISTRACT_END', {'problems': 4, 'correct': 4, 'listno': L})
        self.log('RECALL_START', {'listno': L})
        self.log('ORIENT', {'scope': 'recall', 'text': '*******', 'beep': True})
        self.advance(500)
        self.log('ORIENT_OFF')
        for slot, (sp, direction) in enumerate(order):
            w1, w2 = pairs[sp]
            probe, expected = (w2, w1) if direction == 1 else (w1, w2)
            self.log('ORIENT', {'scope': 'probe', 'text': '?????', 'slot': slot})
            self.advance(250)
            self.log('ORIENT_OFF')
            self.advance(rng.uniform(500, 750))
            self.log('TEST_PROBE', {'listno': L, 'slot': slot, 'serialpos': sp, 'probe': probe,
                                    'expected': expected, 'direction': direction})
            self.log('REC_START', {'file': '%d_%d' % (L, slot)})
            if slot == stop_after_slot:
                self.advance(1500)      # the task is quit mid-probe
                return
            self.advance(4000)
            self.log('PROBE_OFF', {'slot': slot})
            self.advance(1000)
            self.log('REC_END', {'file': '%d_%d' % (L, slot)})
            self.log('REC_EVENT', {'listno': L, 'slot': slot, 'response': '', 'correct': False})
        self.log('RECALL_END', {'listno': L})
        self.advance(3000)

    def add_pulses(self):
        """syncPulse lines for each launch; returns the true EEG ms of every edge."""
        edges = []
        for k, L in enumerate(self.launches):
            t, index = 1000.0, 0
            while t < L['end'] - 200:
                L['lines'].append(dict(type='syncPulse', time=int(t), data=dict(
                    index=index, targetMs=round(t, 3), sendDoneMs=round(t + 0.3, 3),
                    intervalMs=0, launch=k + 1)))
                edges.append(self.eeg_ms(k, t) + self.rng.gauss(0, JITTER_MS))
                t += self.rng.uniform(800, 1200)
                index += 1
        return np.array(edges)


def make_lists(rng):
    def make(words):
        pairs = [(words[2 * i], words[2 * i + 1]) for i in range(6)]
        order = list(range(6))
        rng.shuffle(order)
        return pairs, [(sp, rng.randint(0, 1)) for sp in order]
    lists = {0: make(PRACTICE_WORDS)}
    for L in (1, 2, 3):
        lists[L] = make(WORDS[(L - 1) * 12: L * 12])
    return lists


def write_ann_files(out, lists):
    scenarios = {}
    for n, (L, (pairs, order)) in enumerate(sorted(lists.items())):
        for slot, (sp, direction) in enumerate(order):
            w1, w2 = pairs[sp]
            probe, expecting = (w2, w1) if direction == 1 else (w1, w2)
            other = pairs[(sp + 1) % 6][0]
            scen = 'pli' if (L, slot) == (3, 0) else SCENARIOS[(n * 6 + slot) % len(SCENARIOS)]
            entries = {
                'correct': [(1234.5, expecting)],
                'voc_then_correct': [(800.0, '<>'), (1500.25, expecting)],
                'wrong_pair': [(1100.0, other)],
                'pass': [(900.0, 'PASS')],
                'empty': [],
                'xli': [(700.0, 'ZEBRAFISH')],
                'probe_word': [(650.0, probe)],
                'correct_twice': [(1000.0, expecting), (1300.0, expecting)],
                'wrong_then_correct': [(1000.0, other), (2000.0, expecting)],
                'pli': [(1200.0, lists[1][0][0][0])],
            }[scen]
            with open(os.path.join(out, '%d_%d.ann' % (L, slot)), 'w') as f:
                f.write(ANN_HEADER)
                for t, word in entries:
                    f.write('%.3f\t%d\t%s\n' % (t, -1 if word == '<>' else 1, word))
            scenarios[(L, slot)] = dict(scen=scen, entries=entries)
    return scenarios


def write_task_log(path, timeline):
    with open(path, 'w') as f:
        for L in timeline.launches:
            for line in sorted(L['lines'], key=lambda l: l['time']):    # stable: events keep order
                f.write(json.dumps(dict(type=line['type'], time=line['time'],
                                        unixMs=L['epoch'] + line['time'], data=line['data'])) + '\n')


# What the task sends Elemem (its ElememReporter.Map), written independently of
# pal_task_log.to_wire_messages so the two parsers are compared on their own terms.
REPORTER = {
    'TRIAL': ('TRIAL', lambda d: dict(trial=d['listno'], stim=False, phase_type=d['type'])),
    'COUNTDOWN_START': ('COUNTDOWN', None), 'COUNTDOWN_END': ('COUNTDOWN_END', None),
    'ENCODING_START': ('ENCODING', None), 'ENCODING_END': ('ENCODING_END', None),
    'ORIENT': ('ORIENT', lambda d: {k: d[k] for k in ('scope', 'text', 'serialpos', 'slot') if k in d}),
    'ORIENT_OFF': ('ORIENT_OFF', None),
    'STUDY_PAIR': ('STUDY_PAIR', lambda d: dict(serialpos=d['serialpos'], word1=d['word1'],
                                               word2=d['word2'], stim=False)),
    'PAIR_OFF': ('PAIR_OFF', lambda d: dict(serialpos=d['serialpos'])),
    'DISTRACT_START': ('DISTRACT', lambda d: {k: d[k] for k in ('type', 'min_duration_ms', 'duration_ms',
                                                                'practice') if k in d}),
    'MATH': ('MATH', lambda d: dict(problem=d['problem'], response=d['response'],
                                    response_time_ms=d['response_time_ms'],
                                    correct='True' if d['correct'] else 'False')),
    'DISTRACT_END': ('DISTRACT_END', lambda d: {k: d[k] for k in ('type', 'problems', 'correct') if k in d}),
    'RECALL_START': ('RECALL', lambda d: dict(duration=0)),
    'TEST_PROBE': ('TEST_PROBE', lambda d: dict(probepos=d['slot'], serialpos=d['serialpos'],
                                               probe=d['probe'], expecting=d['expected'],
                                               direction=d['direction'], stim=False)),
    'PROBE_OFF': ('PROBE_OFF', lambda d: dict(probepos=d['slot'])),
    'REC_START': ('REC_START', lambda d: dict(file=d['file'], probepos=int(d['file'].split('_')[1]))),
    'REC_END': ('REC_END', lambda d: dict(file=d['file'], probepos=int(d['file'].split('_')[1]))),
    'RECALL_END': ('TRIALEND', None),
    'LIST_RESTARTED': ('LIST_RESTARTED', None), 'PRACTICE_REPEATED': ('PRACTICE_REPEATED', None),
}


def write_event_log(folder, timeline):
    """The same session as an Elemem event.log (System 4), Elemem clock = task clock + 1 ms."""
    lines, trial = [], 0
    for k, L in enumerate(timeline.launches):
        e0 = 1.76e12 + k * 1e7

        def at(t):
            return e0 + (1 + DRIFT) * t
        lines.append(dict(type='CONNECTED', data={}, time=at(0)))
        lines.append(dict(type='START', data={}, time=at(0) + 1))
        for t in np.arange(0, L['end'], 1000.):
            lines.append(dict(type='HEARTBEAT', data={'count': 0, 'task_time_ms': float(t)}, time=at(t) + 1))
        for line in sorted(L['lines'], key=lambda l: l['time']):
            if line['type'] == 'SESSION_END':
                lines.append(dict(type='EXIT', data={}, time=at(line['time']) + 1))
                continue
            if line['type'] not in REPORTER:
                continue
            wire, f = REPORTER[line['type']]
            d = f(line['data']) if f else {}
            if wire == 'TRIAL':
                trial = d['trial']
            d.setdefault('trial', line['data'].get('listno', trial))
            d['task_time_ms'] = line['time']
            lines.append(dict(type=wire, data=d, time=at(line['time']) + 1))
    os.makedirs(folder)
    with open(os.path.join(folder, 'event.log'), 'w') as f:
        for m in sorted(lines, key=lambda m: m['time']):
            f.write(json.dumps(m) + '\n')
    with open(os.path.join(folder, 'experiment_config.json'), 'w') as f:
        json.dump({'experiment': {'type': timeline.experiment}}, f)


def write_edf(path, n_samples, edges_samples, rng, negative=False, start=None):
    """EEG noise channels, the two sync inputs, and an EKG channel."""
    t = np.zeros(n_samples)
    width = int(round(0.020 * SAMPLE_RATE))
    for s in edges_samples:
        if 0 <= s < n_samples:
            t[int(s):int(s) + width] = 800.
    # first-order 0.5 Hz high-pass, as a clinical amplifier would apply
    alpha = 1. / (1. + 2 * np.pi * 0.5 / SAMPLE_RATE)
    hp = np.zeros_like(t)
    for i in range(1, len(t)):
        hp[i] = alpha * (hp[i - 1] + t[i] - t[i - 1])
    sync = -hp if negative else hp
    signals = [rng.normal(0, 30, n_samples), rng.normal(0, 30, n_samples),
               sync / 2 + rng.normal(0, 3, n_samples), -sync / 2 + rng.normal(0, 3, n_samples),
               rng.normal(0, 100, n_samples)]
    labels = ['LA1', 'LA2', 'EEG %s-Ref' % SYNC_LABELS[0], 'EEG %s-Ref' % SYNC_LABELS[1], 'EKG1']
    w = pyedflib.EdfWriter(path, len(labels), file_type=pyedflib.FILETYPE_EDFPLUS)
    if start is not None:
        # the split names its files by start minute: files of one session must differ
        w.setStartdatetime(start)
    w.setSignalHeaders([dict(label=l, dimension='uV', sample_frequency=SAMPLE_RATE,
                             physical_max=3000, physical_min=-3000, digital_max=32767,
                             digital_min=-32768) for l in labels])
    w.writeSamples(signals)
    w.close()


def make_session(out, experiment, n_files=1, seed=11):
    """Write a synthetic System 1 session to out/; return what the tests need to check it."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    tl = Timeline(experiment, seed)
    lists = make_lists(rng)

    # launch 1 begins 30 s into the recording; launch 2 a minute after launch 1 ended
    tl.launch(EPOCHS[0], EEG_START + 30000.)
    tl.run_list(0, *lists[0])
    tl.log('PRACTICE_REPEATED', {'listno': 0})
    tl.run_list(0, *lists[0])
    tl.run_list(1, *lists[1])
    tl.run_list(2, *lists[2], stop_after_slot=2)
    end1 = tl.t
    tl.launch(EPOCHS[0] + int(end1) + 60000, tl.eeg_ms(0, end1) + 60000.)
    tl.log('LIST_RESTARTED', {'listno': 2, 'movedTo': 'interrupted/x', 'files': '2_0,2_1,2_2'})
    tl.run_list(2, *lists[2])
    tl.run_list(3, *lists[3])
    tl.log('SESSION_END', {'listsRun': 4})
    tl.advance(500)
    edges = tl.add_pulses()
    # "Test Syncbox" before the session: pulses on the EEG that the task log does not have
    test_pulses = EEG_START + 10000. + np.cumsum(np_rng.uniform(800, 1200, 8))
    eeg_edges = np.sort(np.r_[test_pulses, edges])

    sess = os.path.join(out, 'behavioral', experiment, 'session_0')
    raw = os.path.join(out, 'raw', '%s_0' % experiment)
    os.makedirs(sess)
    os.makedirs(raw)
    write_task_log(os.path.join(sess, 'pal_events.jsonl'), tl)
    scenarios = write_ann_files(sess, lists)
    write_event_log(os.path.join(out, 'elemem', SUBJECT + '_x'), tl)
    with open(os.path.join(raw, 'sync_channel.json'), 'w') as f:
        json.dump({'channels': list(SYNC_LABELS)}, f)

    end_ms = tl.eeg_ms(1, tl.t) + 10000.
    total = int((end_ms - EEG_START) * SAMPLE_RATE / 1000.)
    # several files are contiguous, as Nihon Kohden splits them; the boundary is inside launch 2
    bounds = [0, total] if n_files == 1 else \
        [0, int((tl.eeg_ms(1, 60000) - EEG_START) * SAMPLE_RATE / 1000.), total]
    files = []
    for i in range(n_files):
        name = '%s_%s_0_part%d.edf' % (SUBJECT, experiment, i + 1) if n_files > 1 else \
            '%s_%s_0.edf' % (SUBJECT, experiment)
        start_ms = EEG_START + bounds[i] * 1000. / SAMPLE_RATE
        n = bounds[i + 1] - bounds[i]
        samples = (eeg_edges - start_ms) * SAMPLE_RATE / 1000.
        start = datetime.datetime(2026, 10, 8, 9, 0, 0) + \
            datetime.timedelta(seconds=int((start_ms - EEG_START) / 1000.))
        write_edf(os.path.join(raw, name), n, np.ceil(samples), np_rng, negative=(i == 1),
                  start=start)
        files.append(dict(name=name, start_ms=start_ms, n_samples=n,
                          edges=np.sort(samples[(samples >= 0) & (samples < n)]),
                          key='%s_%s_0_part%d' % (SUBJECT, experiment, i + 1)))
    return dict(timeline=tl, scenarios=scenarios, sess=sess, raw=raw, files=files,
                edges=edges, out=out, experiment=experiment)


def write_sources(path, session, sample_rate=SAMPLE_RATE):
    with open(path, 'w') as f:
        json.dump({fl['key']: dict(name=fl['key'], source_file=fl['name'], sample_rate=sample_rate,
                                   n_samples=fl['n_samples'], data_format='int16',
                                   start_time_ms=0, start_time_str='')
                   for fl in session['files']}, f)


def parse(files, experiment, parser_class=PALTaskLogParser):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        parser = parser_class('r1', SUBJECT, '0.0', experiment, 0, files)
        events = parser.parse()
    return parser, events, [str(w.message) for w in caught]


def behavioral_files(session):
    return {'session_log': os.path.join(session['sess'], 'pal_events.jsonl'),
            'annotations': sorted(glob.glob(os.path.join(session['sess'], '*.ann')))}


def extract(session):
    outs = []
    for fl in session['files']:
        s = extract_sync_pulses(os.path.join(session['raw'], fl['name']))
        outs.append(s)
    return outs


def truth_samples(session, events):
    """True sample (file key, index) of every event, from its task-clock mstime."""
    tl = session['timeline']
    epochs = [L['epoch'] for L in tl.launches]
    out = []
    for e in events:
        k = int(np.searchsorted(epochs, e['mstime'], side='right') - 1)
        eeg = tl.eeg_ms(k, e['mstime'] - epochs[k])
        for fl in session['files']:
            s = (eeg - fl['start_ms']) * SAMPLE_RATE / 1000.
            if 0 <= s < fl['n_samples']:
                out.append((fl['key'], s))
                break
        else:
            out.append(('', -1))
    return out


@pytest.fixture(scope='module', params=[('IPAL1', 1), ('PAL1', 1), ('IPAL1', 2)],
                ids=['IPAL1-1file', 'PAL1-1file', 'IPAL1-2files'])
def session(request, tmp_path_factory):
    experiment, n_files = request.param
    out = str(tmp_path_factory.mktemp('%s_%d' % (experiment, n_files)))
    s = make_session(out, experiment, n_files)
    s['extracted'] = extract(s)
    s['sources'] = os.path.join(out, 'sources.json')
    write_sources(s['sources'], s)
    s['files_dict'] = dict(behavioral_files(s), eeg_sources=s['sources'],
                           sync_pulses=[x['out'] for x in s['extracted']])
    s['parser'], s['events'], s['warnings'] = parse(s['files_dict'], experiment)
    return s


# ---- the parser ------------------------------------------------------------------------

def test_lists_restarts_and_launches(session):
    ev, parser = session['events'], session['parser']
    assert ev[ev.type == 'TRIAL'].list.tolist() == [-1, 1, 2, 3]
    assert [t for t, _ in parser.dropped_attempts] == [0, 2]
    assert parser.incomplete_lists == []
    assert (ev.type == 'SESS_START').sum() == 2 and (ev.type == 'SESS_END').sum() == 1
    assert [l['launch'] for l in parser.launches] == [1, 2]
    pairs, probes = ev[ev.type == 'STUDY_PAIR'], ev[ev.type == 'TEST_PROBE']
    assert len(pairs) == 24 and len(probes) == 24
    for lst in (-1, 1, 2, 3):
        assert sorted(probes[probes.list == lst].probepos.tolist()) == [1, 2, 3, 4, 5, 6]
    # mstime = LAUNCH.epochUnixMs + time: the rerun of list 2 is in launch 2
    epoch2 = session['timeline'].launches[1]['epoch']
    assert (ev[ev.list == 3].mstime >= epoch2).all()
    assert (ev[(ev.type == 'TRIAL') & (ev.list == 1)].mstime < epoch2).all()
    assert parser.check_event_quality(parser.clean_events(ev), session['files_dict']) == []


def test_scoring(session):
    ev = session['events']
    seen = set()
    for probe in ev[ev.type == 'TEST_PROBE']:
        L = 0 if probe.list == -1 else probe.list
        scenario = session['scenarios'][(L, probe.probepos - 1)]
        want = EXPECTED[scenario['scen']]
        seen.add(scenario['scen'])
        for field in ('correct', 'resp_pass', 'intrusion'):
            assert probe[field] == want[field], (scenario['scen'], field)
        assert abs(probe.RT - want['RT']) <= 1, scenario['scen']
        recs = ev[(ev.type == 'REC_EVENT') & (ev.list == probe.list) & (ev.probepos == probe.probepos)]
        assert len(recs) == len(scenario['entries'])
    assert seen == set(EXPECTED)


def test_retention_interval_events(session):
    ev = session['events']
    if session['experiment'] == 'IPAL1':
        assert ev[ev.type == 'FIXATION_START'].list.tolist() == [-1, 1, 2, 3]
        assert not np.isin(ev.type, ['MATH_START', 'MATH_END']).any()
    else:
        assert ev[ev.type == 'MATH_START'].list.tolist() == [-1, 1, 2, 3]
        assert not np.isin(ev.type, ['FIXATION_START', 'FIXATION_END']).any()


def test_same_events_as_system4_parser(session):
    """Same types, order and fields as ElememPALLogParser on the event.log of the session."""
    el = glob.glob(os.path.join(session['out'], 'elemem', '*'))[0]
    files4 = {'event_log': [os.path.join(el, 'event.log')],
              'experiment_config': os.path.join(el, 'experiment_config.json'),
              'annotations': session['files_dict']['annotations']}
    _, ev4, _ = parse(files4, session['experiment'], ElememPALLogParser)
    ev1 = session['events']
    assert ev1.dtype == ev4.dtype
    assert ev1.type.tolist() == ev4.type.tolist()
    for field in ev1.dtype.names:
        if field in ('mstime',):
            continue
        assert ev1[field].tolist() == ev4[field].tolist(), field


def test_ignores_task_only_lines(session):
    ev = session['events']
    # the task's own REC_EVENT lines carry no word; only .ann REC_EVENTs are made
    n_ann = sum(len(s['entries']) for s in session['scenarios'].values())
    assert (ev.type == 'REC_EVENT').sum() == n_ann
    assert not np.isin(ev.type, ['PRESENT', 'HEARTBEAT_RTT', 'OPERATOR_PROMPT', 'syncPulse',
                                 'SYNCBOX_OPENED', 'LAUNCH']).any()


# ---- pulse extraction ------------------------------------------------------------------

def test_extracted_pulses(session):
    for fl, s in zip(session['files'], session['extracted']):
        found = np.loadtxt(s['out'], ndmin=1)
        assert s['polarity'] == ('negative' if fl['key'].endswith('part2') else 'positive')
        assert len(found) == len(fl['edges'])
        # the first sample at/after the rising edge: within a sample of the true edge
        assert np.abs(found - fl['edges']).max() <= 1.0
        assert 18 <= s['width_ms'] <= 22


# ---- alignment -------------------------------------------------------------------------

def align(session, events=None, files=None, aligner=TaskLogSystem1Aligner):
    events = session['events'].copy() if events is None else events
    a = aligner(events, files or session['files_dict'])
    return a, a.align()


def test_alignment_every_event(session):
    a, ev = align(session)
    truth = truth_samples(session, ev)
    keys = [k for k, _ in truth]
    assert ev.eegfile.tolist() == keys
    err = ev.eegoffset - np.array([s for _, s in truth])
    assert (ev.eegoffset >= 0).all()
    assert np.abs(err).max() < 2, np.abs(err).max()
    # one fit per launch and file the launch overlaps
    n_files = len(session['files'])
    assert sorted((f['launch'], f['source']) for f in a.fits) == (
        [(1, session['files'][0]['key']), (2, session['files'][0]['key'])] if n_files == 1 else
        [(1, session['files'][0]['key']), (2, session['files'][0]['key']),
         (2, session['files'][1]['key'])])
    for f in a.fits:
        assert abs(f['slope'] - (1 + DRIFT)) < 2e-5    # short fits (2 min) are ~7 ppm off
        assert f['max_residual_ms'] < 3 and f['n_outliers'] == 0
        assert f['matched_fraction'] == 1.0


def test_alignment_robust_to_missing_pulses(session, tmp_path):
    # 5 % of the EEG pulses lost
    rng = np.random.default_rng(3)
    syncs = []
    for s in session['extracted']:
        idx = np.loadtxt(s['out'], ndmin=1)
        keep = rng.random(len(idx)) > 0.05
        p = str(tmp_path / os.path.basename(s['out']))
        np.savetxt(p, idx[keep], fmt='%d')
        syncs.append(p)
    a, ev = align(session, files=dict(session['files_dict'], sync_pulses=syncs))
    err = ev.eegoffset - np.array([s for _, s in truth_samples(session, ev)])
    assert np.abs(err).max() < 2


def corrupt(session, tmp_path, fn):
    syncs = []
    for s in session['extracted']:
        idx = np.loadtxt(s['out'], ndmin=1)
        p = str(tmp_path / os.path.basename(s['out']))
        np.savetxt(p, fn(idx), fmt='%.3f')
        syncs.append(p)
    return dict(session['files_dict'], sync_pulses=syncs)


def test_gate_residual(session, tmp_path):
    rng = np.random.default_rng(4)
    files = corrupt(session, tmp_path, lambda idx: idx + rng.normal(0, 4, len(idx)))
    with pytest.raises(AlignmentError, match='more than 5 ms off the fit'):
        align(session, files=files)


def test_gate_slope(session, tmp_path):
    # sync indices written at 1015 Hz while sources.json says 1000 Hz
    files = corrupt(session, tmp_path, lambda idx: idx * 1.015)
    with pytest.raises(AlignmentError, match='slope'):
        align(session, files=files)


def test_gate_matched_fraction(session, tmp_path):
    rng = np.random.default_rng(5)
    files = corrupt(session, tmp_path, lambda idx: idx[rng.random(len(idx)) > 0.3])
    with pytest.raises(AlignmentError, match='matched'):
        align(session, files=files)


def test_unrelated_pulses_do_not_align(session, tmp_path):
    rng = np.random.default_rng(6)
    files = corrupt(session, tmp_path,
                    lambda idx: idx[0] + np.cumsum(rng.uniform(800, 1200, len(idx))))
    with pytest.raises(AlignmentError, match='matches the pulses of no|No launch'):
        align(session, files=files)


def test_events_outside_the_recording(session, tmp_path):
    # the recording cut 90 s into launch 2: later events get -1
    tl = session['timeline']
    last = session['files'][-1]
    cut = int((tl.eeg_ms(1, 90000) - last["start_ms"]) * SAMPLE_RATE / 1000.)
    if cut <= 0:
        pytest.skip('launch 2 at 90 s is not in the last file')
    sources = json.load(open(session['sources']))
    sources[last['key']]['n_samples'] = cut
    path = str(tmp_path / 'sources.json')
    json.dump(sources, open(path, 'w'))
    idx = np.loadtxt(session['extracted'][-1]['out'], ndmin=1)
    p = str(tmp_path / os.path.basename(session['extracted'][-1]['out']))
    np.savetxt(p, idx[idx < cut], fmt='%d')
    syncs = [x['out'] for x in session['extracted'][:-1]] + [p]
    a, ev = align(session, files=dict(session['files_dict'], eeg_sources=path, sync_pulses=syncs))
    late = ev.mstime > tl.launches[1]['epoch'] + 90000
    early = ev.mstime < tl.launches[1]['epoch'] + 89000
    assert late.sum() > 50 and early.sum() > 50
    assert (ev.eegoffset[late] == -1).all() and (ev.eegfile[late] == '').all()
    assert (ev.eegoffset[early] >= 0).all()


def test_launch_lines_without_epoch_use_unixms(tmp_path):
    from ..submission.parsers.pal_task_log import read_task_log
    p = str(tmp_path / 'pal_events.jsonl')
    with open(p, 'w') as f:
        f.write(json.dumps(dict(type='LAUNCH', time=5, unixMs=1005, data={})) + '\n')
        f.write(json.dumps(dict(type='TRIAL', time=10, unixMs=1010, data={'listno': 1})) + '\n')
    launches = read_task_log(p)
    assert launches[0]['epoch_unix_ms'] == 1000 and launches[0]['messages'][1]['_abs'] == 1010


# ---- math (PAL1) -----------------------------------------------------------------------

def test_math_from_task_log(session):
    if session['experiment'] != 'PAL1':
        pytest.skip('IPAL1 has no math')
    files = session['files_dict']
    parser = MathLogParser('r1', SUBJECT, '0.0', 'PAL1', 0, files)
    assert isinstance(parser, MathPALTaskLogParser)
    math = parser.parse()
    distract = math[math['type'] == 'DISTRACT_START']
    assert distract['list'].tolist() == [-1, -1, 1, 2, 2, 3]
    probs = math[math['type'] == 'PROB']
    assert len(probs) == 24 and (probs['iscorrect'] == 1).all()
    # aligned per launch like the task events
    ev = TaskLogSystem1Aligner(math, files).align()
    assert (ev['eegoffset'] >= 0).all()
    err = ev['eegoffset'] - np.array([s for _, s in truth_samples(session, ev)])
    assert np.abs(err).max() < 2


# ---- routing and registry --------------------------------------------------------------

def test_registry():
    from ..submission.events_tasks import EventCreationTask
    from ..submission.parsers.pal_log_parser import PALSessionLogParser
    assert EventCreationTask.R1_PARSERS(1.0)['IPAL'] is PALTaskLogParser
    assert EventCreationTask.R1_PARSERS(1.0)['PAL'] is PALSystem1Parser
    assert PALTaskLogParser.SYSTEM1_ALIGNER is TaskLogSystem1Aligner
    # System 4 is unchanged
    assert EventCreationTask.R1_PARSERS(4.0)['IPAL'] is ElememPALLogParser


def test_pal_system1_dispatch(session):
    parser = PALSystem1Parser('r1', SUBJECT, '0.0', 'PAL1', 0, session['files_dict'])
    assert isinstance(parser, PALTaskLogParser)


@pytest.fixture
def data_root(tmp_path):
    """A data_root with the Colorado layout for R1999A IPAL1 and PAL1 session 0."""
    old = paths.data_root
    root = tmp_path / 'data' / 'eeg'
    for exp in ('IPAL1', 'PAL1'):
        sess = root / SUBJECT / 'behavioral' / exp / 'session_0'
        raw = root / SUBJECT / 'raw' / ('%s_0' % exp)
        sess.mkdir(parents=True)
        raw.mkdir(parents=True)
        (sess / 'pal_events.jsonl').write_text('{}\n')
        (sess / '0_0.ann').write_text('')
        (sess / '0_0.wav').write_text('')
        (raw / ('%s_%s_0.edf' % (SUBJECT, exp))).write_text('')
        (raw / ('%s_%s_0.sync.txt' % (SUBJECT, exp))).write_text('1\n')
        (raw / 'sync_channel.json').write_text('{"channel": "C129"}')
    paths.set('data_root', str(root))
    yield str(root)
    paths.set('data_root', old)


def test_routing(data_root):
    from ..submission.pipelines import determine_groups
    from ..submission.transfer_inputs import TRANSFER_INPUTS
    from ..submission.transfer_config import TransferConfig
    beh = TRANSFER_INPUTS['behavioral']
    # Unchanged without an explicit choice: suffix A is System 3.3
    groups = determine_groups('r1', SUBJECT, 'IPAL1', 0, beh, 'transfer')
    assert 'system_3_3' in groups and 'system_1' not in groups
    # --sys1 (or a JSON import with "system_1": true) puts system_1 in the groups first
    for exp in ('IPAL1', 'PAL1'):
        groups = determine_groups('r1', SUBJECT, exp, 0, beh, 'transfer', 'system_1')
        assert 'system_1' in groups and 'system_3_3' not in groups and 'system_4' not in groups
        assert 'pal_task_log' in groups and 'verbal' in groups
        cfg = TransferConfig(beh, groups, protocol='r1', subject=SUBJECT, code=SUBJECT,
                             experiment=exp, new_experiment=exp, original_session=0, session=0,
                             localization=0, montage_num=0, db_root=paths.db_root,
                             data_root=data_root, events_root='', ram_experiment='RAM_' + exp)
        cfg.locate_origin_files()
        names = lambda n: [os.path.basename(p) for p in cfg.get_file(n).origin_paths]
        assert names('session_log') == ['pal_events.jsonl']
        assert names('sync_pulses') == ['%s_%s_0.sync.txt' % (SUBJECT, exp)]
        assert names('annotations') == ['0_0.ann'] and names('sound_files') == ['0_0.wav']
        missing = [f.name for f in cfg.missing_required_files()]
        assert missing == ['contacts'], missing     # the montage is not in this tmp root
    # Subject J (System 4) with --sys1 is System 1 too; without it, System 4 as before
    groups = determine_groups('r1', 'R1999J', 'IPAL1', 0, beh, 'transfer')
    assert 'system_4' in groups and 'system_1' not in groups


def test_sys1_option():
    from ..submission.configuration import config
    assert 'sys1' in config.options
