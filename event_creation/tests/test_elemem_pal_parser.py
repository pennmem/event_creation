"""
Tests for the System 4 PAL1/IPAL1 parser (ElememPALLogParser), the System 4 math parser's
list numbering, and the System 4 aligner.

The sessions are synthetic: an Elemem event.log built from the messages the PAL1/IPAL1
task sends, with clock drift between the task and Elemem, network latency and send
queueing, plus one .ann file per probe. One session runs:
  practice, PRACTICE_REPEATED + LIST_RESTARTED, practice again,
  list 1, list 2 cut off after probe slot 2 (no TRIALEND),
  the task relaunched (new CONNECTED, task clock back at 0),
  LIST_RESTARTED, list 2 again, list 3, EXIT.
"""
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

from ..submission.exc import AlignmentError, NoAnnotationError
from ..submission.parsers.elemem_pal_parser import ElememPALLogParser
from ..submission.parsers.math_parser import MathElememLogParser


SUBJECT = 'R1999J'
DRIFT = 35e-6
EEG_DIR = SUBJECT + '_2026-10-06_10-00-00'

PRACTICE_WORDS = ['APPLE', 'RIVER', 'CHAIR', 'MOUSE', 'CLOUD', 'TRAIN',
                  'STONE', 'GRASS', 'BREAD', 'HORSE', 'LIGHT', 'PAPER']
WORDS = ['ANCHOR', 'BASKET', 'CANDLE', 'DESERT', 'FOREST', 'GARDEN', 'HAMMER', 'ISLAND',
         'JACKET', 'KETTLE', 'LADDER', 'MARKET', 'NAPKIN', 'OYSTER', 'PENCIL', 'RABBIT',
         'SADDLE', 'TABLET', 'VALLEY', 'WAGON', 'BUTTON', 'CASTLE', 'DOLPHIN', 'FINGER',
         'GOBLET', 'HELMET', 'INSECT', 'JUNGLE', 'KNIGHT', 'LEMON', 'MIRROR', 'NEEDLE',
         'ORANGE', 'PILLOW', 'ROCKET', 'SHOVEL']

# Scoring scenario per probe: the .ann lines and what the pair should be scored as.
# RT is the onset of the last line whose word differs from the line before it.
SCENARIOS = ['correct', 'voc_then_correct', 'wrong_pair', 'pass', 'empty', 'xli',
             'probe_word', 'correct_twice', 'wrong_then_correct']
EXPECTED = {
    'correct': dict(correct=1, intrusion=0, resp_pass=0, RT=1234.5),
    'voc_then_correct': dict(correct=1, intrusion=0, resp_pass=0, RT=1500.25),
    'wrong_pair': dict(correct=0, intrusion=0, resp_pass=0, RT=1100),
    'pass': dict(correct=0, intrusion=0, resp_pass=1, RT=900),
    'empty': dict(correct=0, intrusion=-999, resp_pass=0, RT=-999),
    'xli': dict(correct=0, intrusion=-1, resp_pass=0, RT=700),
    'probe_word': dict(correct=0, intrusion=0, resp_pass=0, RT=650),
    'correct_twice': dict(correct=1, intrusion=0, resp_pass=0, RT=1000),
    'wrong_then_correct': dict(correct=1, intrusion=0, resp_pass=0, RT=2000),
    'pli': dict(correct=0, intrusion=2, resp_pass=0, RT=1200),
}

ANN_HEADER = ('#Begin Header. [Do not edit before this line. Never edit with an instance of '
              'the program open.]\n#Annotator: test\n#Program Version: 2.1.0\n\n')


def make_session(out, experiment, seed=7):
    """Write a synthetic session to out/ and return the true onsets and scenarios."""
    rng = random.Random(seed)
    lines, truth = [], {}
    clock = dict(E0=1759780000000.0, t=0.0, hb=0, next_hb=0.0, last_recv=0.0, last_send=-1e18,
                 mid=0)

    def emap(t):
        return clock['E0'] + (1 + DRIFT) * t

    def recv(send_t):
        # one TCP stream from one writer thread: arrival order = send order
        r = max(emap(send_t) + rng.uniform(0.3, 1.5), clock['last_recv'] + 0.01)
        clock['last_recv'] = r
        return r

    def task(type_, data=None, queue=True, key=None):
        t = clock['t']
        send = max(t + (rng.uniform(0, 3) if queue else 0), clock['last_send'] + 0.01)
        clock['last_send'] = send
        msg = dict(type=type_, data=data or {}, id=clock['mid'], time=recv(send))
        clock['mid'] += 1
        lines.append(msg)
        if key:
            truth[key] = emap(t)

    def elemem(type_, data=None, at=None):
        lines.append(dict(type=type_, data=data or {}, id=0,
                          time=at if at is not None else emap(clock['t']) + 0.05))

    def heartbeat():
        task('HEARTBEAT', {'count': clock['hb'], 'task_time_ms': clock['t']}, queue=False)
        elemem('HEARTBEAT_OK', {'count': clock['hb']}, at=recv(clock['t']) + 0.05)
        clock['hb'] += 1

    def advance(ms):
        end = clock['t'] + ms
        while clock['next_hb'] <= end:
            clock['t'] = clock['next_hb']
            heartbeat()
            clock['next_hb'] += 1000.0
        clock['t'] = end

    def connect():
        clock.update(t=0.0, hb=0, next_hb=1e12, last_send=-1e18)
        task('CONNECTED', queue=False)
        elemem('CONNECTED_OK')
        task('CONFIGURE', {'stim_mode': 'none', 'experiment': experiment, 'subject': SUBJECT,
                           'session': 0}, queue=False)
        elemem('CONFIGURE_OK')
        for _ in range(20):
            clock['t'] += 50
            heartbeat()
        task('READY', queue=False)
        elemem('START')
        task('SESSION', {'session': 0, 'task_epoch_unix_ms': 1759779993000}, queue=False)
        clock['next_hb'] = clock['t'] + 1000.0

    def run_list(L, pairs, order, stop_after_slot=None):
        def d(extra=None):
            return dict(extra or {}, trial=L, task_time_ms=clock['t'])
        task('TRIAL', d({'stim': False, 'phase_type': 'PRACTICE' if L == 0 else 'NON-STIM'}))
        task('COUNTDOWN', d())
        advance(10000)
        task('COUNTDOWN_END', d())
        task('ENCODING', d())
        for sp, (w1, w2) in enumerate(pairs):
            task('ORIENT', d({'scope': 'pair', 'serialpos': sp}))
            advance(250)
            task('ORIENT_OFF', d())
            advance(rng.uniform(500, 750))
            task('STUDY_PAIR', d({'serialpos': sp, 'word1': w1, 'word2': w2, 'stim': False}),
                 key='%d/SP%d' % (L, sp))
            advance(4000)
            task('PAIR_OFF', d({'serialpos': sp}))
            advance(1000)
        task('ENCODING_END', d())
        if experiment == 'IPAL1':
            task('DISTRACT', d({'type': 'fixation', 'duration_ms': 10000}))
            advance(10000)
            task('DISTRACT_END', d({'type': 'fixation'}))
        else:
            task('DISTRACT', d({'min_duration_ms': 20000, 'practice': L == 0}))
            for _ in range(4):
                a, b, c = [rng.randint(1, 9) for _ in range(3)]
                rt = rng.randint(1500, 5000)
                advance(rt)
                task('MATH', d({'problem': '%d + %d + %d = ' % (a, b, c), 'response': str(a + b + c),
                                'response_time_ms': rt, 'correct': 'True'}))
            task('DISTRACT_END', d({'problems': 4, 'correct': 4}))
        task('RECALL', d({'duration': 0}))
        task('ORIENT', d({'scope': 'recall', 'text': '*******'}))
        advance(500)
        task('ORIENT_OFF', d())
        for slot, (sp, direction) in enumerate(order):
            w1, w2 = pairs[sp]
            probe, expecting = (w2, w1) if direction == 1 else (w1, w2)
            task('ORIENT', d({'scope': 'probe', 'text': '?????', 'slot': slot}))
            advance(250)
            task('ORIENT_OFF', d())
            advance(rng.uniform(500, 750))
            task('TEST_PROBE', d({'probepos': slot, 'serialpos': sp, 'probe': probe,
                                  'expecting': expecting, 'direction': direction, 'stim': False}),
                 key='%d/TP%d' % (L, slot))
            task('REC_START', d({'file': '%d_%d' % (L, slot), 'probepos': slot}),
                 key='%d/RS%d' % (L, slot))
            if slot == stop_after_slot:
                advance(1500)    # link lost mid-probe
                return
            advance(4000)
            task('PROBE_OFF', d({'probepos': slot}))
            advance(1000)
            task('REC_END', d({'file': '%d_%d' % (L, slot), 'probepos': slot}))
        task('TRIALEND', d())
        advance(3000)

    def make_list(words):
        pairs = [(words[2 * i], words[2 * i + 1]) for i in range(6)]
        order = list(range(6))
        rng.shuffle(order)
        return pairs, [(sp, rng.randint(0, 1)) for sp in order]

    lists = {0: make_list(PRACTICE_WORDS)}
    for L in (1, 2, 3):
        lists[L] = make_list(WORDS[(L - 1) * 12: L * 12])

    elemem('ELEMEM', {'version': 'v2.0.0-test'}, at=clock['E0'] - 5000)
    elemem('EEGSTART', {'sub_dir': EEG_DIR}, at=clock['E0'] - 4000)
    connect()
    advance(30000)
    run_list(0, *lists[0])
    task('PRACTICE_REPEATED', {'listno': 0, 'trial': 0, 'task_time_ms': clock['t']})
    task('LIST_RESTARTED', {'listno': 0, 'trial': 0, 'task_time_ms': clock['t']})
    run_list(0, *lists[0])
    run_list(1, *lists[1])
    run_list(2, *lists[2], stop_after_slot=2)
    # relaunch: new connection; the task clock restarts at 0 at a later Elemem time
    clock['E0'] = emap(clock['t']) + 60000
    connect()
    advance(5000)
    task('LIST_RESTARTED', {'listno': 2, 'trial': 2, 'task_time_ms': clock['t']})
    run_list(2, *lists[2])
    run_list(3, *lists[3])
    task('EXIT', queue=False)

    el = os.path.join(out, 'elemem', EEG_DIR)
    os.makedirs(el)
    with open(os.path.join(el, 'event.log'), 'w') as f:
        for m in sorted(lines, key=lambda m: m['time']):
            f.write(json.dumps(m) + '\n')
    with open(os.path.join(el, 'experiment_config.json'), 'w') as f:
        json.dump({'experiment': {'type': experiment}}, f)

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
    return dict(truth=truth, scenarios=scenarios, el=el)


def session_files(out, event_log=None):
    el = glob.glob(os.path.join(out, 'elemem', '*'))[0]
    return {'event_log': [event_log or os.path.join(el, 'event.log')],
            'experiment_config': os.path.join(el, 'experiment_config.json'),
            'annotations': sorted(glob.glob(os.path.join(out, '*.ann')))}


def parse(files, experiment, parser_class=ElememPALLogParser):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        parser = parser_class('r1', SUBJECT, '0.0', experiment, 0, files)
        events = parser.clean_events(parser.parse())
    return parser, events, [str(w.message) for w in caught]


@pytest.fixture(scope='module', params=['PAL1', 'IPAL1'])
def session(request, tmp_path_factory):
    out = str(tmp_path_factory.mktemp(request.param))
    synth = make_session(out, request.param)
    files = session_files(out)
    parser, events, caught = parse(files, request.param)
    return dict(synth, experiment=request.param, out=out, files=files, parser=parser,
                events=events, warnings=caught)


def test_lists_and_positions(session):
    ev = session['events']
    assert ev[ev.type == 'TRIAL'].list.tolist() == [-1, 1, 2, 3]
    pairs = ev[ev.type == 'STUDY_PAIR']
    probes = ev[ev.type == 'TEST_PROBE']
    assert len(pairs) == 24 and len(probes) == 24
    for lst in (-1, 1, 2, 3):
        assert sorted(pairs[pairs.list == lst].serialpos.tolist()) == [1, 2, 3, 4, 5, 6]
        assert sorted(probes[probes.list == lst].probepos.tolist()) == [1, 2, 3, 4, 5, 6]
    assert (ev[ev.list == -1].phase == 'PRACTICE').all()
    assert session['parser'].check_event_quality(ev, session['files']) == []


def test_restarts_keep_last_attempt(session):
    ev, parser, truth = session['events'], session['parser'], session['truth']
    # practice #1 and list 2 #1 are dropped
    assert [trial for trial, _ in parser.dropped_attempts] == [0, 2]
    assert parser.incomplete_lists == []
    assert (ev.type == 'PRACTICE_REPEATED').sum() == 1
    assert (ev.type == 'LIST_RESTARTED').sum() == 2
    # the kept events are those of the last attempt (truth holds the last attempt's onsets)
    for lst, key in ((-1, '0/SP0'), (2, '2/SP0'), (2, '2/TP2')):
        kind = 'STUDY_PAIR' if 'SP' in key else 'TEST_PROBE'
        field = 'serialpos' if kind == 'STUDY_PAIR' else 'probepos'
        e = ev[(ev.type == kind) & (ev.list == lst) & (getattr(ev, field) == int(key[-1]) + 1)]
        assert len(e) == 1
        assert abs(e[0].mstime - truth[key]) < 2
    assert (ev[ev.list == 2].type == 'RECALL_END').sum() == 1


def test_incomplete_list_reported_and_optionally_dropped(session, tmp_path):
    # the first of two Elemem folders: everything before the relaunch
    lines = open(session['files']['event_log'][0]).read().splitlines()
    cut = [i for i, l in enumerate(lines) if json.loads(l)['type'] == 'CONNECTED'][1]
    log = str(tmp_path / 'event.log')
    with open(log, 'w') as f:
        f.write('\n'.join(lines[:cut]) + '\n')
    files = session_files(session['out'], event_log=log)

    parser, ev, caught = parse(files, session['experiment'])
    assert parser.incomplete_lists == [2]
    assert any('list 2 never reached TRIALEND' in w for w in caught)
    assert ev[ev.type == 'TRIAL'].list.tolist() == [-1, 1, 2]

    class DropIncomplete(ElememPALLogParser):
        DROP_INCOMPLETE_LISTS = True
    parser, ev, _ = parse(files, session['experiment'], DropIncomplete)
    assert ev[ev.type == 'TRIAL'].list.tolist() == [-1, 1]
    assert not (ev.list == 2).any()


def test_clock_fit(session):
    fits = session['parser'].clock_fits
    assert len(fits) == 2    # one per task connection
    for fit in fits:
        assert fit['method'] == 'heartbeat_fit'
        assert abs(fit['slope'] - (1 + DRIFT)) < 2e-6
        assert fit['n_lead_violations'] == 0
    ev, truth = session['events'], session['truth']
    errors = []
    for e in ev[(ev.type == 'TEST_PROBE') | (ev.type == 'STUDY_PAIR')]:
        L = 0 if e.list == -1 else e.list
        if e.type == 'TEST_PROBE':
            errors.append(e.mstime - truth['%d/TP%d' % (L, e.probepos - 1)])
        else:
            errors.append(e.mstime - truth['%d/SP%d' % (L, e.serialpos - 1)])
    recs = ev[ev.type == 'REC_EVENT']
    for (L, slot), scenario in session['scenarios'].items():
        lst = -1 if L == 0 else L
        mstimes = np.sort(recs[(recs.list == lst) & (recs.probepos == slot + 1)].mstime)
        onsets = [t for t, _ in scenario['entries']]
        assert len(mstimes) == len(onsets)
        errors.extend(mstimes - truth['%d/RS%d' % (L, slot)] - np.array(onsets))
    assert len(errors) > 48
    assert np.abs(errors).max() <= 2.0


def test_clock_fit_falls_back_to_offset_only(session, tmp_path):
    # keep only 10 heartbeats after the relaunch
    lines = open(session['files']['event_log'][0]).read().splitlines()
    kept, connections, n = [], 0, 0
    for l in lines:
        msg_type = json.loads(l)['type']
        connections += msg_type == 'CONNECTED'
        if msg_type == 'HEARTBEAT' and connections == 2:
            n += 1
            if n > 10:
                continue
        kept.append(l)
    log = str(tmp_path / 'event.log')
    with open(log, 'w') as f:
        f.write('\n'.join(kept) + '\n')
    parser, ev, _ = parse(session_files(session['out'], event_log=log), session['experiment'])
    assert [fit['method'] for fit in parser.clock_fits] == ['heartbeat_fit', 'offset_only']
    assert len(ev[ev.type == 'TEST_PROBE']) == 24


def test_scoring(session):
    ev = session['events']
    seen = set()
    for probe in ev[ev.type == 'TEST_PROBE']:
        L = 0 if probe.list == -1 else probe.list
        scenario = session['scenarios'][(L, probe.probepos - 1)]
        want = EXPECTED[scenario['scen']]
        seen.add(scenario['scen'])
        assert probe.correct == want['correct'], scenario['scen']
        assert probe.resp_pass == want['resp_pass'], scenario['scen']
        assert probe.intrusion == want['intrusion'], scenario['scen']
        assert abs(probe.RT - want['RT']) <= 1, scenario['scen']
        recs = ev[(ev.type == 'REC_EVENT') & (ev.list == probe.list) & (ev.probepos == probe.probepos)]
        assert len(recs) == len(scenario['entries'])
        if scenario['scen'] == 'voc_then_correct':
            assert recs.vocalization.tolist() == [1, 0]
        if scenario['scen'] == 'empty':
            assert probe.resp_word == ''
    assert seen == set(EXPECTED)


def test_backfill_onto_study_pair(session):
    ev = session['events']
    for probe in ev[ev.type == 'TEST_PROBE']:
        pair = ev[(ev.type == 'STUDY_PAIR') & (ev.list == probe.list) &
                  (ev.serialpos == probe.serialpos)]
        assert len(pair) == 1
        for field in ('probepos', 'correct', 'RT', 'resp_word', 'resp_pass', 'probe_word',
                      'expecting_word', 'cue_direction'):
            assert pair[0][field] == probe[field], field


def test_missing_ann(session):
    files = dict(session['files'])
    # a real list: raises, as the FR parsers do
    files['annotations'] = [f for f in session['files']['annotations']
                            if os.path.basename(f) != '1_0.ann']
    with pytest.raises(NoAnnotationError):
        parse(files, session['experiment'])
    # practice: warns and leaves the probe unscored
    files['annotations'] = [f for f in session['files']['annotations']
                            if os.path.basename(f) != '0_0.ann']
    _, ev, caught = parse(files, session['experiment'])
    assert any('no 0_0.ann' in w for w in caught)
    probe = ev[(ev.type == 'TEST_PROBE') & (ev.list == -1) & (ev.probepos == 1)]
    assert probe[0].correct == -999


def test_retention_interval_events(session):
    ev = session['events']
    starts, ends = ('FIXATION_START', 'FIXATION_END') if session['experiment'] == 'IPAL1' \
        else ('MATH_START', 'MATH_END')
    others = {'FIXATION_START', 'FIXATION_END', 'MATH_START', 'MATH_END'} - {starts, ends}
    assert ev[ev.type == starts].list.tolist() == [-1, 1, 2, 3]
    assert ev[ev.type == ends].list.tolist() == [-1, 1, 2, 3]
    assert not np.isin(ev.type, list(others)).any()


def test_math_lists_numbered_from_trial(tmp_path):
    make_session(str(tmp_path), 'PAL1')
    math = MathElememLogParser('r1', SUBJECT, '0.0', 'PAL1', 0, session_files(str(tmp_path))).parse()
    distract = math[math['type'] == 'DISTRACT_START']
    # practice twice, list 2 twice (the cut-off attempt and its rerun)
    assert distract['list'].tolist() == [-1, -1, 1, 2, 2, 3]
    probs = math[math['type'] == 'PROB']
    assert sorted(set(probs['list'].tolist())) == [-1, 1, 2, 3]
    assert (probs['list'] == -1).sum() == 8 and (probs['list'] == 2).sum() == 8


def test_math_lists_counted_without_trial(tmp_path):
    # FR's messages carry no trial: lists are still counted from DISTRACT, as before
    t = 1.76e12
    lines = []
    for L in range(3):
        lines.append({'type': 'TRIAL', 'data': {'trial': L, 'stim': False}, 'time': t})
        lines.append({'type': 'DISTRACT', 'data': {'phase_type': 'NON-STIM'}, 'time': t + 1000})
        for k in range(2):
            lines.append({'type': 'MATH', 'data': {'problem': '1 + 2 + %d = ' % k, 'response': str(3 + k),
                                                   'response_time_ms': '1500', 'correct': 'True'},
                          'time': t + 3000 + 2000 * k})
        t += 60000
    with open(str(tmp_path / 'event.log'), 'w') as f:
        f.write('\n'.join(json.dumps(l) for l in lines) + '\n')
    with open(str(tmp_path / 'experiment_config.json'), 'w') as f:
        json.dump({}, f)
    files = {'event_log': [str(tmp_path / 'event.log')],
             'experiment_config': str(tmp_path / 'experiment_config.json')}
    math = MathElememLogParser('r1', SUBJECT, '0.0', 'FR1', 0, files).parse()
    assert math['list'].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_registry():
    from ..submission.events_tasks import EventCreationTask
    from ..submission.pipelines import determine_groups
    from ..submission.transfer_inputs import TRANSFER_INPUTS
    assert EventCreationTask.R1_PARSERS(4.0)['PAL'] is ElememPALLogParser
    assert EventCreationTask.R1_PARSERS(4.0)['IPAL'] is ElememPALLogParser
    groups = determine_groups('r1', SUBJECT, 'IPAL1', 0, TRANSFER_INPUTS['behavioral'], 'transfer')
    assert 'verbal' in groups and 'system_4' in groups and 'math' not in groups
    groups = determine_groups('r1', SUBJECT, 'PAL1', 0, TRANSFER_INPUTS['behavioral'], 'transfer')
    assert 'verbal' in groups and 'math' in groups


def test_system4_alignment(session, tmp_path):
    pyedflib = pytest.importorskip('pyedflib')
    from ..submission.alignment.system4 import System4Offset
    eeg_dir = str(tmp_path / EEG_DIR)
    os.makedirs(eeg_dir)
    log = [json.loads(l) for l in open(session['files']['event_log'][0])]
    eeg_start = [m['time'] for m in log if m['type'] == 'EEGSTART'][0]
    sample_rate = 250
    n_samples = int((log[-1]['time'] - eeg_start) / 1000. * sample_rate) + 5 * sample_rate
    writer = pyedflib.EdfWriter(os.path.join(eeg_dir, 'eeg_data.edf'), 1,
                                file_type=pyedflib.FILETYPE_EDFPLUS)
    writer.setSignalHeaders([dict(label='LA1', dimension='uV', sample_frequency=sample_rate,
                                  physical_max=3000, physical_min=-3000, digital_max=32767,
                                  digital_min=-32768)])
    writer.writeSamples([np.zeros(n_samples)])
    writer.close()
    sources = str(tmp_path / 'sources.json')
    with open(sources, 'w') as f:
        json.dump({'R1999J_PAL_0': {'sample_rate': sample_rate, 'n_samples': n_samples}}, f)
    files = dict(session['files'], eeg_sources=sources)

    parser = ElememPALLogParser('r1', SUBJECT, '0.0', session['experiment'], 0, files)
    events = parser.clean_events(System4Offset(parser.parse(), files, eeg_dir).align())
    assert (events.eegoffset >= 0).all()
    probes = events[events.type == 'TEST_PROBE']
    assert (probes.eegoffset == np.round((probes.mstime - eeg_start) * sample_rate / 1000.)).all()

    with open(sources, 'w') as f:
        json.dump({'R1999J_PAL_0': {}, 'R1999J_PAL_0b': {}}, f)
    with pytest.raises(AlignmentError):
        System4Offset(events, files, eeg_dir)
