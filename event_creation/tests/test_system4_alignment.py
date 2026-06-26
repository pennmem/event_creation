"""Unit tests for ``System4AlignerCorrection._correct_events``.

The System-4 clock correction must touch ONLY the four time fields
(``mstime``, ``eegoffset``, ``mstime_uncorrected``, ``eegoffset_uncorrected``)
and leave every other behavioral column byte-identical.

Semantics (see ``_correct_events``): the relative-time heartbeat fit returns
``(slope, b, Tt0)`` and ``mstime``/``eegoffset`` are computed INDEPENDENTLY:

  * ``mstime``    = ``Tt0 + (M - eeg_start_ms - b)/slope`` for every event
                    (continuous inverse fit onto the task clock).
  * ``eegoffset`` = ``round(slope*(M - eeg_start_ms)*sr/1000)`` for behavioral
                    (task) events; STIM / Elemem-originated (host-clock) events
                    keep the plain host sample (``eegoffset`` restored, no slope).

Because ``mstime`` lives on the task clock and ``eegoffset`` on host samples, the
two are no longer EXACTLY ``eegoffset == samples(mstime)`` -- they are only CLOSE
(they differ by the slope drift, ~rel*2*(slope-1)). The *uncorrected* pair, both
on the host clock, stays exact. These two facts are the guard against the
original "eegoffset computed from the wrong ms channel" bug.

Pure unit test: no rhino data, no EEG, no pipeline.
"""
import numpy as np

from event_creation.submission.alignment.system4 import System4AlignerCorrection

# The only fields the correction is permitted to modify.
TIME_FIELDS = {'mstime', 'eegoffset', 'mstime_uncorrected', 'eegoffset_uncorrected'}

# Fit parameters used across the tests. ``Tt0 == EEG_START_MS`` makes the task and
# host clocks coincide, so the corrected (task-clock) mstime stays on the host
# frame and the eegoffset<->mstime closeness check below is meaningful. ``slope``
# is a realistic ~200 ppm sample-rate drift; ``b`` is the relative-time intercept
# (~0 ms in practice).
EEG_START_MS = 1_000_000
SAMPLE_RATE = 1000.0
SLOPE = 1.0002
B = 0.0
TT0 = EEG_START_MS
# EEG source stem stamped onto in-bounds events, and a recording length long
# enough that every synthetic event falls inside it (so eegfile is stamped).
EEG_FILE_STEM = 'R1999X_catFR1_0_01Jan20_0000'
NUM_SAMPLES = 10_000_000


def _make_events():
    """A small synthetic events recarray: three task rows + one locked STIM row.

    The session is long enough (events hundreds of seconds apart) that the ~200 ppm
    slope drift actually moves the integer eegoffset. ``mstime_uncorrected`` /
    ``eegoffset_uncorrected`` start at the dtype default (-1), exactly as they
    arrive from the parser before alignment runs.
    """
    dtype = np.dtype([
        ('type', 'U20'),
        ('item_name', 'U20'),
        ('serialpos', '<i8'),
        ('list', '<i8'),
        ('recalled', '<i8'),
        ('eegfile', 'U32'),
        ('mstime', '<i8'),
        ('eegoffset', '<i8'),
        ('mstime_uncorrected', '<i8'),
        ('eegoffset_uncorrected', '<i8'),
    ])
    rows = [
        # type,      item,  spos,  list, rec, eegfile,  mstime,    eegoffset, mu, eu
        ('WORD',     'CAT',  0,     1,   1,   'r1.edf', 1_100_000, -1, -1, -1),
        ('WORD',     'DOG',  1,     1,   0,   'r1.edf', 1_500_000, -1, -1, -1),
        ('REC_WORD', 'CAT', -999,   1,   1,   'r1.edf', 1_900_000, -1, -1, -1),
        ('STIM',     '',    -999,   1,   0,   'r1.edf', 1_950_000, -1, -1, -1),  # locked
    ]
    return np.array(rows, dtype=dtype).view(np.recarray)


def _make_aligner(experiment='catFR1', eeg_start_ms=EEG_START_MS, sample_rate=SAMPLE_RATE,
                  eeg_file_stem=EEG_FILE_STEM, num_samples=NUM_SAMPLES):
    """A bare aligner with only the attributes ``_correct_events`` reads."""
    aligner = System4AlignerCorrection.__new__(System4AlignerCorrection)
    aligner.experiment = experiment
    aligner.eeg_start_ms = eeg_start_ms
    aligner.sample_rate = sample_rate
    aligner.eeg_file_stem = eeg_file_stem
    aligner.num_samples = num_samples
    return aligner


def _correct(events):
    return _make_aligner()._correct_events(events, slope=SLOPE, b=B, Tt0=TT0)


# The correction is also permitted to (re)write eegfile -- it is the field the
# aligner stamps now that it is the complete standalone System-4 aligner.
MUTABLE_FIELDS = TIME_FIELDS | {'eegfile'}


def test_correct_events_changes_only_time_and_eegfile_fields():
    events = _make_events()
    # Snapshot BEFORE: _correct_events fills mstime_uncorrected on the input in place.
    original = events.copy()

    out = _correct(events)

    # No column added, dropped, or renamed.
    assert out.dtype.names == original.dtype.names

    # Every column except the time fields and eegfile is byte-identical to the input.
    for name in original.dtype.names:
        if name in MUTABLE_FIELDS:
            continue
        assert np.array_equal(out[name], original[name]), \
            'correction modified column %r it should not touch' % name


def test_correct_events_stamps_eegfile_on_in_bounds_events():
    """Regression guard for the 2026-06-12 bug where System4Offset was commented
    out and the corrector ran standalone, leaving eegfile BLANK on every System-4
    session. The corrector must now stamp the EEG source stem on in-bounds events."""
    events = _make_events()
    out = _correct(events)  # default aligner: NUM_SAMPLES covers all events

    # Every event is in-bounds here, so all must carry the EEG file stem...
    assert np.all(out['eegfile'] == EEG_FILE_STEM)
    # ...and none may be left blank (the exact failure the bug produced).
    assert not np.any(out['eegfile'] == '')


def test_correct_events_blanks_eegfile_when_out_of_bounds():
    """Events whose corrected sample falls outside the recording keep an empty
    eegfile (mirroring System4Offset), so downstream readers skip them."""
    events = _make_events()
    # Recording shorter than the later events: only the first event stays in-bounds.
    aligner = _make_aligner(num_samples=200_000)
    out = aligner._correct_events(events, slope=SLOPE, b=B, Tt0=TT0)

    in_bounds = (out['eegoffset'] >= 0) & (out['eegoffset'] <= aligner.num_samples)
    assert in_bounds.any() and not in_bounds.all(), 'test needs a mix of in/out-of-bounds'
    assert np.all(out['eegfile'][in_bounds] == EEG_FILE_STEM)
    assert np.all(out['eegfile'][~in_bounds] == '')


def test_task_corrected_locked_eegoffset_restored():
    events = _make_events()
    original = events.copy()

    out = _correct(events)

    types = np.array([str(t).upper() for t in out['type']])
    is_locked = types == 'STIM'
    is_task = ~is_locked

    # The uncorrected snapshot equals the original mstime for every row.
    assert np.array_equal(out['mstime_uncorrected'], original['mstime'])

    # Task rows: BOTH time fields actually moved (this is the eegoffset bug guard --
    # with the realistic slope the slope-fit must shift the integer eegoffset).
    assert np.all(out['mstime'][is_task] != out['mstime_uncorrected'][is_task])
    assert np.all(out['eegoffset'][is_task] != out['eegoffset_uncorrected'][is_task])

    # Locked (host-clock) rows: eegoffset is restored to the plain host sample...
    assert np.all(out['eegoffset'][is_locked] == out['eegoffset_uncorrected'][is_locked])
    # ...but mstime is still mapped onto the (task) clock like every other event,
    # so all events share one mstime clock.
    assert np.all(out['mstime'][is_locked] != out['mstime_uncorrected'][is_locked])


def test_eegoffset_tracks_mstime_in_samples():
    """eegoffset must TRACK its own mstime in EEG samples -- exactly for the
    uncorrected (host-clock) pair, and closely (not exactly) for the corrected
    pair, since corrected mstime is on the task clock and eegoffset on host
    samples. These are the invariants the wrong-ms-channel bug broke."""
    events = _make_events()
    aligner = _make_aligner()
    out = aligner._correct_events(events, slope=SLOPE, b=B, Tt0=TT0)

    # Uncorrected pair: both on the host clock, so EXACT.
    assert np.array_equal(out['eegoffset_uncorrected'],
                          aligner._calc_eegoffset(out['mstime_uncorrected']))

    # Corrected pair: CLOSE but not exact. The gap is the slope drift,
    # ~rel*2*(slope-1) (~0.04% here); a wrong-ms-channel would be wildly off.
    assert np.allclose(out['eegoffset'],
                       aligner._calc_eegoffset(out['mstime']),
                       rtol=1e-3, atol=1)

    # Task eegoffset is exactly the slope-fit definition.
    types = np.array([str(t).upper() for t in out['type']])
    is_task = types != 'STIM'
    M = np.asarray(out['mstime_uncorrected'], dtype=float)
    expected = np.round(SLOPE * (M - EEG_START_MS) * SAMPLE_RATE / 1000.).astype(int)
    assert np.array_equal(out['eegoffset'][is_task], expected[is_task])

    # Per-row coupling for task rows: wherever a task mstime moved, its eegoffset
    # moved too. (Locked rows are excluded: their mstime is corrected onto the task
    # clock but their eegoffset is intentionally restored to the host sample.)
    ms_moved = out['mstime'] != out['mstime_uncorrected']
    off_moved = out['eegoffset'] != out['eegoffset_uncorrected']
    task_moved = ms_moved & is_task
    assert np.all(off_moved[task_moved]), 'a task row whose mstime moved kept its eegoffset'
