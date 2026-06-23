"""Unit tests for ``System4AlignerCorrection._correct_events``.

The System-4 clock correction must touch ONLY the four time fields
(``mstime``, ``eegoffset``, ``mstime_uncorrected``, ``eegoffset_uncorrected``)
and leave every other behavioral column byte-identical. Task events get
corrected; STIM and Elemem-originated (host-clock) events are restored to their
uncorrected values.

This is the authoritative "correction changes nothing but the time fields"
invariant -- it is the only place both the pre- and post-correction events are
in hand (the persisted task_events.json saves no baseline for the other
columns). Pure unit test: no rhino data, no EEG, no pipeline.
"""
import numpy as np

from event_creation.submission.alignment.system4 import System4AlignerCorrection

# The only fields the correction is permitted to modify.
TIME_FIELDS = {'mstime', 'eegoffset', 'mstime_uncorrected', 'eegoffset_uncorrected'}


def _make_events():
    """A small synthetic events recarray: three task rows + one locked STIM row.

    ``mstime_uncorrected`` / ``eegoffset_uncorrected`` start at the dtype default
    (-1), exactly as they arrive from the parser before alignment runs.
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
        # type,      item,  spos,  list, rec, eegfile,  mstime,  eegoffset, mu, eu
        ('WORD',     'CAT',  0,     1,   1,   'r1.edf', 1000500, -1, -1, -1),
        ('WORD',     'DOG',  1,     1,   0,   'r1.edf', 1001000, -1, -1, -1),
        ('REC_WORD', 'CAT', -999,   1,   1,   'r1.edf', 1002000, -1, -1, -1),
        ('STIM',     '',    -999,   1,   0,   'r1.edf', 1003000, -1, -1, -1),  # locked
    ]
    return np.array(rows, dtype=dtype).view(np.recarray)


def _make_aligner(experiment='catFR1', eeg_start_ms=1000000, sample_rate=1000.0):
    """A bare aligner with only the attributes ``_correct_events`` reads."""
    aligner = System4AlignerCorrection.__new__(System4AlignerCorrection)
    aligner.experiment = experiment
    aligner.eeg_start_ms = eeg_start_ms
    aligner.sample_rate = sample_rate
    return aligner


def test_correct_events_changes_only_time_fields():
    events = _make_events()
    # Snapshot BEFORE: _correct_events fills mstime_uncorrected on the input in place.
    original = events.copy()

    out = _make_aligner()._correct_events(events, slope=1.0, offset=200.0)

    # No column added, dropped, or renamed.
    assert out.dtype.names == original.dtype.names

    # Every non-time column is byte-identical to the input.
    for name in original.dtype.names:
        if name in TIME_FIELDS:
            continue
        assert np.array_equal(out[name], original[name]), \
            'correction modified non-time column %r' % name


def test_correct_events_task_corrected_locked_restored():
    events = _make_events()
    original = events.copy()

    out = _make_aligner()._correct_events(events, slope=1.0, offset=200.0)

    types = np.array([str(t).upper() for t in out['type']])
    is_locked = types == 'STIM'
    is_task = ~is_locked

    # The uncorrected snapshot equals the original mstime for every row.
    assert np.array_equal(out['mstime_uncorrected'], original['mstime'])

    # Task rows: BOTH time fields actually moved (this is the eegoffset bug guard).
    assert np.all(out['mstime'][is_task] != out['mstime_uncorrected'][is_task])
    assert np.all(out['eegoffset'][is_task] != out['eegoffset_uncorrected'][is_task])

    # Locked (host-clock) rows: restored to uncorrected.
    assert np.all(out['mstime'][is_locked] == out['mstime_uncorrected'][is_locked])
    assert np.all(out['eegoffset'][is_locked] == out['eegoffset_uncorrected'][is_locked])


def test_eegoffset_is_exactly_mstime_in_samples():
    """eegoffset must be EXACTLY its own mstime converted to EEG samples, and must
    track mstime row-for-row -- the two invariants that the wrong-ms-channel bug broke."""
    events = _make_events()
    aligner = _make_aligner()
    out = aligner._correct_events(events, slope=1.0, offset=200.0)

    # (2) eegoffset == samples(mstime), for both the corrected and uncorrected pair.
    assert np.array_equal(out['eegoffset'], aligner._calc_eegoffset(out['mstime']))
    assert np.array_equal(out['eegoffset_uncorrected'],
                          aligner._calc_eegoffset(out['mstime_uncorrected']))

    # (1) per-row coupling: wherever mstime moved, eegoffset moved too.
    ms_moved = out['mstime'] != out['mstime_uncorrected']
    off_moved = out['eegoffset'] != out['eegoffset_uncorrected']
    assert np.all(off_moved[ms_moved]), 'a row whose mstime moved kept its eegoffset'
