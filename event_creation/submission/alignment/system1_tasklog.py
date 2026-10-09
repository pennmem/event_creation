"""
System 1 alignment for tasks that log their sync pulses in their own JSON-lines log
(PAL1/IPAL1's pal_events.jsonl), with launches, several EEG files and fit-quality gates.

Inputs (the ``files`` of the events pipeline)
--------------------------------------------
* ``session_log``: the task log. Every pulse is a line
  ``{"type": "syncPulse", "time": <stopwatch ms just before the send>, "data": {"launch": k, ...}}``
  and every launch starts with ``LAUNCH {"epochUnixMs"}``; the stopwatch restarts at each
  launch, so task pulse times are taken as ``epochUnixMs + time`` (pal_task_log.py), the
  same absolute time the parser puts in ``mstime``.
* ``sync_pulses``: one ``.sync.txt`` per EEG file, one sample index per line, as written by
  readers/sync_pulse_extractor.py (rising edges, at the split rate). With one EEG file the
  name does not matter; with several, ``<recording stem>.sync.txt`` (or
  ``<recording stem>.<anything>.sync.txt``) is paired with the sources.json entry whose
  ``source_file`` is ``<recording stem>.<ext>``.
* ``eeg_sources``: sources.json from the split (sample_rate, n_samples, source_file).

Method
------
For every launch and every EEG file:

1. Anchor: find a window of ``ANCHOR_INTERVALS`` consecutive task inter-pulse intervals
   that matches exactly one window of EEG intervals: after the common scale (the median
   ratio of the intervals), the 75th percentile of the absolute interval differences must
   be under ``ANCHOR_TOLERANCE_MS``. This is LTPAligner.times_to_offsets' median match
   made stricter, so that a launch and a file that do not overlap practically never
   anchor by chance. The pulses are 800-1200 ms apart at random, so ~20 intervals
   identify a stretch uniquely; a missing or extra pulse only costs the windows that
   contain it, and no common start is assumed (a launch need not be inside the file, and
   a file need not start before the launch). The line is seeded from the intervals that
   match one by one (median offset), not from the window paired by index.
2. Grow: fit a line on the anchor, match every task pulse within ``MATCH_TOLERANCE_MS`` of
   its predicted EEG time to the nearest EEG pulse, refit, doubling the stretch of the
   launch used each time, so drift is followed outwards.
3. Gates, each an AlignmentError naming the launch and file:
   * slope more than ``MAX_SLOPE_ERROR`` (1 %) from 1: a sample-rate mismatch between the
     .sync.txt and sources.json, not clock drift;
   * more matched pulses off the line by more than ``MAX_RESIDUAL_MS`` (5 ms) than
     ``MAX_OUTLIER_FRACTION`` (5 %) of them, or ``MIN_OUTLIERS_ALLOWED`` (2) in a short
     train; fewer are dropped and the line refitted, so every pulse kept is within 5 ms.
     A LabJack pulse is sometimes tens of ms late (USB scheduling), so a few late pulses
     are expected; a wrong match puts many off the line, which this and the matched
     fraction below catch;
   * fewer than ``MIN_MATCHED_FRACTION`` of the task pulses that fall inside the file's
     pulse span matched, or fewer than ``MIN_MATCHED_PULSES`` in all.

Each event is then put in its launch by mstime (launch k covers mstime from its epoch to
the next launch's), mapped through that launch's fit for each file that has one, and
given the sample in the file that contains it (``eegoffset`` rounded, ``eegfile`` = the
sources.json key). Events outside every file, or in a launch with no fit, keep
``eegoffset`` -1 and ``eegfile`` ''. ``self.fits`` holds each fit's statistics.

Not handled
-----------
* Pulse-send latency: task pulse times are when the task asked for the pulse; the delay
  to the edge on the EEG (helper + LabJack USB, a few ms) is a constant bias that the
  fit cannot see. Measure it on the bench (photodiode) if it matters.
* A task-computer clock set back between launches (launch k+1 starting before launch k
  ended on the absolute clock) makes the launches ambiguous: AlignmentError.
"""

import json
import os
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..exc import AlignmentError
from ..log import logger
from ..parsers.pal_task_log import read_task_log, SYNC_PULSE


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _sync_stem(path):
    name = os.path.basename(path)
    return name[:-len('.sync.txt')] if name.endswith('.sync.txt') else os.path.splitext(name)[0]


class TaskLogSystem1Aligner(object):
    """Per-launch, per-file sync-pulse alignment. See the module docstring."""

    TASK_TIME_FIELD = 'mstime'
    EEG_TIME_FIELD = 'eegoffset'
    EEG_FILE_FIELD = 'eegfile'

    ANCHOR_INTERVALS = 20        # intervals in an anchor window
    MIN_ANCHOR_INTERVALS = 8     # shortest usable window (a launch with ~10 pulses)
    ANCHOR_TOLERANCE_MS = 10.    # |EEG interval - scale * task interval| of a matching interval
    ANCHOR_QUANTILE = 75         # this percentile of a window's differences must be within it
    ANCHOR_SCALE = (0.9, 1.1)    # scales considered when anchoring (the slope gate is 1 %)
    MATCH_TOLERANCE_MS = 10.     # a task pulse matches the nearest EEG pulse within this
    MAX_RESIDUAL_MS = 5.         # max |residual| of a pulse kept in the fit
    MAX_OUTLIER_FRACTION = 0.05  # more matched pulses than this beyond MAX_RESIDUAL_MS: error
    MIN_OUTLIERS_ALLOWED = 2     # ... but a short train may always drop this many
    MAX_SLOPE_ERROR = 0.01       # |slope - 1|
    MIN_MATCHED_PULSES = 10
    MIN_MATCHED_FRACTION = 0.9   # of the task pulses inside the file's pulse span

    def __init__(self, events, files):
        self.events = events
        self.task_log = files['session_log']
        if isinstance(self.task_log, (list, tuple)):
            self.task_log = self.task_log[0]
        self.sync_files = _as_list(files.get('sync_pulses'))
        if not self.sync_files:
            raise AlignmentError('No sync pulse file (.sync.txt) for this session')
        with open(files['eeg_sources']) as f:
            self.sources = json.load(f)
        if not self.sources:
            raise AlignmentError('sources.json lists no EEG files')
        self.fits = []

    # ---- inputs ---------------------------------------------------------------------

    def read_launches(self):
        """[(launch dict, absolute task pulse times)] in file order."""
        launches = read_task_log(self.task_log)
        out = []
        for launch in launches:
            pulses = [m for m in launch['messages'] if m['type'] == SYNC_PULSE]
            for m in pulses:
                stated = m['data'].get('launch')
                if stated is not None and int(stated) != launch['launch']:
                    warnings.warn('syncPulse on line %d says launch %s but follows LAUNCH #%d'
                                  % (m['_line'], stated, launch['launch']))
            out.append((launch, np.sort(np.array([m['_abs'] for m in pulses], dtype=float))))
        for (a, _), (b, _) in zip(out[:-1], out[1:]):
            if b['epoch_unix_ms'] < a['last_abs']:
                raise AlignmentError(
                    'Launch %d starts (%.0f) before launch %d ends (%.0f) on the task clock '
                    '(was the clock set back?); events cannot be assigned to launches'
                    % (b['launch'], b['epoch_unix_ms'], a['launch'], a['last_abs']))
        return out

    def pair_sources(self):
        """{source name: sync file}; sources without a sync file are left out."""
        names = list(self.sources)
        if len(names) == 1 and len(self.sync_files) == 1:
            return {names[0]: self.sync_files[0]}
        pairs, used = {}, set()
        for name in names:
            src = self.sources[name].get('source_file') or name
            stem = os.path.splitext(os.path.basename(src))[0]
            hits = [s for s in self.sync_files
                    if _sync_stem(s) == stem or _sync_stem(s).startswith(stem + '.')]
            if len(hits) > 1:
                raise AlignmentError('Several sync files for %s: %s' % (src, hits))
            if hits:
                pairs[name] = hits[0]
                used.add(hits[0])
            else:
                logger.warn('No .sync.txt for EEG file %s; its events cannot be aligned' % src)
        unused = [s for s in self.sync_files if s not in used]
        if unused:
            raise AlignmentError('Sync files %s match no EEG file in sources.json (%s); name '
                                 'them <recording stem>.sync.txt'
                                 % ([os.path.basename(s) for s in unused],
                                    [self.sources[n].get('source_file') for n in names]))
        if not pairs:
            raise AlignmentError('No EEG file has a sync file')
        return pairs

    @staticmethod
    def read_eeg_pulses(sync_file, sample_rate):
        idx = np.loadtxt(sync_file, ndmin=1, dtype=float)
        return np.sort(idx) * 1000. / float(sample_rate)

    # ---- matching -------------------------------------------------------------------

    def find_anchor(self, task_ms, eeg_ms):
        """(task index, eeg index, n intervals, scale) of a uniquely matching window, or None."""
        d_task, d_eeg = np.diff(task_ms), np.diff(eeg_ms)
        w = min(self.ANCHOR_INTERVALS, len(d_task))
        if w < self.MIN_ANCHOR_INTERVALS or len(d_eeg) < w:
            return None
        eeg_win = sliding_window_view(d_eeg, w)
        for i in range(0, len(d_task) - w + 1, max(1, w // 2)):
            seg = d_task[i:i + w]
            scale = np.median(eeg_win / seg, axis=1)
            cost = np.percentile(np.abs(eeg_win - scale[:, None] * seg), self.ANCHOR_QUANTILE,
                                 axis=1)
            cost[(scale < self.ANCHOR_SCALE[0]) | (scale > self.ANCHOR_SCALE[1])] = np.inf
            good = np.flatnonzero(cost < self.ANCHOR_TOLERANCE_MS)
            if len(good) == 1:
                return i, int(good[0]), w, float(scale[good[0]])
        return None

    def _nearest(self, eeg_ms, pred):
        k = np.clip(np.searchsorted(eeg_ms, pred), 1, len(eeg_ms) - 1)
        left, right = eeg_ms[k - 1], eeg_ms[k]
        nn = np.where(np.abs(pred - left) <= np.abs(right - pred), k - 1, k)
        if len(eeg_ms) == 1:
            nn = np.zeros(len(pred), dtype=int)
        return nn, eeg_ms[nn] - pred

    def _pairs(self, task_ms, eeg_ms, slope, offset, keep):
        nn, d = self._nearest(eeg_ms, slope * task_ms + offset)
        sel = keep & (np.abs(d) < self.MATCH_TOLERANCE_MS)
        # one EEG pulse per task pulse: keep the closest if two claim the same one
        ti = np.flatnonzero(sel)
        order = ti[np.argsort(np.abs(d[ti]))]
        _, first = np.unique(nn[order], return_index=True)
        ti = np.sort(order[first])
        return ti, nn[ti]

    def match(self, task_ms, eeg_ms, where=''):
        """Fit task -> EEG ms for one launch and one file; None if they do not overlap."""
        anchor = self.find_anchor(task_ms, eeg_ms)
        if anchor is None:
            return None
        i, j, w, slope = anchor
        # Seed the line from the intervals of the window that match one by one: a pulse
        # missing on one side shifts the pairing after it, so pairing the whole window by
        # index would be wrong; the median offset of the matching pairs is not.
        d = np.abs(np.diff(eeg_ms[j:j + w + 1]) - slope * np.diff(task_ms[i:i + w + 1]))
        m = np.flatnonzero(d < self.ANCHOR_TOLERANCE_MS)
        t_seed = np.r_[task_ms[i + m], task_ms[i + m + 1]]
        e_seed = np.r_[eeg_ms[j + m], eeg_ms[j + m + 1]]
        offset = np.median(e_seed - slope * t_seed)
        center = np.mean(task_ms[i:i + w + 1])
        radius = max(task_ms[i + w] - task_ms[i], 1.)
        span = max(task_ms[-1] - center, center - task_ms[0])
        while True:
            ti, ei = self._pairs(task_ms, eeg_ms, slope, offset,
                                 np.abs(task_ms - center) <= radius)
            if len(ti) >= 2:
                slope, offset = np.polyfit(task_ms[ti], eeg_ms[ei], 1)
            if radius >= span:
                break
            radius *= 2
        ti, ei = self._pairs(task_ms, eeg_ms, slope, offset, np.ones(len(task_ms), bool))
        if len(ti) < self.MIN_MATCHED_PULSES:
            raise AlignmentError('%s: only %d pulses matched (need %d)'
                                 % (where, len(ti), self.MIN_MATCHED_PULSES))
        slope, offset = np.polyfit(task_ms[ti], eeg_ms[ei], 1)

        # Gates
        if abs(slope - 1) > self.MAX_SLOPE_ERROR:
            raise AlignmentError(
                '%s: slope %.5f is more than %g%% from 1. That is a sample-rate mismatch '
                '(.sync.txt not at the sources.json rate?), not clock drift'
                % (where, slope, 100 * self.MAX_SLOPE_ERROR))
        resid = eeg_ms[ei] - (slope * task_ms[ti] + offset)
        out = np.abs(resid) > self.MAX_RESIDUAL_MS
        if out.sum() > max(self.MIN_OUTLIERS_ALLOWED, self.MAX_OUTLIER_FRACTION * len(ti)):
            raise AlignmentError(
                '%s: %d of %d matched pulses are more than %g ms off the fit (max %.1f ms, '
                'RMS %.2f ms); the pulse train is unreliable'
                % (where, out.sum(), len(ti), self.MAX_RESIDUAL_MS, np.abs(resid).max(),
                   np.sqrt(np.mean(resid ** 2))))
        n_outliers = int(out.sum())
        if n_outliers:
            ti, ei = ti[~out], ei[~out]
            slope, offset = np.polyfit(task_ms[ti], eeg_ms[ei], 1)
            resid = eeg_ms[ei] - (slope * task_ms[ti] + offset)
        pred = slope * task_ms + offset
        inside = (pred >= eeg_ms[0] - self.MATCH_TOLERANCE_MS) & \
                 (pred <= eeg_ms[-1] + self.MATCH_TOLERANCE_MS)
        fraction = len(ti) / float(max(inside.sum(), 1))
        if fraction < self.MIN_MATCHED_FRACTION:
            raise AlignmentError(
                '%s: only %d of the %d task pulses inside the EEG pulse span matched (%.0f%%)'
                % (where, len(ti), inside.sum(), 100 * fraction))
        return dict(slope=float(slope), offset=float(offset), n_matched=int(len(ti)),
                    n_outliers=n_outliers, n_task_pulses=int(len(task_ms)),
                    n_inside=int(inside.sum()), matched_fraction=float(fraction),
                    rms_ms=float(np.sqrt(np.mean(resid ** 2))),
                    max_residual_ms=float(np.abs(resid).max()),
                    task_first=float(task_ms[ti[0]]), task_last=float(task_ms[ti[-1]]))

    # ---- alignment ------------------------------------------------------------------

    def align(self):
        launches = self.read_launches()
        pairs = self.pair_sources()
        eeg = {name: self.read_eeg_pulses(sync, self.sources[name]['sample_rate'])
               for name, sync in pairs.items()}

        fits = {}
        for launch, task_ms in launches:
            if len(task_ms) == 0:
                continue
            for name, eeg_ms in eeg.items():
                where = 'launch %d, %s' % (launch['launch'], self.sources[name].get('source_file', name))
                fit = self.match(task_ms, eeg_ms, where)
                if fit is None:
                    continue
                fit.update(launch=launch['launch'], source=name,
                           sample_rate=float(self.sources[name]['sample_rate']),
                           n_samples=int(self.sources[name]['n_samples']),
                           n_eeg_pulses=int(len(eeg_ms)))
                fits.setdefault(launch['launch'], []).append(fit)
                self.fits.append(fit)
                logger.info('Sync fit %s: %d of the %d task pulses inside the file matched (%d in '
                            'the launch), slope %+.1f ppm, RMS %.2f ms, max %.2f ms, %d outliers '
                            'dropped' % (where, fit['n_matched'], fit['n_inside'],
                                         fit['n_task_pulses'], (fit['slope'] - 1) * 1e6,
                                         fit['rms_ms'], fit['max_residual_ms'], fit['n_outliers']))
            if launch['launch'] not in fits:
                logger.warn('Launch %d: its %d pulses match no EEG file; its events stay unaligned'
                            % (launch['launch'], len(task_ms)))
        if not fits:
            raise AlignmentError('No launch of %s matches the pulses of any EEG file'
                                 % os.path.basename(self.task_log))

        times = np.asarray(self.events[self.TASK_TIME_FIELD], dtype=float)
        starts = np.array([l['epoch_unix_ms'] for l, _ in launches], dtype=float)
        which = np.searchsorted(starts, times, side='right') - 1
        offsets = np.full(len(times), -1, dtype=np.int64)
        eegfiles = np.array([''] * len(times), dtype=object)
        for k, (launch, _) in enumerate(launches):
            these = np.flatnonzero((which == k) & (times > 0))
            for e in these:
                best = None
                for fit in fits.get(launch['launch'], []):
                    sample = int(round((fit['slope'] * times[e] + fit['offset'])
                                       * fit['sample_rate'] / 1000.))
                    if not 0 <= sample < fit['n_samples']:
                        continue
                    gap = max(fit['task_first'] - times[e], times[e] - fit['task_last'], 0.)
                    if best is None or gap < best[0]:
                        best = (gap, sample, fit['source'])
                if best is not None:
                    offsets[e], eegfiles[e] = best[1], best[2]

        n_bad = int((offsets < 0).sum())
        if n_bad == len(times):
            raise AlignmentError('Could not align any events.')
        if n_bad:
            logger.warn('{} events outside every EEG file (eegoffset -1)'.format(n_bad))
        self.events[self.EEG_TIME_FIELD] = offsets
        self.events[self.EEG_FILE_FIELD] = eegfiles.astype(str)
        return self.events
