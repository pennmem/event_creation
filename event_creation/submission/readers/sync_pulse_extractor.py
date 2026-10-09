"""
Headless sync-pulse extraction from a clinical EEG file, for System 1 sessions.

System 1 aligns the task to the EEG with sync pulses: the task logs every pulse it sends
(``syncPulse`` lines in its log) and the clinical system records them on one or two
ordinary inputs. This module finds the pulses in the recording and writes the sample
index of each one to ``<raw dir>/<recording stem>.sync.txt``, one integer per line: the
file ``transferer.find_sync_file`` and the System 1 aligners read (``System1Aligner``
divides each index by ``sources.json``'s ``sample_rate``). It replaces the PyQt4 GUI in
pulse_extraction.py / pulse_extraction_2.py for these sessions.

What it does
------------
* Reads the sync channel by label from an EDF/EDF+ (pyedflib; preferred: Colorado's
  Nihon Kohden exports) or a Nihon Kohden .EEG + .21E (through NK_reader; untested on
  real files). Labels match loosely: case, spaces, an ``EEG`` prefix, a ``-REF`` suffix and
  leading zeros of the number are ignored. EKG-labelled channels are allowed here
  (EDF_reader drops them from the split, so pulses on EKG inputs must come from here).
* Two labels -> their difference (first minus second): the box's + and - leads on two
  inputs. One label -> that channel.
* Takes the RISING edge of each pulse: the first sample at or above half the pulse height,
  after the baseline (median) is removed. Polarity is detected (the side with the higher
  robust peak; a high-pass filter's undershoot after each pulse is smaller than the
  pulse), so inverted leads need no setting. Hysteresis (the signal must fall below a
  quarter of the height) and a refractory period (``min_interval_ms``, default 200 ms)
  make each 20 ms pulse one edge, whatever its width.
* Writes indices at the rate the split uses, i.e. the rate in sources.json: EDF_reader
  downsamples an EDF whose first channel is at 10 kHz or more to 1 kHz when it splits, so
  edges found at the raw rate are converted (round(index * 1000 / raw rate)); a sync
  channel at a different rate from the split channels is converted by time the same way.

Which channel
-------------
From ``--channel`` (once or twice), else from a sidecar JSON, by default
``<raw dir>/sync_channel.json``::

    {"channels": ["C129", "C130"]}     or     {"channel": "DC01"}

with optional ``"polarity": "auto" | "positive" | "negative"`` and ``"threshold"`` (in the
channel's physical units, after the baseline is removed).

Command line
------------
::

    python -m event_creation.submission.readers.sync_pulse_extractor \\
        <data_root>/R1999A/raw/IPAL1_0            # every .edf/.EEG in the folder
    python -m event_creation.submission.readers.sync_pulse_extractor rec.edf --channel C129 --channel C130

It prints the number of pulses, polarity, threshold, pulse width and interval range per
file, and exits non-zero if any file fails.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

SIDECAR_NAME = 'sync_channel.json'
RAW_PATTERNS = ('*.edf', '*.EDF', '*.eeg', '*.EEG')
# EDF_reader downsamples to this rate when the first channel is at DOWNSAMPLE_FROM or more
DOWNSAMPLE_FROM = 10000
DOWNSAMPLED_RATE = 1000


class SyncExtractionError(Exception):
    pass


def norm_label(label):
    """Loose form of a channel label: 'EEG C0129-Ref' -> 'C129'."""
    s = str(label).upper().replace(' ', '')
    s = re.sub(r'^EEG(?=.)', '', s)
    s = re.sub(r'-?REF$', '', s)
    s = re.sub(r'0+(?=[0-9]+$)', '', s)
    return s


def find_channel(labels, wanted):
    """Index of the one label in ``labels`` that matches ``wanted``."""
    for key in (lambda s: s, lambda s: str(s).upper().replace(' ', ''), norm_label):
        hits = [i for i, l in enumerate(labels) if key(l) == key(wanted)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise SyncExtractionError('channel %r matches %s' % (wanted, [labels[i] for i in hits]))
    raise SyncExtractionError('no channel %r; the file has %s' % (wanted, list(labels)))


def load_sidecar(path):
    with open(path) as f:
        cfg = json.load(f)
    if 'channels' in cfg:
        channels = cfg['channels']
        channels = [channels] if isinstance(channels, str) else list(channels)
    elif 'channel' in cfg:
        channels = [cfg['channel']]
    else:
        raise SyncExtractionError('%s has neither "channels" nor "channel"' % path)
    if not 1 <= len(channels) <= 2:
        raise SyncExtractionError('%s: give one channel or two (+ and -)' % path)
    return dict(channels=channels, polarity=cfg.get('polarity', 'auto'),
                threshold=cfg.get('threshold'))


def _split_rate_edf(reader):
    """The rate EDF_reader splits at (and writes to sources.json)."""
    if reader.getSampleFrequency(0) >= DOWNSAMPLE_FROM:
        return float(DOWNSAMPLED_RATE)
    for i, label in enumerate(reader.getSignalLabels()):
        if label != '' and label[0] != '_' and 'EKG' not in label:
            return float(reader.getSampleFrequency(i))
    return float(reader.getSampleFrequency(0))


def _writable_log_root():
    """eeg_reader imports ..log, which opens <db_root>/protocols/log.txt on import; on a
    machine without a writable db_root (macOS, '/') point it at a temporary folder."""
    if 'event_creation.submission.log' in sys.modules:
        return
    from ..configuration import paths
    protocols = os.path.join(paths.db_root, 'protocols')
    if not (os.access(protocols, os.W_OK) or os.access(paths.db_root, os.W_OK)):
        import tempfile
        paths.set('db_root', tempfile.mkdtemp(prefix='sync_extract_'))


def read_sync_signal(raw_file, channels):
    """The sync trace (one channel, or the difference of two) and its rates.

    :return: (signal, raw_rate, split_rate, labels used)
    """
    ext = os.path.splitext(raw_file)[1].lower()
    if ext == '.edf':
        import pyedflib
        reader = pyedflib.EdfReader(raw_file)
        try:
            labels = reader.getSignalLabels()
            idx = [find_channel(labels, c) for c in channels]
            rates = [float(reader.getSampleFrequency(i)) for i in idx]
            if len(set(rates)) > 1:
                raise SyncExtractionError('sync channels at different rates: %s' % rates)
            signal = reader.readSignal(idx[0]).astype(float)
            if len(idx) == 2:
                signal = signal - reader.readSignal(idx[1]).astype(float)
            return signal, rates[0], _split_rate_edf(reader), [labels[i] for i in idx]
        finally:
            reader.close()
    if ext == '.eeg':
        _writable_log_root()
        from .eeg_reader import NK_reader
        reader = NK_reader(raw_file)
        wanted = {str(c).upper(): i + 1 for i, c in enumerate(channels)}
        data = reader.get_data(wanted, {})
        missing = [c for c, i in wanted.items() if i not in data]
        if missing:
            raise SyncExtractionError('no channel %s in %s' % (missing, raw_file))
        signal = data[1].astype(float)
        if len(channels) == 2:
            signal = signal - data[2].astype(float)
        rate = float(reader.sample_rate)
        return signal, rate, rate, list(channels)
    raise SyncExtractionError('cannot read %s (EDF or Nihon Kohden .EEG only)' % raw_file)


def detect_rising_edges(signal, rate, polarity='auto', threshold=None, min_interval_ms=200.,
                        min_snr=8.):
    """Sample indices of the rising edges of the pulses in ``signal``.

    :return: (indices, info) with info: polarity, threshold, height, noise, width_ms
    """
    x = np.asarray(signal, dtype=float)
    x = x - np.median(x)
    noise = 1.4826 * np.median(np.abs(x))
    floor = noise if noise > 0 else (np.max(np.abs(x)) or 1.0) * 1e-6
    if polarity == 'auto':
        hi = x[x > min_snr * floor]
        lo = -x[x < -min_snr * floor]
        peak_hi = np.percentile(hi, 95) if len(hi) else 0.
        peak_lo = np.percentile(lo, 95) if len(lo) else 0.
        if peak_hi == 0 and peak_lo == 0:
            raise SyncExtractionError('no pulses: nothing above %g x the noise (%.3g)'
                                      % (min_snr, noise))
        polarity = 'positive' if peak_hi >= peak_lo else 'negative'
    if polarity == 'negative':
        x = -x
    elif polarity != 'positive':
        raise SyncExtractionError('polarity must be auto, positive or negative')

    if threshold is None:
        above = x[x > min_snr * floor]
        if not len(above):
            raise SyncExtractionError('no %s pulses above %g x the noise (%.3g)'
                                      % (polarity, min_snr, noise))
        height = np.percentile(above, 95)
        threshold = height / 2.
    else:
        threshold = float(threshold)
        height = 2. * threshold
    rearm = threshold / 2.

    # Upward crossings only: a pulse already high at the first sample has no edge
    up = np.flatnonzero((x[1:] >= threshold) & (x[:-1] < threshold)) + 1
    below = x < rearm
    refractory = int(round(min_interval_ms * rate / 1000.))
    edges, widths = [], []
    last = -np.inf
    for i in up:
        if i - last < refractory:
            continue
        # Hysteresis: since the previous edge the signal must have gone below rearm
        if edges and not below[edges[-1]:i].any():
            continue
        edges.append(int(i))
        last = i
        fall = np.flatnonzero(x[i:] < threshold)
        widths.append((fall[0] if len(fall) else len(x) - i) * 1000. / rate)
    info = dict(polarity=polarity, threshold=float(threshold) * (-1 if polarity == 'negative' else 1),
                height=float(height), noise=float(noise),
                width_ms=float(np.median(widths)) if widths else float('nan'))
    return np.array(edges, dtype=np.int64), info


def default_output(raw_file):
    return os.path.splitext(raw_file)[0] + '.sync.txt'


def extract_sync_pulses(raw_file, channels=None, sidecar=None, out=None, polarity=None,
                        threshold=None, min_interval_ms=200.):
    """Find the pulses in one recording and write ``<stem>.sync.txt``.

    :param channels: one or two labels; default from the sidecar
    :param sidecar: default <raw dir>/sync_channel.json
    :return: a summary dict (also what the command line prints)
    """
    cfg = dict(polarity='auto', threshold=None)
    if not channels:
        sidecar = sidecar or os.path.join(os.path.dirname(os.path.abspath(raw_file)), SIDECAR_NAME)
        if not os.path.exists(sidecar):
            raise SyncExtractionError('no --channel given and no %s' % sidecar)
        cfg = load_sidecar(sidecar)
        channels = cfg['channels']
    polarity = polarity or cfg['polarity'] or 'auto'
    threshold = threshold if threshold is not None else cfg['threshold']

    signal, raw_rate, split_rate, used = read_sync_signal(raw_file, channels)
    edges, info = detect_rising_edges(signal, raw_rate, polarity, threshold, min_interval_ms)
    if raw_rate != split_rate:
        out_idx = np.round(edges * split_rate / raw_rate).astype(np.int64)
    else:
        out_idx = edges
    out = out or default_output(raw_file)
    with open(out, 'w') as f:
        for i in out_idx:
            f.write('%d\n' % i)

    ipi = np.diff(edges) * 1000. / raw_rate
    return dict(file=raw_file, out=out, channels=used, n_pulses=int(len(edges)),
                raw_rate=raw_rate, split_rate=split_rate,
                first_s=float(edges[0] / raw_rate) if len(edges) else None,
                last_s=float(edges[-1] / raw_rate) if len(edges) else None,
                ipi_min_ms=float(ipi.min()) if len(ipi) else None,
                ipi_median_ms=float(np.median(ipi)) if len(ipi) else None,
                ipi_max_ms=float(ipi.max()) if len(ipi) else None, **info)


def raw_files_in(folder):
    files = []
    for pattern in RAW_PATTERNS:
        files.extend(glob.glob(os.path.join(folder, pattern)))
    return sorted(set(files))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Find System 1 sync pulses in a clinical EDF or Nihon Kohden .EEG and '
                    'write <stem>.sync.txt (sample indices at the split rate).')
    ap.add_argument('raw', nargs='+', help='recording(s), or a raw/<exp>_<session> folder')
    ap.add_argument('--channel', action='append', help='sync channel label; twice for + and -')
    ap.add_argument('--sidecar', help='JSON naming the channel(s); default <raw dir>/%s' % SIDECAR_NAME)
    ap.add_argument('--out', help='output file (one recording only)')
    ap.add_argument('--polarity', choices=['auto', 'positive', 'negative'])
    ap.add_argument('--threshold', type=float, help='edge threshold, physical units above baseline')
    ap.add_argument('--min-interval-ms', type=float, default=200.,
                    help='refractory period after an edge (default 200 ms)')
    args = ap.parse_args(argv)

    files = []
    for r in args.raw:
        files.extend(raw_files_in(r) if os.path.isdir(r) else [r])
    if not files:
        ap.error('no .edf/.EEG files in %s' % args.raw)
    if args.out and len(files) > 1:
        ap.error('--out needs exactly one recording')

    failed = 0
    for f in files:
        try:
            s = extract_sync_pulses(f, args.channel, args.sidecar, args.out, args.polarity,
                                    args.threshold, args.min_interval_ms)
        except Exception as e:
            failed += 1
            print('%s: FAILED: %s' % (f, e))
            continue
        print('%(file)s -> %(out)s\n  channels %(channels)s, %(n_pulses)d pulses, %(polarity)s, '
              'threshold %(threshold).4g (noise %(noise).3g), width %(width_ms).1f ms' % s)
        if s['n_pulses']:
            print('  %.1f-%.1f s into the file; intervals %.0f / %.0f / %.0f ms (min/median/max); '
                  'indices at %g Hz (raw %g Hz)' % (s['first_s'], s['last_s'], s['ipi_min_ms'] or 0,
                                                    s['ipi_median_ms'] or 0, s['ipi_max_ms'] or 0,
                                                    s['split_rate'], s['raw_rate']))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
