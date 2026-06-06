"""Tests for submission/helpers.py butter_filt()."""
import numpy as np

from ..submission import helpers


def _tone(freq, fs, n=2000):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq * t)


def test_bandstop_attenuates_target_frequency():
    fs = 500
    # 60 Hz tone + 10 Hz tone
    sig = _tone(60, fs) + _tone(10, fs)
    out = helpers.butter_filt(sig, [[58, 62]], sample_rate=fs, filt_type="bandstop")
    # 60 Hz component should be attenuated -> lower overall power
    assert np.std(out) < np.std(sig)


def test_scalar_freq_range_highpass():
    fs = 500
    sig = _tone(2, fs) + _tone(100, fs)
    out = helpers.butter_filt(sig, 50, sample_rate=fs, filt_type="highpass")
    # low-frequency (2 Hz) energy removed
    assert np.std(out) < np.std(sig)


def test_lowpass_passes_low_freq():
    fs = 500
    sig = _tone(5, fs)
    out = helpers.butter_filt(sig, 50, sample_rate=fs, filt_type="lowpass")
    # passband signal largely preserved
    assert np.std(out) > 0.5 * np.std(sig)


def test_bandpass_multiple_ranges():
    fs = 500
    sig = _tone(20, fs)
    out = helpers.butter_filt(sig, [[10, 30]], sample_rate=fs, filt_type="bandpass")
    assert out.shape == sig.shape
