import numpy as np
from scipy.signal import butter, filtfilt


def butter_filt(data, freq_range, sample_rate=500, filt_type='bandstop', order=4):
    """
    Designs and runs an Nth order digital butterworth filter on an array of data.

    NOTE: In order to match the original MATLAB implementation of the filtfilt function, the padlen argument must be
    set to 3*(max(len(Ab), len(Bb))-1)). Default padlen in SciPy is 3*max(len(a), len(b)), which will cause it to return
    different values from our MATLAB scripts if not manually set (i.e. default padlen for SciPy is always 3 greater than
    the default padlen for MATLAB).

    :param data: An array containing the data to be filtered
    :param freq_range: A single frequency to filter on or a 2D matrix where each row is a frequency range to filter on
    :param sample_rate: The sampling rate of the EEG recording
    :param filt_type: The type of filter to run - can be 'bandstop', 'highpass', 'lowpass', or 'bandpass'
    :param order: The order of the filter
    :return: The filtered data
    """
    # Calculate Nyquist frequency
    nyq = sample_rate/2.
    # Convert the frequency range to a 2D matrix if it was input as an integer
    freq_range = np.array([[freq_range]]) if isinstance(freq_range, (int, float)) else np.array(freq_range)
    freq_range = freq_range / nyq
    # Get the Butterworth values and run the filter for zero phase distortion
    for i in range(np.shape(freq_range)[0]):
        Bb, Ab = butter(order, freq_range[i, :], btype=filt_type)
        pad = 3 * (max(len(Ab), len(Bb)) - 1)  # calculate padlen like MATLAB instead of SciPy
        data = filtfilt(Bb, Ab, data, padlen=pad)
    return data
