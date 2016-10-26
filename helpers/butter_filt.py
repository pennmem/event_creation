from scipy.signal import butter, filtfilt


def butter_filt(data, freq_range=(58, 62), sample_rate=256, filt_type='bandstop', order=4):
    """
    Designs and runs an Nth order digital butterworth filter on an array of data.

    NOTE: In order to match the original MATLAB implementation of the filtfilt function, the padlen argument must be
    set to 3. Default padlen in SciPy is 6, which will cause it to filter the data differently from our old MATLAB
    scripts if not set to 3.

    :param data: An array containing the data to be filtered
    :param freq_range: The range of the filter
    :param sample_rate: The sampling rate of the EEG recording
    :param filt_type: The type of filter to run - can be 'bandstop', 'highpass', 'lowpass', or 'bandpass'
    :param order: The order of the filter
    :return: The filtered data
    """
    # Calculate Nyquist frequency
    nyq = sample_rate/2.
    # Get the Butterworth values and run the filter for zero phase distortion
    freq_range = [freq_range] if isinstance(freq_range, (int, float)) else freq_range
    for i in range(len(freq_range)):
        Bb, Ab = butter(order, freq_range[i]/nyq, btype=filt_type)
        data = filtfilt(Bb, Ab, data, padlen=3)  # PADLEN MUST BE SET TO 3 TO MATCH MATLAB IMPLEMENTATION
    return data