import os
import mne
import numpy as np
from ptsa.data.TimeSeriesX import TimeSeriesX
from cluster_helper.cluster import cluster_view
from ..log import logger


def run_lcf(events, eeg_dict, ephys_dir, method='fastica', highpass_freq=.5, badchan_method=None, iqr_thresh=3, lcf_winsize=.2):
    """
    Runs localized component filtering (DelPozo-Banos & Weidemann, 2017) to clean artifacts from EEG data. Cleaned data
    is written to a new file in the ephys directory for the session. The pipeline is as follows, repeated for each
    EEG recording the session had:

    1) Drop EOG channels and (optionally) bad channels from the data.
    2) High-pass filter the data.
    3) Common average re-reference the data.
    4) Run ICA on the data. A new ICA is fit after each session break, while excluding the time points before the
        session, after the session, and during breaks. The final saved file will still contain all data points, but the
        actual ICA solutions will not be influenced by breaks.
    5) Remove artifacts using LCF, paired with the ICA solutions calculated in #4.
    6) Save a cleaned version of the EEG data to a .fif file.

    To illustrate how this function divides up a session, a session with 2 breaks would have 3 parts:
    1) Start of recording -> End of break 1
    2) End of break 1 -> End of break 2
    3) End of break 2 -> End of recording

    Note that his function uses ipython-cluster-helper to clean all partitions of the session in parallel.

    :param events: Events structure for the session.
    :param eeg_dict: Dictionary mapping the basename of each EEG recording for the session to an MNE raw object
        containing the data from that recording.
    :param ephys_dir: File path of the current_processed EEG directory for the session.
    :param method: String defining which ICA algorithm to use (fastica, infomax, extended-infomax, picard).
    :param highpass_freq: The frequency in Hz at which to high-pass filter the data prior to ICA (recommended >= .5)
    :param badchan_method: If 'interpolate', uses the spherical spline interpolation method in MNE to repair bad
    channels before re-referencing and running LCF. If 'exclude', drops bad channels from the data prior to
    re-referencing and LCF, meaning that bad channels will be missing entirely from the cleaned data. If 'reref',
    excludes bad channels from the calculation of the common average reference, but leaves them in during LCF. If None
    or 'none', do not attempt to automatically determine bad channels, and just leave all channels in.
    :param iqr_thresh: The number of interquartile ranges above the 75th percentile or below the 25th percentile that a
        sample must be for LCF to mark it as artifactual.
    :param lcf_winsize: The width (in seconds) of the LCF dilator and transition windows.
    :return: None
    """

    # Loop over all of the session's EEG recordings
    for eegfile in eeg_dict:

        ##########
        #
        # Initialization
        #
        ##########

        basename, filetype = os.path.splitext(eegfile)
        logger.debug('Cleaning data from {}'.format(basename))

        # Make sure LCF hasn't already been run for this file (prevents re-running LCF during math event creation)
        # TODO: Revert to search for _clean.h5
        if os.path.exists(os.path.join(ephys_dir, '%s_clean_5.h5' % basename )):
            continue

        # Select EEG data and events from current recording
        eeg = eeg_dict[eegfile]
        samp_rate = eeg.info['sfreq']
        evs = events[events.eegfile == os.path.join(ephys_dir, eegfile)]
        if len(evs) == 0:
            continue

        ##########
        #
        # EEG Pre-Processing
        #
        ##########

        # Drop EOG channels
        eeg.pick_types(eeg=True, eog=False)

        # High-pass filter the data, since LCF will not work properly if the baseline voltage shifts
        eeg.filter(highpass_freq, None, fir_design='firwin')

        ##########
        #
        # Identification of session breaks
        #
        ##########

        onsets = []
        offsets = []
        sess_start_in_recording = False
        # Mark all time points before and after the session for exclusion
        if evs[0].type == 'SESS_START':
            sess_start_in_recording = True
            onsets.append(0)
            offsets.append(evs[0].eegoffset)
        if evs[-1].type == 'SESS_END':
            onsets.append(evs[-1].eegoffset)
            offsets.append(eeg.n_times - 1)

        # Mark breaks for exclusion
        # Identify break start/stop times. PyEPL used REST_REWET events; UnityEPL uses BREAK_START/STOP.
        rest_rewet_idx = np.where(evs.type == 'REST_REWET')[0]
        break_start_idx = np.where(evs[:-1].type == 'BREAK_START')[0]
        break_stop_idx = np.where(evs[1:].type == 'BREAK_STOP')[0]

        # Handling for PyEPL studies (only break starts are logged)
        if len(rest_rewet_idx > 0):
            onsets = np.concatenate((evs[rest_rewet_idx].eegoffset, onsets))
            for i, idx in enumerate(rest_rewet_idx):
                # If break is the final event in the current recording, set the offset as the final sample
                # Otherwise, set the offset as 8 seconds before the first event following the break
                if len(evs) == idx + 1:
                    o = eeg.n_times - 1
                else:
                    o = int(evs[idx + 1].eegoffset - 8 * samp_rate)
                # Make sure that offsets cannot occur before onsets (happens if a break lasts less than 5 seconds)
                if o <= onsets[i]:
                    o = onsets[i] + 1
                offsets.append(o)

        # Handling for UnityEPL studies (break starts and stops are both logged)
        elif len(break_start_idx) > 0:
            # If the recordings starts in the middle of a break, the first event will be a break stop.
            # In this case, the break onset is set as the start of the recording.
            if evs[0].type == 'BREAK_STOP':
                onsets.append(0)
                offsets.append(evs[0].eegoffset)
            # If the recording ends in the middle of a break, the last event will be a break start.
            # In this case, set the break offset as the last time point in the recording.
            if evs[-1].type == 'BREAK_START':
                onsets.append(evs[-1].eegoffset)
                offsets.append(eeg.n_times-1)
            # All other break starts and stops are contained fully within the recording
            for i, idx in enumerate(break_start_idx):
                onsets.append(evs[idx].eegoffset)
                offsets.append(evs[break_stop_idx[i]].eegoffset)

        # Annotate the EEG object with the timings of excluded periods (pre-session, post-session, & breaks)
        onsets = np.sort(onsets)
        offsets = np.sort(offsets)
        onset_times = eeg.times[onsets]
        offset_times = eeg.times[offsets]
        durations = offset_times - onset_times
        descriptions = ['bad_break' for _ in onsets]
        annotations = mne.Annotations(eeg.times[onsets], durations, descriptions)
        eeg.set_annotations(annotations)

        # Skip over the offset for the session start when splitting, as we only want to split after breaks
        split_samples = offsets[1:] if sess_start_in_recording else offsets

        # Create file with information on partitions and when breaks occur
        partition_info = [['index', 'start_sample', 'end_sample']]
        start = 0
        for i, stop in enumerate(split_samples):
            partition_info.append([str(i), str(start), str(stop)])  # Add partition number and start and end samples
            for j, onset in enumerate(onsets):  # List session break periods that occur in that partition
                if start <= onset < stop:
                    partition_info[-1].append(str(onsets[j]))
                    partition_info[-1].append(str(offsets[j]))
            start = stop + 1
        # Add extra columns based on the maximum number of break periods that occurs in any one partition
        ncol = max([len(row) for row in partition_info[1:]])
        for i, row in enumerate(partition_info):
            if i == 0:  # Header row
                j = 0
                while len(partition_info[0]) < ncol:
                    partition_info[0] += ['skip_start%i' % j, 'skip_end%i' % j]
                    j += 1
            else:  # Data rows
                partition_info[i] += ['' for _ in range(ncol - len(partition_info[i]))]
            partition_info[i] = '\t'.join(partition_info[i])
        partition_info = '\n'.join(partition_info)

        # Write break/partition info to a tsv file
        breakfile_path = os.path.join(ephys_dir, '%s_breaks.tsv' % basename)
        with open(breakfile_path, 'w') as f:
            f.write(partition_info)
        os.chmod(breakfile_path, 0644)

        ##########
        #
        # ICA + LCF
        #
        ##########

        # Create inputs for running LCF in parallel with ipython-cluster-helper
        inputs = []
        start = 0
        for i, stop in enumerate(split_samples):

            d = dict(index=i, basename=basename, ephys_dir=ephys_dir, filetype=filetype, method=method,
                     iqr_thresh=iqr_thresh, lcf_winsize=lcf_winsize, badchan_method=badchan_method)
            inputs.append(d)

            # Copy the session data, then crop it down to one part of the session. Save it temporarily for parallel jobs
            split_eeg = eeg.copy()
            split_eeg.crop(split_eeg.times[start], split_eeg.times[stop])
            split_eeg_path = os.path.join(ephys_dir, '%s_%i_raw.fif' % (basename, i))
            split_eeg.save(split_eeg_path, overwrite=True)

            # Set the next part of the EEG recording to begin one sample after the current one
            start = stop + 1

        # Run ICA and then LCF on each part of the sesion in parallel. Sometimes cluster helper returns errors even
        # when successful, so avoid crashing event creation if an error comes up here.
        try:
            with cluster_view(scheduler='sge', queue='RAM.q', num_jobs=len(inputs), cores_per_job=6) as view:
                view.map(run_split_lcf, inputs)
        except Exception:
            logger.warn('Cluster helper returned an error. This may happen even if LCF was successful, so attempting to'
                        ' continue anyway...')

        # Load cleaned EEG data partitions
        for iqr_thresh in (range(1, 6)):
            clean = []
            for d in inputs:
                index = d['index']
                clean_eegfile = os.path.join(ephys_dir, '%s_%i_clean_%s_raw.fif' % (basename, index, iqr_thresh))
                clean.append(mne.io.read_raw_fif(clean_eegfile, preload=True))
                os.remove(clean_eegfile)

            # Concatenate the cleaned partitions of the recording back together
            logger.debug('Constructing cleaned data file for {}'.format(basename))
            clean = mne.concatenate_raws(clean)
            logger.debug('Saving cleaned data for {}'.format(basename))

            ##########
            #
            # Data saving
            #
            ##########

            # Save cleaned version of data to hdf as a TimeSeriesX object
            clean_eegfile = os.path.join(ephys_dir, '%s_clean_%s.h5' % (basename, iqr_thresh))
            TimeSeriesX(clean._data.astype(np.float32), dims=('channels', 'time'),
                        coords={'channels': clean.info['ch_names'], 'time': clean.times,
                                'samplerate': clean.info['sfreq']}).to_hdf(clean_eegfile)
            os.chmod(clean_eegfile, 0644)

            del clean


def run_split_lcf(inputs):
    """
    Runs ICA followed by LCF on one partition of the session. Note that this function has been designed to work as a
    parallel job managed via ipython-cluster-helper. As such, all necessary imports and function declarations have to be
    made within this function, leading to the odd code structure here. This is also why inputs must be passed as a
    dictionary.

    :param inputs: A dictionary specifying the "index" of the partition (for coordination with other parallel jobs), the
        "basename" of the EEG recording, the "ephys_dir" path to the current_processed folder, the "method" of ICA to
        use, the "iqr_thresh" IQR threshold to use for LCF, and the "lcf_winsize" to be used for LCF.
    :return: An MNE raw object containing the cleaned version of the data.
    """
    import os
    import mne
    import numpy as np
    import pandas as pd
    import scipy.stats as ss
    import scipy.signal as sp_signal
    from nolds import hurst_rs
    from ..log import logger

    def detect_bad_channels(eeg, index, basename, ephys_dir, filetype, ignore=None):
        """
        Runs several bad channel detection tests, records the test scores in a TSV file, and saves the list of bad
        channels to a text file. The detection methods are as follows:

        1) High voltage offset from the reference channel. This corresponds to the electrode offset screen in BioSemi's
        ActiView, and can be used to identify channels with poor connection to the scalp. The percent of the recording
        during which the voltage offset exceeds 30 mV is calculated for each channel. Any channel that spends more than
        15% of the total duration of the recording above this offset threshold is marked as bad.

        2) Log-transformed variance of the channel. The variance is useful for identifying both flat channels and
        extremely noisy channels. Because variance has a log-normal distribution across channels, log-transforming the
        variance allows for more reliable detection of outliers.

        3) Hurst exponent of the channel. The Hurst exponent is a measure of the long-range dependency of a time series.
        As physiological signals consistently have a Hurst exponent of around .7, channels with extreme deviations from
        this value are unlikely to be measuring physiological activity.

        Note that high-pass filtering is required prior to calculating the variance and Hurst exponent of each channel,
        as baseline drift will artificially increase the variance and invalidate the Hurst exponent.

        Through parameter optimization, it was found that channels should be marked as bad if they have a z-scored Hurst
        exponent greater than 3.1 or z-scored log variance less than -1.9 or greater than 1.7. This combination of
        thresholds, alongside the voltage offset test, successfully identified ~80.5% of bad channels with a false
        positive rate of ~2.9% when tested on a set of 20 manually-annotated sessions. It was additionally found that
        marking bad channels based on a low Hurst exponent failed to identify any channels that had not already marked
        by the log-transformed variance test. Similarly, marking channels that were poorly correlated with other
        channels as bad (see the "FASTER" method by Nolan, Whelan, and Reilly (2010)) was an accurate metric, but did
        not improve the hit rate beyond what the log-transformed variance and Hurst exponent could achieve on their own.

        Optimization was performed using a simple grid search of z-score threshold combinations for the different bad
        channel detection methods, with the goal of optimizing the trade-off between hit rate and false positive rate
        (hit_rate - false_positive_rate). The false positive rate was weighted at either 5 or 10 times the hit rate, to
        strongly penalize the system for throwing out good channels (both weightings produced similar optimal
        thresholds).

        Following bad channel detection, two bad channel files are created. The first is a file named
        <eegfile_basename>_bad_chan.txt, and is a text file containing the names of all channels that were identifed
        as bad. The second is a tab-separated values (.tsv) file called <eegfile_basename>_bad_chan_info.tsv, which
        contains the actual detection scores for each EEG channel.

        :return: None
        """
        logger.debug('Identifying bad channels for part %i of %s' % (index, basename))

        # Set thresholds for bad channel criteria (see docstring for details on how these were optimized)
        offset_th = .03  # Samples over ~30 mV (.03 V) indicate poor contact with the scalp (BioSemi only)
        offset_rate_th = .15  # If >15% of the recording has poor scalp contact, mark as bad (BioSemi only)
        low_var_th = -2  # If z-scored log variance < -2, channel is most likely flat
        high_var_th = 2  # If z-scored log variance > 2, channel is likely too noisy to analyze
        hurst_th = 3  # If z-scored Hurst exponent > 3, channel is unlikely to be physiological

        n_chans = eeg._data.shape[0]
        # Method 1: Percent of samples with a high voltage offset (>30 mV) from the reference channel
        if filetype == '.bdf':
            if ignore is None:
                ref_offset = np.mean(np.abs(eeg._data) > offset_th, axis=1)
            else:
                ref_offset = np.mean(np.abs(eeg._data[:, ~ignore]) > offset_th, axis=1)
        else:
            ref_offset = np.zeros(n_chans)

        # Method 2: High or low log-transformed variance
        if ignore is None:
            var = np.log(np.var(eeg._data, axis=1))
        else:
            var = np.log(np.var(eeg._data[:, ~ignore], axis=1))
        zvar = ss.zscore(var)

        # Method 3: High Hurst exponent
        hurst = np.zeros(n_chans)
        for i in range(n_chans):
            if ignore is None:
                hurst[i] = hurst_rs(eeg._data[i, :])
            else:
                hurst[i] = hurst_rs(eeg._data[i, ~ignore])
        zhurst = ss.zscore(hurst)

        # Identify bad channels using optimized thresholds
        bad = (ref_offset > offset_rate_th) | (zvar < low_var_th) | (zvar > high_var_th) | (zhurst > hurst_th)
        badch = np.array(eeg.ch_names)[bad]

        # Save list of bad channels to a text file
        badchan_file = os.path.join(ephys_dir, basename + '_bad_chan%i.txt' % index)
        np.savetxt(badchan_file, badch, fmt='%s')
        os.chmod(badchan_file, 0644)

        # Save a TSV file with extended info about each channel's scores
        badchan_file = os.path.join(ephys_dir, basename + '_bad_chan_info%i.tsv' % index)
        with open(badchan_file, 'w') as f:
            f.write('name\thigh_offset_rate\tlog_var\thurst\tbad\n')
            for i, ch in enumerate(eeg.ch_names):
                f.write('%s\t%f\t%f\t%f\t%i\n' % (ch, ref_offset[i], var[i], hurst[i], bad[i]))
        os.chmod(badchan_file, 0644)

        return badch.tolist()

    def lcf(S, feat, sfreq, iqr_thresh, dilator_width, transition_width, ignore=None):

        dilator_width = int(dilator_width * sfreq)
        transition_width = int(transition_width * sfreq)

        ##########
        #
        # Classification
        #
        ##########
        # Find interquartile range of each component
        if ignore is None:
            p75 = np.percentile(feat, 75, axis=1)
            p25 = np.percentile(feat, 25, axis=1)
        else:
            p75 = np.percentile(feat[:, ~ignore], 75, axis=1)
            p25 = np.percentile(feat[:, ~ignore], 25, axis=1)
        iqr = p75 - p25

        # Tune artifact thresholds for each component according to the IQR and the iqr_thresh parameter
        pos_thresh = p75 + iqr * iqr_thresh
        neg_thresh = p25 - iqr * iqr_thresh

        # Detect artifacts using the IQR threshold. Dilate the detected zones to account for the mixer transition equation
        ctrl_signal = np.zeros(feat.shape, dtype=float)
        dilator = np.ones(dilator_width)
        for i in range(ctrl_signal.shape[0]):
            ctrl_signal[i, :] = (feat[i, :] > pos_thresh[i]) | (feat[i, :] < neg_thresh[i])
            ctrl_signal[i, :] = np.convolve(ctrl_signal[i, :], dilator, 'same')
        del p75, p25, iqr, pos_thresh, neg_thresh, dilator

        # Binarize signal
        ctrl_signal = (ctrl_signal > 0).astype(float)

        ##########
        #
        # Mixing
        #
        ##########

        # Allocate normalized transition window
        trans_win = sp_signal.hann(transition_width, True)
        trans_win /= trans_win.sum()

        # Pad extremes of control signal
        pad_width = [tuple([0, 0])] * ctrl_signal.ndim
        pad_size = int(transition_width / 2 + 1)
        pad_width[1] = (pad_size, pad_size)
        ctrl_signal = np.pad(ctrl_signal, tuple(pad_width), mode='edge')
        del pad_width

        # Combine the transition window and the control signal to build a final transition-control signal, which can be applied to the components
        for i in range(ctrl_signal.shape[0]):
            ctrl_signal[i, :] = np.convolve(ctrl_signal[i, :], trans_win, 'same')
        del trans_win

        # Remove padding from transition-control signal
        rm_pad_slice = [slice(None)] * ctrl_signal.ndim
        rm_pad_slice[1] = slice(pad_size, -pad_size)
        ctrl_signal = ctrl_signal[rm_pad_slice]
        del rm_pad_slice, pad_size

        # Mix sources with control signal to get cleaned sources
        S_clean = S * (1 - ctrl_signal)

        return S_clean

    def reconstruct_signal(sources, ica):

        # Mix sources to translate back into PCA components (PCA components x Time)
        data = np.dot(ica.mixing_matrix_, sources)

        # Mix PCA components to translate back into original EEG channels (Channels x Time)
        data = np.dot(np.linalg.inv(ica.pca_components_), data)

        # Invert transformations that MNE performs prior to PCA
        data += ica.pca_mean_[:, None]
        data *= ica.pre_whitener_

        return data

    ######
    # Initialization
    ######

    # Pull parameters out of input dictionary
    index = inputs['index']
    basename = inputs['basename']
    ephys_dir = inputs['ephys_dir']
    filetype = inputs['filetype']
    method = inputs['method']
    iqr_thresh = inputs['iqr_thresh']
    lcf_winsize = inputs['lcf_winsize']
    badchan_method = inputs['badchan_method']

    # Load temporary split EEG file and delete it
    split_eeg_path = os.path.join(ephys_dir, '%s_%i_raw.fif' % (basename, index))
    eeg = mne.io.read_raw_fif(split_eeg_path, preload=True)
    # os.remove(split_eeg_path)

    # Read locations of breaks and create a mask for use in leaving breaks out of IQR calculation
    ignore = np.zeros(eeg._data.shape[1], dtype=bool)
    breakfile_path = os.path.join(ephys_dir, '%s_breaks.tsv' % basename)
    breaks = pd.read_csv(breakfile_path, delimiter='\t')
    breaks = breaks.loc[index]
    start = int(breaks['start_sample'])
    i = 0
    while 'skip_start%i' % i in breaks and not np.isnan(breaks['skip_start%i' % i]):
        skip_start = int(breaks['skip_start%i' % i]) - start
        skip_end = int(breaks['skip_end%i' % i]) - start
        ignore[skip_start:skip_end+1] = True
        i += 1

    ######
    # Pre-processing
    ######

    if badchan_method == 'interpolate':  # Repair bad channels using spherical spline interpolation
        eeg.info['bads'] = detect_bad_channels(eeg, index, basename, ephys_dir, filetype, ignore=ignore)
        eeg.interpolate_bads(reset_bads=True, mode='accurate')
        eeg.set_eeg_reference(projection=False)
    elif badchan_method == 'exclude':  # Drop bad channels from entire process; they will not be present in cleaned data
        eeg.info['bads'] = detect_bad_channels(eeg, index, basename, ephys_dir, filetype, ignore=ignore)
        eeg.set_eeg_reference(projection=False)
    elif badchan_method == 'reref':  # Leave bad channels out of the common average, but don't exclude from ICA/LCF
        eeg.info['bads'] = detect_bad_channels(eeg, index, basename, ephys_dir, filetype, ignore=ignore)
        eeg.set_eeg_reference(projection=False)
        eeg.info['bads'] = []
    elif badchan_method == 'None' or badchan_method is None:  # Skip bad channel detection and just re-reference data
        eeg.set_eeg_reference(projection=False)
    else:
        raise ValueError('%s is an invalid setting for badchan_method. Must be "interpolate", "exclude", "reref", or '
                         'None.' % badchan_method)

    ######
    # ICA
    ######

    # Run (or load) ICA for the current part of the session
    ica_path = os.path.join(ephys_dir, '%s_%i-ica.fif' % (basename, index))
    if os.path.exists(ica_path):
        logger.debug('Loading ICA (part %i) for %s' % (index, basename))
        ica = mne.preprocessing.read_ica(ica_path)
    else:
        logger.debug('Running ICA (part %i) on %s' % (index, basename))
        ica = mne.preprocessing.ICA(method=method)
        ica.fit(eeg, reject_by_annotation=True)
        ica.save(ica_path)

    ######
    # LCF
    ######

    logger.debug('Running LCF (part %i) on %s' % (index, basename))
    # Convert data to sources
    S = ica.get_sources(eeg)._data
    lcf_winsize = .2
    for iqr_thresh in range(1, 6):

        # Clean artifacts from sources using LCF
        cS = lcf(S, S, eeg.info['sfreq'], iqr_thresh, lcf_winsize, lcf_winsize, ignore=ignore)

        # Reconstruct data from cleaned sources
        eeg._data = reconstruct_signal(cS, ica)

        clean_eegfile = os.path.join(ephys_dir, '%s_%i_clean_%s_raw.fif' % (basename, index, iqr_thresh))
        eeg.save(clean_eegfile, overwrite=True)
