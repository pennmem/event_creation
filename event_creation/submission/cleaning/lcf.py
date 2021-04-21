import os
import mne
import numpy as np
from glob import glob
from scipy import linalg
from cluster_helper.cluster import cluster_view
from ..log import logger


def run_lcf(events, eeg_dict, ephys_dir, method='fastica', highpass_freq=.5, iqr_thresh=3, lcf_winsize=.2):
    """
    Runs localized component filtering (DelPozo-Banos & Weidemann, 2017) to clean artifacts from EEG data. Cleaned data
    is written to a new file in the ephys directory for the session. The pipeline is as follows, repeated for each
    EEG recording the session had:

    1) Drop EOG and trigger channels from the data.
    2) High-pass filter the data to eliminate baseline drift.
    3) Split the recording into partitions, separated by mid-session breaks. A new partition begins immediately after
        each mid-session break. Also mark break periods for subsequent exclusion while fitting ICA.
    4) Detect and drop bad channels separately for each partition.
    5) Apply a common average reference to the data.
    6) Run ICA on the data, with break periods and pre- and post-session EEG data excluded from this fitting process.
    7) Use ICA solution to decompose data into its sources.
    8) Remove artifacts from each source using localized component filtering.
    9) Reconstruct the original channels from the cleaned sources.
    10) Use the cleaned data to interpolate the bad channels that were dropped prior to ICA.
    11) Concatenate the partitions of data back into a single, continuous time series.
    12) Save the cleaned session data to a .fif file.

    To illustrate how this function divides up a session, a session with 2 breaks would have 3 parts:
    1) Start of recording -> End of break 1
    2) End of break 1 -> End of break 2
    3) End of break 2 -> End of recording

    Note that this function uses ipython-cluster-helper to clean all partitions of the session in parallel.

    :param events: Events structure for the session.
    :param eeg_dict: Dictionary mapping the basename of each EEG recording for the session to an MNE raw object
        containing the data from that recording.
    :param ephys_dir: File path of the current_processed EEG directory for the session.
    :param method: String defining which ICA algorithm to use (fastica, infomax, extended-infomax, picard).
    :param highpass_freq: The frequency in Hz at which to high-pass filter the data prior to ICA (recommended >= .5)
    :param iqr_thresh: The number of interquartile ranges above the 75th percentile or below the 25th percentile that a
        sample must be for LCF to mark it as artifactual.
    :param lcf_winsize: The width (in seconds) of the LCF dilator and transition windows.
    :return: None
    """
    # Loop over each of the session's EEG recordings
    for eegfile in eeg_dict:

        ##########
        #
        # Initialization
        #
        ##########

        basename, filetype = os.path.splitext(eegfile)
        logger.debug('Cleaning data from {}'.format(basename))
        clean_eegfile = os.path.join(ephys_dir, '%s_clean_raw.fif' % basename)

        # Make sure LCF hasn't already been run for this file (prevents re-running LCF during math event creation)
        if os.path.exists(clean_eegfile):
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

        # The recording for LTP346-12 was left on for several hours after the experiment ended. Truncate the recording
        # to avoid memory errors.
        if basename.endswith('LTP346_session_12'):
            eeg.crop(0, 5050)

        # Drop all channels except EEG data channels. Manually specify channels for BioSemi in case E, F, G, and H
        # electrodes were accidentally included in the recording.
        if filetype == '.bdf':
            eeg_chans = ['A%i' % i for i in range(1, 33)] + ['B%i' % i for i in range(1, 33)] +\
                        ['C%i' % i for i in range(1, 33)] + ['D%i' % i for i in range(1, 33)]

            eeg.pick_channels(eeg_chans)
        else:
            eeg.pick_types(eeg=True, eog=False, misc=False, stim=False)

        # High-pass filter the data, since LCF will not work properly if the baseline voltage shifts
        eeg.filter(highpass_freq, None, fir_design='firwin')

        ##########
        #
        # Identification of session breaks
        #
        ##########

        # Mark all time points before and after the session for exclusion
        onsets = []
        offsets = []
        sess_start_in_recording = False
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
        break_start_idx = np.where(evs.type == 'BREAK_START')[0]
        break_stop_idx = np.where(evs.type == 'BREAK_STOP')[0]

        # Handling for PyEPL studies (only break starts are logged)
        if len(rest_rewet_idx) > 0:
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
            # In this case, the start of the recording is set as the onset.
            if evs[0].type == 'BREAK_STOP':
                onsets.append(0)
                offsets.append(evs[0].eegoffset)
                break_stop_idx = break_stop_idx[1:]
            # If the recording ends in the middle of a break, the last event will be a break start.
            # In this case, set the last time point in the recording as the offset.
            if evs[-1].type == 'BREAK_START':
                onsets.append(evs[-1].eegoffset)
                offsets.append(eeg.n_times-1)
                break_start_idx = break_start_idx[:-1]
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
        if sess_start_in_recording:
            if len(offsets) > 1:
                split_samples = offsets[1:]
            else:
                split_samples = []
        else:
            split_samples = offsets

        # If session has no breaks, just treat the entire recording as one partition
        if len(split_samples) == 0:
            split_samples.append(eeg.n_times - 1)

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

            d = dict(index=i, basename=basename, ephys_dir=ephys_dir, method=method, iqr_thresh=iqr_thresh,
                     lcf_winsize=lcf_winsize)
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

        # Load cleaned EEG data partitions and remove the temporary partition files and their subfiles (.fif files are
        # broken into multiple 2 GB subfiles)
        clean = []
        for d in inputs:
            index = d['index']
            clean_partfile = os.path.join(ephys_dir, '%s_clean%i_raw.fif' % (basename, index))
            clean.append(mne.io.read_raw_fif(clean_partfile, preload=True))
            for subfile in glob(os.path.join(ephys_dir, '%s_clean%i_raw*.fif' % (basename, index))):
                os.remove(subfile)

        # Concatenate the cleaned partitions of the recording back together
        logger.debug('Constructing cleaned data file for {}'.format(basename))
        clean = mne.concatenate_raws(clean)
        logger.debug('Saving cleaned data for {}'.format(basename))

        ##########
        #
        # Data saving
        #
        ##########

        # Save data to a .fif file and set permissions to read only
        clean.save(clean_eegfile, fmt='single')
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
    import mkl
    mkl.set_num_threads(1)
    import os
    import mne
    import numpy as np
    import pandas as pd
    import scipy.stats as ss
    import scipy.signal as sp_signal
    from nolds import hurst_rs
    from ..log import logger

    def detect_bad_channels(eeg, index, basename, ephys_dir, ignore=None):
        """
        Runs several bad channel detection tests, records the test scores in a TSV file, and saves the list of bad
        channels to a text file. The detection methods are as follows:

        1) Log-transformed variance of the channel. The variance is useful for identifying both flat channels and
        extremely noisy channels. Because variance has a log-normal distribution across channels, log-transforming the
        variance allows for more reliable detection of outliers.

        2) Hurst exponent of the channel. The Hurst exponent is a measure of the long-range dependency of a time series.
        As physiological signals consistently have similar Hurst exponents, channels with extreme deviations from this
        value are unlikely to be measuring physiological activity.

        A third method used to look for high voltage offset from the reference channel. This corresponds to the
        electrode offset screen in BioSemi's ActiView, and can be used to identify channels with poor connection to the
        scalp. The percent of the recording during which the voltage offset exceeds 40 mV would be calculated for each
        channel. Any channel that spent more than 25% of the total duration of the partition above this offset
        threshold would be marked as bad. Unfortunately, this method does not work after high-pass filtering, due to the
        centering of each channel around 0. Because sessions are now partitioned for bad channel detection, high pass
        filtering must be done prior to all bad channel detection, resulting in this method no longer working.

        Note that high-pass filtering is required prior to calculating the variance and Hurst exponent of each channel,
        as baseline drift will artificially increase the variance and invalidate the Hurst exponent.

        Following bad channel detection, two bad channel files are created. The first is a file named
        <eegfile_basename>_bad_chan<index>.txt (where index is the partition number for that part of the session), and
        is a text file containing the names of all channels that were identifed as bad. The second is a tab-separated
        values (.tsv) file called <eegfile_basename>_bad_chan_info<index>.tsv, which contains the actual detection
        scores for each EEG channel.

        :param eeg: An mne Raw object containing the EEG data to run bad channel detection on.
        :param index: The partition number of this part of the session. Used so that each parallel job writes a
            different bad channel file.
        :param basename: The basename of the EEG recording. Used for naming bad channel files in a consistent manner.
        :param ephys_dir: The path to the ephys directory for the session.
        :param ignore: A boolean array indicating whether each time point in the EEG signal should be excluded/ignored
            during bad channel detection.
        :return: A list containing the string names of each bad channel.
        """
        logger.debug('Identifying bad channels for part %i of %s' % (index, basename))

        # Set thresholds for bad channel criteria (see docstring for details on how these were optimized)
        low_var_th = -3  # If z-scored log variance < -3, channel is most likely flat
        high_var_th = 3  # If z-scored log variance > 3, channel is likely too noisy
        hurst_th = 3  # If z-scored Hurst exponent > 3, channel is unlikely to be physiological
        n_chans = eeg._data.shape[0]

        """
        # Deprecated Method 1: Percent of samples with a high voltage offset (>40 mV) from the reference channel
        # Does not work after high-pass filtering
        offset_th = .04  # Samples over ~40 mV (.04 V) indicate poor contact with the scalp (BioSemi only)
        offset_rate_th = .25  # If >25% of the recording partition has poor scalp contact, mark as bad (BioSemi only)
        if filetype == '.bdf':
            if ignore is None:
                ref_offset = np.mean(np.abs(eeg._data) > offset_th, axis=1)
            else:
                ref_offset = np.mean(np.abs(eeg._data[:, ~ignore]) > offset_th, axis=1)
        else:
            ref_offset = np.zeros(n_chans)
        """

        # Method 1: High or low log-transformed variance
        if ignore is None:
            var = np.log(np.var(eeg._data, axis=1))
        else:
            var = np.log(np.var(eeg._data[:, ~ignore], axis=1))
        zvar = ss.zscore(var)

        # Method 2: High Hurst exponent
        hurst = np.zeros(n_chans)
        for i in range(n_chans):
            if ignore is None:
                hurst[i] = hurst_rs(eeg._data[i, :])
            else:
                hurst[i] = hurst_rs(eeg._data[i, ~ignore])
        zhurst = ss.zscore(hurst)

        # Identify bad channels using optimized thresholds
        bad = (zvar < low_var_th) | (zvar > high_var_th) | (zhurst > hurst_th)
        badch = np.array(eeg.ch_names)[bad]

        # Save list of bad channels to a text file
        badchan_file = os.path.join(ephys_dir, basename + '_bad_chan%i.txt' % index)
        np.savetxt(badchan_file, badch, fmt='%s')
        os.chmod(badchan_file, 0644)

        # Save a TSV file with extended info about each channel's scores
        badchan_file = os.path.join(ephys_dir, basename + '_bad_chan_info%i.tsv' % index)
        with open(badchan_file, 'w') as f:
            f.write('name\tlog_var\thurst\tbad\n')
            for i, ch in enumerate(eeg.ch_names):
                f.write('%s\t%f\t%f\t%i\n' % (ch, var[i], hurst[i], bad[i]))
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
        data = np.dot(linalg.pinv(ica.pca_components_), data)

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
    method = inputs['method']
    iqr_thresh = inputs['iqr_thresh']
    lcf_winsize = inputs['lcf_winsize']

    # Load temporary split EEG file and delete it and its subfiles (.fif files are broken into 2 GB subfiles)
    split_eeg_path = os.path.join(ephys_dir, '%s_%i_raw.fif' % (basename, index))
    eeg = mne.io.read_raw_fif(split_eeg_path, preload=True)
    for subfile in glob(os.path.join(ephys_dir, '%s_%i_raw*.fif' % (basename, index))):
        os.remove(subfile)

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

    # Rank of the data decreases by 1 after common average reference, and an additional 1 for each excluded/interpolated
    # channel. Reduce the number of PCA/ICA components accordingly, or else you may get components that are identical
    # with opposite polarities. See url for details: https://sccn.ucsd.edu/wiki/Chapter_09:_Decomposing_Data_Using_ICA
    n_components = len(eeg.ch_names)
    eeg.info['bads'] = detect_bad_channels(eeg, index, basename, ephys_dir, ignore=ignore)
    eeg.set_eeg_reference(projection=False)
    n_components -= 1 + len(eeg.info['bads'])

    ######
    # ICA
    ######

    # Run ICA for the current partition of the session. Note that ICA automatically excludes bad channels.
    logger.debug('Running ICA (part %i) on %s' % (index, basename))
    ica = mne.preprocessing.ICA(method=method, max_pca_components=n_components)
    ica.fit(eeg, reject_by_annotation=True)

    ######
    # LCF
    ######

    logger.debug('Running LCF (part %i) on %s' % (index, basename))
    # Convert data to sources
    S = ica.get_sources(eeg)._data

    # Clean artifacts from sources using LCF
    S = lcf(S, S, eeg.info['sfreq'], iqr_thresh, lcf_winsize, lcf_winsize, ignore=ignore)

    # Reconstruct data from cleaned sources
    eeg._data.fill(0)
    good_idx = mne.pick_channels(eeg.ch_names, [], exclude=eeg.info['bads'])
    eeg._data[good_idx, :] = reconstruct_signal(S, ica)

    # Interpolate bad channels
    eeg.interpolate_bads(reset_bads=True, mode='accurate')

    # Save clean data from current partition of session
    clean_eegfile = os.path.join(ephys_dir, '%s_clean%i_raw.fif' % (basename, index))
    eeg.save(clean_eegfile, overwrite=True)
