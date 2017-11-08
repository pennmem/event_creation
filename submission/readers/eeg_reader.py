from __future__ import print_function
import warnings
import struct
import datetime
import sys

import os
import re
import numpy as np
import json
from scipy.linalg import pinv

import tables
import mne

try:
    import pyedflib
except ImportError:
    warnings.warn("pyEDFlib not available")

from .. import fileutil
from ..log import logger
from .nsx_utility.brpylib import NsxFile
from ..exc import EEGError


class EEG_reader(object):

    DATA_FORMAT = 'int16'

    STRFTIME = '%d%b%y_%H%M'
    MAX_CHANNELS = 256

    EPOCH = datetime.datetime.utcfromtimestamp(0)

    def get_start_time(self):
        raise NotImplementedError

    def get_start_time_string(self):
        return self.get_start_time().strftime(self.STRFTIME)

    def get_start_time_ms(self):
        return int((self.get_start_time() - self.EPOCH).total_seconds() * 1000)

    def get_sample_rate(self):
        raise NotImplementedError

    def get_source_file(self):
        raise NotImplementedError

    def get_n_samples(self):
        raise NotImplementedError

    def write_sources(self, location, basename):
        try:
            with open(os.path.join(location, 'sources.json')) as source_file:
                sources = json.load(source_file)
        except:
            sources = {}

        sources[basename] = {
            'name': basename,
            'source_file': os.path.basename(self.get_source_file()),
            'start_time_str': self.get_start_time_string(),
            'start_time_ms': self.get_start_time_ms(),
            'n_samples': self.get_n_samples(),
            'sample_rate': self.get_sample_rate(),
            'data_format': self.DATA_FORMAT
        }

        with fileutil.open_with_perms(os.path.join(location, 'sources.json'), 'w') as source_file:
            json.dump(sources, source_file, indent=2, sort_keys=True)

    def split_data(self, location, basename):
        noreref_location = os.path.join(location, 'noreref')
        if not os.path.exists(noreref_location):
            fileutil.makedirs(noreref_location)
        logger.info("Splitting data into {}/{}".format(noreref_location, basename))
        self._split_data(noreref_location, basename)
        self.write_sources(location, basename)
        logger.info("Splitting complete")

    def _split_data(self, location, basename):
        return NotImplementedError


    def get_matching_jacksheet_dict_label(self, label, jacksheet_dict, channel_map):
        if label in channel_map:
            return channel_map[label]

        label = label.replace(' ', '').upper()
        if label in jacksheet_dict:
            return label

        ref_label = label.replace('-REF', '')
        if ref_label in jacksheet_dict:
            return ref_label
        num_label = re.sub(r'0(?=[0-9]+$)', '', label)
        if num_label in jacksheet_dict:
            return num_label
        num_ref_label = re.sub(r'0(?=[0-9]+$)', '', ref_label)
        if num_ref_label in jacksheet_dict:
            return num_ref_label


class HD5_reader(EEG_reader):

    def __init__(self, hd5_filename, experiment_config_filename, channel_map_filename=None):
        self.raw_filename = hd5_filename
        self.exp_config_filename = experiment_config_filename
        self.exp_config = json.load(open(experiment_config_filename))
        self.sample_rate = self.exp_config['global_settings']['sampling_rate']

        # UNUSED IN SYSTEM 3
        self.channel_map_filesname = channel_map_filename

        self.start_datetime = None
        self.num_samples = None
        self._h5file = None

    @property
    def h5file(self):
        if self._h5file is None:
            logger.debug("Opening {}".format(self.raw_filename))
            self._h5file = tables.open_file(self.raw_filename, mode='r')
        return self._h5file

    @property
    def bipolar(self):
        return 'monopolar_possible' in self.h5file.root and self.h5file.root.monopolar_possible[:]==False

    @property
    def should_split(self):
        return not self.bipolar

    @property
    def by_row(self):
         return 'orient' in self.h5file.root.timeseries.attrs and  self.h5file.root.timeseries.attrs['orient']=='row'

    def get_start_time(self):
        try:
            return datetime.datetime.utcfromtimestamp(self.h5file.root.start_ms.read()/1000.)
        except TypeError:
            return datetime.datetime.utcfromtimestamp(self.h5file.root.start_ms.read()[0]/1000.)

    def get_sample_rate(self):
        return self.sample_rate

    def get_source_file(self):
        return self.raw_filename

    def get_n_samples(self):
        if self.by_row:
            return self.h5file.root.timeseries.shape[0]
        else:
            return self.h5file.root.timeseries.shape[1]

    def write_sources(self, location, basename):
        if self.should_split:
            super(HD5_reader, self).write_sources(location,basename)
        else:
            super(HD5_reader, self).write_sources(location,basename+'.h5')

    def _split_data(self, location, basename):
        if self.should_split:
            time_series = self.h5file.get_node('/','timeseries').read()
            if self.by_row:
                time_series = time_series.T
            if 'bipolar_to_monopolar_matrix' in self.h5file.root:
                transform = self.h5file.root.bipolar_to_monopolar_matrix.read()
                time_series = np.dot(transform,time_series).astype(self.DATA_FORMAT)
            ports = self.h5file.root.ports
            for i, port in enumerate(ports):
                filename = os.path.join(location, basename + ('.%03d' % port))
                data = time_series[i]
                logger.debug("Writing channel {} ({})".format(self.h5file.root.names[i], port))
                logger.debug('len(data):%s'%len(data))
                data.tofile(filename)
        else:
            filename= os.path.join(location,basename+'.h5')
            logger.debug('Moving HD5 file')
            copy(self.raw_filename,filename)


class NK_reader(EEG_reader):

    def __init__(self, nk_filename, jacksheet_filename=None, channel_map_filename=None):
        self.raw_filename = nk_filename
        if jacksheet_filename:
            self.jacksheet = {v:k for k,v in read_jacksheet(jacksheet_filename).items()}
        else:
            self.jacksheet = None

        if channel_map_filename:
            self.channel_map = json.load(open(channel_map_filename))
        else:
            self.channel_map = dict()

        self.sample_rate = None
        self.start_datetime = None
        self.num_samples = None
        self._data = None

    def get_source_file(self):
        return self.raw_filename

    def get_n_samples(self):
        return self.num_samples

    def get_sample_rate(self):
        return self.sample_rate

    def set_jacksheet(self, jacksheet_filename):
        self.jacksheet = {v:k for k,v in read_jacksheet(jacksheet_filename).items()}

    def get_start_time(self):
        with open(self.raw_filename, 'rb') as f:
            # Skipping device block
            deviceBlockLen = 128
            f.seek(deviceBlockLen)

            # Reading EEG1 control block
            _ = f.read(1)  # block ID
            device_type = self.char_(f, 16)  # Device type
            new_format = device_type[:9] == 'EEG-1200A'
            number_of_blocks = self.uint8(f)
            if number_of_blocks > 1:
                raise EEGError('%s EEG2 Control blocks detected' % number_of_blocks)
            block_address = self.int32(f)
            _ = f.read(16)  # EEG control block name

            # Reading EEG2 Control block
            f.seek(block_address, 0)
            _ = f.read(1)  # Block ID
            _ = f.read(16)  # Data format
            number_of_blocks = self.uint8(f)
            if number_of_blocks > 1:
                raise EEGError('%d waveform blocks detected' % number_of_blocks)
            block_address = self.int32(f)
            _ = self.char_(f)  # Name of waveform block

            # Reading waveform block
            f.seek(block_address, 0)
            _ = self.uint8(f)  # Block ID
            _ = self.char_(f,16)  # Data format
            _ = self.uint8(f)  # Data type
            L = self.uint8(f)  # Byte length of one data
            M = self.uint8(f)  # Mark/event flag

            T_year = self.bcd_converter(self.uint8(f)) + 2000
            T_month = self.bcd_converter(self.uint8(f))
            T_day = self.bcd_converter(self.uint8(f))
            T_hour = self.bcd_converter(self.uint8(f))
            T_minute = self.bcd_converter(self.uint8(f))
            T_second = self.bcd_converter(self.uint8(f))

            dt = datetime.datetime(T_year, T_month, T_day, T_hour, T_minute, T_second)
            return dt

    @staticmethod
    def bcd_converter(bits_in):
        x = '{0:08b}'.format(bits_in)
        return 10*int(x[:4],2) + int(x[4:],2)

    @staticmethod
    def fread(f, format, n, size):
        bits = f.read(n * size)
        rtn = struct.unpack(str(n) + format, bits)
        if n == 1:
            return rtn[0]
        else:
            return rtn

    @classmethod
    def int64(cls, f, n=1):
        return cls.fread(f, 'q', n, 8)

    @classmethod
    def uint64(cls, f, n=1):
        return cls.fread(f, 'Q', n, 8)

    @classmethod
    def int64(cls, f, n=1):
        return cls.fread(f, 'Q', n, 8)

    @classmethod
    def uint32(cls, f, n=1):
        return cls.fread(f, 'I', n, 4)

    @classmethod
    def int32(cls, f, n=1):
        return cls.fread(f, 'i', n, 4)

    @classmethod
    def uint16(cls, f, n=1):
        return cls.fread(f, 'H', n, 2)

    @classmethod
    def int16(cls, f, n=1):
        return cls.fread(f, 'h', n, 2)

    @classmethod
    def uint8(cls, f, n=1):
        return cls.fread(f, 'B', n, 1)

    @classmethod
    def char_(cls, f, n=1):
        out = cls.fread(f, 'c', n, 1)
        if len(out) > 1:
            return ''.join(out)
        else:
            return out

    @property
    def labels(self):
        return self.get_labels(elec_file =os.path.splitext(self.raw_filename)[0] + '.21E')

    @classmethod
    def get_labels(cls, elec_file):

        lines = [line.strip() for line in open(elec_file).readlines()]
        end_range = lines.index('[SD_DEF]')
        split_lines = [line.split('=') for line in lines[:end_range] if '=' in line]
        nums_21e, names_21e = zip(*split_lines[:end_range])
        nums_21e = np.array([int(n) for n in nums_21e])
        names_21e = np.array(names_21e)

        channel_order = (range(10) + [22, 23] + range(10, 19) + [20, 21] + range(24, 37) + [74, 75] +
                        range(100, 254) +range(256,321)+ [50, 51])

        channels = []
        for channel in channel_order:
            channels.append(names_21e[nums_21e == channel])
            if channels[-1] == '':
                channels[-1] = ' '

        return {i+1:c[0] for i,c in enumerate(channels)}

    def get_data(self, jacksheet_dict, channel_map):
        eeg_file = self.raw_filename
        elec_file = os.path.splitext(eeg_file)[0] + '.21E'

        with open(eeg_file, 'rb') as f:
            # Skipping device block
            deviceBlockLen = 128
            f.seek(deviceBlockLen)

            # Reading EEG1 control block
            _ = f.read(1)  # block ID
            device_type = self.char_(f, 16)  # Device type
            new_format = device_type[:9] == 'EEG-1200A'
            number_of_blocks = self.uint8(f)
            if number_of_blocks > 1:
                raise EEGError('%s EEG2 Control blocks detected' % number_of_blocks)
            block_address = self.int32(f)
            _ = f.read(16)  # EEG control block name

            # Reading EEG2 Control block
            f.seek(block_address, 0)
            _ = f.read(1)  # Block ID
            _ = f.read(16)  # Data format
            number_of_blocks = self.uint8(f)
            if number_of_blocks > 1:
                raise EEGError('%d waveform blocks detected' % number_of_blocks)
            block_address = self.int32(f)
            _ = f.read(16)  # Name of waveform block

            # Reading waveform block
            f.seek(block_address, 0)
            _ = f.read(1)  # Block ID
            _ = f.read(16)  # Data format
            _ = f.read(1)  # Data type
            L = self.uint8(f)  # Byte length of one data
            M = self.uint8(f)  # Mark/event flag

            T_year = self.uint8(f)
            T_month = self.uint8(f)
            T_day = self.uint8(f)
            T_hour = self.uint8(f)
            T_minute = self.uint8(f)
            T_second = self.uint8(f)

            logger.debug('Date of session: %d/%d/%d\n' % (T_month, T_day, T_year))
            logger.debug('Time of start: %02d:%02d:%02d\n' % (T_hour, T_minute, T_second))

            sample_rate = self.uint16(f)
            sample_rate_conversion = {
                int('C064', 16): 100,
                int('C068', 16): 200,
                int('C068', 16): 500,
                int('C3E8', 16): 1000,
                int('C7D0', 16): 2000,
                int('D388', 16): 5000,
                int('E710', 16): 10000
            }

            if sample_rate in sample_rate_conversion:
                actual_sample_rate = sample_rate_conversion[sample_rate]
                self.sample_rate = actual_sample_rate
            else:
                raise EEGError('Unknown sample rate')

            num_100_ms_blocks = self.uint32(f)
            logger.debug('Length of session: %2.2f hours\n' % (num_100_ms_blocks / 10. / 3600.))
            num_samples = actual_sample_rate * num_100_ms_blocks / 10.
            self.num_samples = num_samples
            ad_off = self.int16(f)
            ad_val = self.uint16(f)
            bit_len = self.uint8(f)
            com_flag = self.uint8(f)
            num_channels = self.uint8(f)

            if (num_channels == 1 and num_samples - actual_sample_rate == 0) and not new_format:
                raise EEGError('Expecting old format, but 1 channel for 1 second')
            elif num_channels > 1 and new_format:
                raise EEGError('Expecting new format, but > 1 channel')

            if new_format:
                logger.debug('NEW FORMAT...')

                waveform_block_old_format = 39 + 10 + 2 * actual_sample_rate + float(M) * actual_sample_rate
                control_block_eeg1_new = 1072
                block_address_eeg2 = block_address + waveform_block_old_format + control_block_eeg1_new

                # EEG2 Format
                add_try = block_address_eeg2
                f.seek(add_try, 0)
                _ = self.uint8(f)  # block ID
                _ = self.char_(f, 16)  # data format
                _ = self.uint16(f)  # # of waveform blocks
                _ = self.char_(f)  # Reserved
                wave_block_new = self.int64(f)  # Address of block 1

                # EEG2 waveform format
                f.seek(wave_block_new, 0)
                _ = self.uint8(f)  # block ID
                _ = self.char_(f, 16)  # Device format
                _ = self.uint8(f)  # Data type
                L = self.uint8(f)  # Byte length of one data
                M = self.uint8(f)  # Mark/event flag

                #  Now things get a little different with the new header
                _ = self.char_(f, 20)  # Start time string
                actual_sample_rate = self.uint32(f)  # Data interval (sample rate)
                self.sample_rate = actual_sample_rate
                num_100_ms_blocks = self.uint64(f)  # Length of session

                num_samples = actual_sample_rate * num_100_ms_blocks / 10
                self.num_samples = num_samples
                ad_off = self.int16(f)  # AD offset at 0V
                ad_val = self.uint16(f)  # ACD val for 1 division
                bit_len = self.uint16(f)  # Bit length of one sample
                com_flag = self.uint16(f)  # Data compression
                reserve_l = self.uint16(f)  # Reserve length
                _ = self.char_(f, reserve_l)  # Reserve data

                num_channels = self.uint32(f)  # Number of RAW recordings

            lines = [line.strip() for line in open(elec_file).readlines()]
            end_range = lines.index('[SD_DEF]')
            split_lines = [line.split('=') for line in lines[:end_range] if '=' in line]
            nums_21e, names_21e = zip(*split_lines[:end_range])
            nums_21e = [int(n) for n in nums_21e]

            channel_order = range(10) + [22, 23] + range(10, 19) + [20, 21] + range(24, 37) + [74, 75] + \
                            range(100, 254) + [50, 51]

            jacksheet_nums = np.arange(len(channel_order)) + 1
            names_21e_ordered = np.chararray(len(channel_order), 16)
            nums_21e_ordered = np.array([-1 for _ in channel_order])
            for i, chan_num in enumerate(channel_order):
                if not chan_num in nums_21e:
                    names_21e_ordered[i] = ''
                    continue
                names_21e_ordered[i] = names_21e[nums_21e.index(chan_num)]
                nums_21e_ordered[i] = chan_num

            jacksheet_nums = jacksheet_nums[names_21e_ordered != '']
            nums_21e_ordered = nums_21e_ordered[names_21e_ordered != '']
            names_21e_ordered = names_21e_ordered[names_21e_ordered != '']

            data_num_to_21e_index = np.zeros(num_channels) - 1

            channel_mask = np.array(
                [not self.get_matching_jacksheet_dict_label(name, jacksheet_dict, self.channel_map) is None
                 for name in names_21e_ordered])
            nums_21e_filtered = nums_21e_ordered[channel_mask]
            names_21e_filtered = names_21e_ordered[channel_mask]
            jacksheet_filtered = [
                jacksheet_dict[self.get_matching_jacksheet_dict_label(name, jacksheet_dict, self.channel_map)]
                for name in names_21e_filtered]

            for e_name, e_num in jacksheet_dict.items():
                if e_num not in jacksheet_filtered:
                    logger.critical("skipping electruode {}: {}".format(e_num, e_name))

            gain = [-1 for _ in range(num_channels)]
            for i in range(num_channels):
                chan = self.int16(f)
                matching_row = (nums_21e_filtered == chan)
                if matching_row.any():
                    data_num_to_21e_index[i] = np.where(matching_row)[0]
                else:
                    data_num_to_21e_index[i] = -1

                f.seek(6, 1)
                chan_sensitivity = self.uint8(f)
                raw_cal = self.uint8(f)
                cal_conversion = {
                    0: 1000.,
                    1: 2.,
                    2: 5.,
                    3: 10.,
                    4: 20.,
                    5: 50.,
                    6: 100.,
                    7: 200.,
                    8: 500.,
                    9: 1000.
                }
                gain[i] = cal_conversion[raw_cal] / float(ad_val)

            if not len(np.unique(gain)) == 1:
                raise EEGError('All channels do not have the same gain')
            self.gain = gain[0]

            if not (np.count_nonzero(data_num_to_21e_index != -1) == len(nums_21e_filtered)):
                bad_indices = [not i in data_num_to_21e_index for i in range(len(names_21e_filtered))]
                bad_names = names_21e_filtered[np.array(bad_indices)]
                good_names = names_21e_filtered[np.logical_not(np.array(bad_indices))]
                unique_bad_names = bad_names[np.array([bad_name not in good_names for bad_name in bad_names])]
                if len(unique_bad_names) > 0:
                    raise EEGError('Could not find recording for channels:\n%s' % unique_bad_names)

            # Get the data!
            logger.debug('Reading...')
            data = np.fromfile(f, 'int16', int((num_channels + 1) * num_samples))
            if len(data) / (num_channels + 1) != num_samples:
                num_samples = len(data) / (num_channels + 1)
                logger.warn(
                    'Number of samples specified in file is wrong. Specified: {}, actual: {}'.format(self.num_samples,
                                                                                                     num_samples))
                self.num_samples = num_samples
            data = data.reshape((int(num_samples), int(num_channels + 1))).T
            # data = np.array(self.uint16(f, int((num_channels + 1) * num_samples))).reshape((int(num_samples), int(num_channels + 1))).T
            logger.debug('Done.')

            # Filter only for relevant channels
            data_num_mask = np.array(data_num_to_21e_index != -1)
            data = data[np.append(data_num_mask, False), :]
            data_num_to_21e_index = data_num_to_21e_index[data_num_mask]

            # Remove offset
            data = data + ad_off
            data_dict = {jacksheet_filtered[int(data_num_to_21e_index[i])]: data[i, :] for i in range(data.shape[0])}
            return data_dict

    def channel_data(self, channel):
        if not self._data:
            if not self.jacksheet:
                raise EEGError("Cannot split EEG without jacksheet")
            self._data = self.get_data(self.jacksheet, self.channel_map)
        return self._data[channel]

    def _split_data(self, location, basename):
        if not os.path.exists(location):
            fileutil.makedirs(location)
        if not self.jacksheet:
            raise EEGError('Jacksheet not specified')
        data = self.get_data(self.jacksheet, self.channel_map)
        if not self.sample_rate:
            raise EEGError('Sample rate not determined')

        sys.stdout.flush()
        for channel, channel_data in data.items():
            filename = os.path.join(location, basename + ('.%03d' % channel))

            logger.debug(channel)
            sys.stdout.flush()
            channel_data.astype(self.DATA_FORMAT).tofile(filename)


class Multi_NSx_reader(EEG_reader):

    def __init__(self, nsx_filenames, jacksheet_filename=None, channel_map_filename=None):
        self.readers = [NSx_reader(filename, jacksheet_filename, i, channel_map_filename)
                        for i, filename in enumerate(nsx_filenames)]

    def get_source_file(self):
        return self.readers[0].get_source_file()

    def get_start_time(self):
        return self.readers[0].get_start_time()

    def get_sample_rate(self):
        sample_rates = np.array([reader.get_sample_rate() for reader in self.readers])
        if len(np.unique(sample_rates)) != 1:
            raise EEGError('Cannot split files with multiple sample rates together: {}'.format(sample_rates))
        return sample_rates[0]

    def get_n_samples(self):
        return min([reader.get_n_samples() for reader in self.readers])

    def write_sources(self, location, basename):
        try:
            with open(os.path.join(location, 'sources.json')) as source_file:
                sources = json.load(source_file)
        except:
            sources = {}

        sources[basename] = {
                'start_time_str': self.get_start_time_string(),
                'start_time_ms': self.get_start_time_ms(),
                'n_samples': self.get_n_samples(),
                'sample_rate': self.get_sample_rate(),
                'source_file': os.path.basename(self.get_source_file()),
                'data_format': self.DATA_FORMAT
                 }
        for i, reader in enumerate(self.readers):
             sources[basename].update({i:
                 {
                'source_file': os.path.join(os.path.basename(os.path.dirname(reader.get_source_file())),
                                            os.path.basename(reader.get_source_file())),
                'start_time_str': reader.get_start_time_string(),
                'start_time_ms': reader.get_start_time_ms(),
                'n_samples': reader.get_n_samples(),
                'sample_rate': reader.get_sample_rate(),
                'data_format': reader.DATA_FORMAT
                 }
            })

        with fileutil.open_with_perms(os.path.join(location, 'sources.json'), 'w') as source_file:
            json.dump(sources, source_file, indent=2, sort_keys=True)

    def _split_data(self, location, basename):
        for reader in self.readers:
            reader._split_data(location, basename)



class NSx_reader(EEG_reader):
    TIC_RATE = 30000

    N_CHANNELS = 128

    SAMPLE_RATES = {
        '.ns2': 1000,
        '.ns5': 30000,
    }


    def __init__(self, nsx_filename, jacksheet_filename=None,  file_number = 0, channel_map_filename=None):
        self.raw_filename = nsx_filename
        self.lowest_channel = file_number * self.N_CHANNELS
        if jacksheet_filename:
            if os.path.splitext(jacksheet_filename)[1] in ('.txt', '.json'):
                self.jacksheet = read_jacksheet(jacksheet_filename)
            else:
                raise NotImplementedError('Non-txt, non-json jacksheet not implemented')
        else:
            self.jacksheet = None
        try:
            self.nsx_info = self.get_nsx_info(nsx_filename)
        except IOError:
            raise IOError("Could not read %s" % nsx_filename)

        if channel_map_filename:
            self.channel_map = json.load(open(channel_map_filename))
        else:
            self.channel_map = dict()

        self._data = None

    def get_source_file(self):
        return self.raw_filename

    def get_start_time(self):
        return self.nsx_info['start_time']

    def get_n_samples(self):
        return self.nsx_info['n_samples']

    def get_sample_rate(self):
        return self.nsx_info['sample_rate']

    @property
    def labels(self):
        return {channel:num for num, channel in self.jacksheet.items()}

    @property
    def data(self):
        if not self._data:
            self._data = self.nsx_info['data'] # I apologize for this.
        return self._data

    def channel_data(self, channel):
        channels = np.array(self.data['elec_ids'])
        return self.data['data'][channels==channel, :]

    @classmethod
    def get_nsx_info(cls, nsx_file):
        reader = NsxFile(nsx_file)
        _, extension = os.path.splitext(nsx_file)
        data = reader.getdata()
        headers = data['data_headers']

        if len(headers) > 1:
            pre_data_points = headers[0]['NumDataPoints']

            # Have to get the data starting at a sample after the last segment starts
            # or reader fills it with all zeros. Because blackrock.
            data_actual = reader.getdata(start_time_s = float(pre_data_points+1)/data['samp_per_s'])
        else:
            pre_data_points = -1
            data_actual = data

        data['data'][:,pre_data_points+1:] = data_actual['data']

        start_time = reader.basic_header['TimeOrigin']
        total_data_points = sum([header['NumDataPoints'] for header in data['data_headers']])
        used_data_points = total_data_points - pre_data_points
        sample_rate = NSx_reader.SAMPLE_RATES[extension]
        length_ms = used_data_points / float(sample_rate) * 1000.
        return {'filename': nsx_file,
                'start_time': start_time,
                'length_ms': length_ms,
                'n_samples': used_data_points,
                'sample_rate': sample_rate,
                'reader': reader,
                'data': data}

    def _split_data(self, location, basename):
        channels = np.array(self.data['elec_ids'])
        buffer_size = self.data['data_headers'][-1]['Timestamp'] / (self.TIC_RATE / self.get_sample_rate())
        for label, channel in self.labels.items():
            recording_channel = channel - self.lowest_channel
            if recording_channel < 0 or not recording_channel in channels:
                logger.debug('Not getting channel {} from file {}'.format(channel, self.raw_filename))
                continue
            filename = os.path.join(location, basename + '.%03d' % channel)
            logger.debug('%s: %s' % (label, channel))
            data = self.data['data'][channels==recording_channel, :].astype(self.DATA_FORMAT)
            if len(data) == 0:
                raise EEGError("EEG File {} contains no data "
                                                   "for channel {}".format(self.raw_filename, recording_channel))
            data = np.concatenate((np.ones((1, buffer_size), self.DATA_FORMAT) * data[0,0], data), 1)
            data.tofile(filename)
            sys.stdout.flush()


class EDF_reader(EEG_reader):

    def __init__(self, edf_filename, jacksheet_filename=None, substitute_raw_file_for_header=None,
                 channel_map_filename=None):
        self.raw_filename = edf_filename
        if jacksheet_filename:
            self.jacksheet = {v:k for k,v in read_jacksheet(jacksheet_filename).items()}
        else:
            self.jacksheet = None
        try:
            self.reader = pyedflib.EdfReader(edf_filename)
        except IOError:
            raise
        self.headers = self.get_channel_info(substitute_raw_file_for_header)

        if channel_map_filename:
            self.channel_map = json.load(open(channel_map_filename))
        else:
            self.channel_map = dict()

    def get_source_file(self):
        return self.raw_filename

    def get_start_time(self):
        return self.reader.getStartdatetime()

    def get_n_samples(self):
        n_samples = np.unique(self.reader.getNSamples())
        n_samples = n_samples[n_samples != 0]
        if len(n_samples) != 1:
            raise EEGError('Could not determine number of samples in file %s' % self.raw_filename)
        return n_samples[0]

    def set_jacksheet(self, jacksheet_filename):
        self.jacksheet = {v:k for k,v in read_jacksheet(jacksheet_filename).items()}

    def get_channel_info(self, substitute_file=None):
        if substitute_file is None:
            reader = self.reader
        else:
            reader = pyedflib.EdfReader(substitute_file)

        headers = {}
        for i in range(self.MAX_CHANNELS):
            header = reader.getSignalHeader(i)
            if header['label']!= '':
                headers[i] = header
        return headers

    def get_sample_rate(self, channels=None):
        sample_rate = None
        for channel, header in self.headers.items():
            if channels and channel not in channels:
                continue
            if header['sample_rate']:
                if not sample_rate:
                    sample_rate = header['sample_rate']
                elif sample_rate != header['sample_rate']:
                    raise EEGError('Different sample rates for recorded channels')
        return sample_rate

    @property
    def labels(self):
        return {i+1:header['label'] for i, header in self.headers.items()}

    def channel_data(self, channel):
        return self.reader.readSignal(channel)


    def _split_data(self, location, basename):
        sys.stdout.flush()
        used_jacksheet_labels = []
        for channel, header in self.headers.items():
            if self.jacksheet:
                label = self.get_matching_jacksheet_dict_label(header['label'], self.jacksheet, self.channel_map)
                if not label:
                    logger.debug("skipping channel {}".format(header['label']))
                    continue
                if label.upper() in self.jacksheet:
                    out_channel = self.jacksheet[label.upper()]
                    used_jacksheet_labels.append(label.upper())
                elif label in self.jacksheet:
                    out_channel = self.jacksheet[label]
                    used_jacksheet_labels.append(label)
                else:
                    logger.debug("skipping channel {}".format(label))
            else:
                out_channel = channel
            filename = os.path.join(location, basename + '.%03d' % (out_channel))

            logger.debug('{}: {}'.format(out_channel, header['label']))
            sys.stdout.flush()
            data = self.reader.readSignal(channel).astype(self.DATA_FORMAT)
            data.tofile(filename)
        if self.jacksheet:
            for label in self.jacksheet:
                if label not in used_jacksheet_labels:
                    logger.critical("label {} not split! Potentially missing data!".format(label))


class ScalpReader(EEG_reader):
    """
    A universal reader for all scalp lab recordings. This reader has support for reading from EGI's .mff
    formats, as well as BioSemi's .bdf format.
    """
    def __init__(self, raw_filename, unused_jacksheet=None):
        """
        :param raw_filename: The file path to the .raw.bz2 file containing the EEG recording from the session.
        :param unused_jacksheet: Exists only because the function get_eeg_reader() automatically passes a jacksheet
        parameter to any reader it creates, even though scalp EEG studies do not use jacksheets.
        """
        self.raw_filename = raw_filename
        self.start_datetime = None
        self.names = None
        self.data = None
        self.filetype = None

    def get_data(self):
        """
        Unpacks the binary in the raw data file to get the voltage data from all channels. In the process, it must
        unzip the data file before use. After reading the data, it removes the unzipped file, leaving only the
        zipped version (or zips the unzipped file if no zipped version exists).

        Note that when MNE reads in a raw data file, it automatically converts the signal to volts.
        """
        ext = os.path.splitext(self.raw_filename)[1].lower()
        is_mff = ext == '.mff'
        already_unzipped = ext != '.bz2'
        # If EEG file is zipped, determine the absolute path of the EEG file before and after unzipping
        if not already_unzipped:
            original_path = os.path.abspath(os.path.join(os.path.dirname(self.raw_filename), os.readlink(self.raw_filename)))
            unzip_path = original_path[:-4]  # remove '.bz2' from end of file name
            already_unzipped = os.path.isfile(unzip_path)
            logger.debug('Unzipping EEG data file ' + self.raw_filename)
            self.filetype = os.path.splitext(unzip_path)[1].lower()
        else:
            unzip_path = self.raw_filename
            self.filetype = ext

        # If data file is already unzipped, use it; otherwise unzip the file for reading
        if is_mff or already_unzipped or os.system('bunzip2 ' + original_path.replace(' ', '\ ')) == 0:
            try:
                logger.debug('Parsing EEG data file ' + unzip_path)
                # Read an EGI recording
                if self.filetype in ('.mff', '.raw'):
                    # self.data = mne.io.read_raw_egi(unzip_path, eog=['E8', 'E25', 'E126', 'E127'], preload=True)
                    self.data = mne.io.read_raw_egi(unzip_path, eog=['E8', 'E25', 'E126', 'E127'], preload=False)
                    self.data.info['description'] = 'system: GSN-HydroCel-129'
                    # Correct the name of channel 129 to Cz
                    self.data.rename_channels({'E129': 'Cz'})
                    self.data.set_montage(mne.channels.read_montage('GSN-HydroCel-129'))
                # Read a BioSemi recording
                elif self.filetype == '.bdf':
                    # self.data = mne.io.read_raw_edf(unzip_path, eog=['EXG1', 'EXG2', 'EXG3', 'EXG4'], misc=['EXG5', 'EXG6', 'EXG7', 'EXG8'], montage='biosemi128', preload=True)
                    self.data = mne.io.read_raw_edf(unzip_path, eog=['EXG1', 'EXG2', 'EXG3', 'EXG4'], misc=['EXG5', 'EXG6', 'EXG7', 'EXG8'], montage='biosemi128', preload=False)
                    self.data.info['description'] = 'system: biosemi128'
                else:
                    logger.critical('Unsupported EEG file type for file %s!' % unzip_path)
                logger.debug('Finished parsing EEG data.')
                # Pull relevant header info
                self.start_datetime = datetime.datetime.utcfromtimestamp(self.data.info['meas_date'])
                self.names = [str(x) for x in self.data.info['ch_names']]
            except:
                logger.critical('Unable to parse EEG data file!')

            """
            # Remove unzipped file if zipped version exists, otherwise zip the file. Leave .mff files alone.
            if already_unzipped:
                pass
            elif os.path.isfile(original_path):
                os.system('rm ' + unzip_path.replace(' ', '\ '))
            else:
                os.system('bzip2 ' + unzip_path.replace(' ', '\ '))
            logger.debug('Finished getting EEG data.')
            """

        # If unzip attempt fails, return error
        else:
            logger.critical('Unzipping failed! Unable to read data file!')

    def postprocess(self):
        """
        Post-process EEG data. Currently, this involves the following:
        - Running a .1 Hz high pass filter on all EEG and EOG channels
        - Generating a common average reference projection on the MNE Raw object, which can later be used to
        re-reference the data
        """
        try:
            # Get indices of EEG and EOG channels
            picks_eeg_eog = mne.pick_types(self.data.info, eeg=True, eog=True)
            # Run a .1 Hz high pass filter on all EEG and EOG channels
            logger.debug('Running .1 Hz highpass filter on all channels.')
            self.data.filter(.1, None, picks=picks_eeg_eog, method='iir', phase='zero-double',
                             l_trans_bandwidth='auto', h_trans_bandwidth='auto')
            # Set common average reference
            self.data.set_eeg_reference(ref_channels=None)
        except:
            logger.critical('Failed to post-process EEG!')

    def run_ica(self, save_path):
        """
        Run ICA on entire session using mne's ICA class. Then identify artifactual components based on high correlation
        with EOG channels, low skewness, low kurtosis, or high variance (via mne's detect_artifacts workflow).

        :param save_path: The filepath where the ICA solution will be saved. To conform with MNE standards, this path
        should end with "-ica.fif".
        """
        eog_chans = None
        if self.filetype in ('.mff', '.raw'):
            eog_chans = ['E8', 'E25', 'E126', 'E127']
        elif self.filetype == '.bdf':
            eog_chans = ['EXG1', 'EXG2', 'EXG3', 'EXG4']
        else:
            logger.critical('Unsupported EEG filetype!')

        # Run ICA and check for artifactual components based on EOG correlation, skewness, kurtosis, and variance
        logger.debug('Running ICA')
        # Down-sample to 512 Hz prior to running ICA. Otherwise, ICA requires 70-80 GB of RAM and will run for hours.
        orig_sfreq = self.data.info['sfreq']
        if orig_sfreq > 512:
            self.data.resample(512)
        ica = mne.preprocessing.ICA(method='fastica')
        ica.fit(self.data, picks=mne.pick_types(self.data.info, eeg=True, eog=True))
        ica.detect_artifacts(self.data, eog_ch=eog_chans)
        ica.info['sfreq'] = orig_sfreq
        ica.save(save_path)  # Save ICA object to a .fif file

        # Save the mixing matrix if we detect that it will be calculated differently upon loading
        # (possible MNE bug, see https://github.com/mne-tools/mne-python/issues/4374)
        if not np.allclose(ica.mixing_matrix_, pinv(ica.unmixing_matrix_)):
            np.save(os.path.join(os.path.dirname(save_path), 'mixing_mat.npy'), ica.mixing_matrix_)
            logger.warn('Possible MNE bug detected, in which mixing matrix will change upon reloading the ICA object. '
                        'Saving true mixing matrix separately for safety.')

    def split_data(self, location, basename):
        """
        This function runs the full EEG post-processing regimen on the recording. Note that "split data" is a misnomer
        for the ScalpReader, as EEG data is no longer split into separate channel files. Rather, the ScalpReader uses
        MNE's post-processing tools and saves the results to .fif files.

        :param location: A string denoting the directory in which the channel files are to be written
        :param basename: The string used to name the processed EEG file. To conform with MNE standards, "-raw.fif" will
        be appended to the path for the EEG save file and "-ica.fif" will be appended to the path for the ICA save file.
        """
        logger.info("Post-processing EEG data into {}/{}".format(location, basename))
        # Load data if we have not already done so
        if self.data is None:
            self.get_data()
        os.symlink(os.path.abspath(os.path.join(os.path.dirname(self.raw_filename), os.readlink(self.raw_filename))), os.path.join(location, basename))
        """
        # Run post-processing regimen on the loaded data
        self.postprocess()

        # Save post-processed data to .fif file
        raw_filename = os.path.join(location, basename + '-raw.fif')
        self.data.save(raw_filename, fmt='single')

        # Run ICA, mark bad components, and save ICA solution to file
        ica_filename = os.path.join(location, basename + '-ica.fif')
        self.run_ica(ica_filename)
        """
        self.write_sources(location, basename)

    def get_start_time(self):
        # Read header info if have not already done so, as the header contains the start time info
        if self.start_datetime is None:
            self.get_data()
        return self.start_datetime

    def get_start_time_string(self):
        return self.get_start_time().strftime(self.STRFTIME)

    def get_start_time_ms(self):
        return int((self.get_start_time() - self.EPOCH).total_seconds() * 1000)

    def get_sample_rate(self):
        if self.data is None:
            self.get_data()
        return self.data.info['sfreq']

    def get_source_file(self):
        return self.raw_filename

    def get_n_samples(self):
        if self.data is None:
            self.get_data()
        return self.data.n_times


def read_jacksheet(filename):
    [_, ext] = os.path.splitext(filename)
    if ext.lower() == '.txt':
        return read_text_jacksheet(filename)
    elif ext.lower() == '.json':
        return read_json_jacksheet(filename)
    else:
        raise NotImplementedError


def read_text_jacksheet(filename):
    lines = [line.strip().split() for line in open(filename).readlines()]
    return {int(line[0]): line[1] for line in lines}


def read_json_jacksheet(filename):
    json_load = json.load(open(filename))
    contacts = json_load.values()[0]['contacts']
    if contacts is None:
        raise Exception("Contacts.json has 'None' for contact list. Rerun localization")
    jacksheet = {int(v['channel']): k for k, v in contacts.items()}
    return jacksheet


def calc_gain(amp_info, amp_fact):
    """
    Calculates the gain factor for converting raw EEG to uV.

    :param amp_info: Info to convert from raw to voltage
    :param amp_fact: Amplification factor to correct for
    :return: Gain factor for converting raw data to uV.
    """
    arange = abs(np.diff(amp_info[0])[0])
    drange = abs(np.diff(amp_info[1])[0])
    if np.diff(abs(amp_info[0]))[0] == 0 and np.diff(abs(amp_info[1]))[0] == 0:
        return (drange * 1000000./amp_fact) / arange
    else:
        logger.warn('WARNING: Amp info ranges were not centered at zero.\nNo gain calculation was possible.')
        return 1


READERS = {
    '.edf': EDF_reader,
    '.eeg': NK_reader,
    '.ns2': NSx_reader,
    '.raw': ScalpReader,
    '.mff': ScalpReader,
    '.bdf': ScalpReader,
    '.h5': HD5_reader
}


def get_eeg_reader(raw_filename, jacksheet_filename=None, **kwargs):
    if isinstance(raw_filename, list):
        return Multi_NSx_reader(raw_filename, jacksheet_filename)
    else:
        file_type = os.path.splitext(raw_filename)[1].lower()
        # If the data file is compressed, get the extension before the .bz2
        file_type = os.path.splitext(os.path.splitext(raw_filename)[0])[1].lower() if file_type == '.bz2' else file_type
        return READERS[file_type](raw_filename, jacksheet_filename, **kwargs)


if __name__ == '__main__':
    hd5_reader = HD5_reader("/Users/iped/event_creation/tests/test_input/R9999X/behavioral/FR1/session_0/host_pc/20170106_115912/eeg_timeseries.h5",
                            "/Users/iped/event_creation/tests/test_input/R9999X/behavioral/FR1/session_0/host_pc/20170106_115912/experiment_config.json")
    output = '/Users/iped/event_creation/tests/test_output/R9999X/eeg_output/'
    hd5_reader.split_data(output, 'TEST')
