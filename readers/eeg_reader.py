from __future__ import print_function
import warnings
try:
    import pyedflib
except ImportError:
    warnings.warn("pyEDFlib not available")
import os
import re
import numpy as np
import json
from readers.nsx_utility.brpylib import NsxFile
import struct
import datetime
import sys
import bz2
import files
from loggers import logger
from shutil import copy
from helpers.butter_filt import butter_filt

class UnSplittableEEGFileException(Exception):
    pass

class EEG_reader:

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

        with files.open_with_perms(os.path.join(location, 'sources.json'), 'w') as source_file:
            json.dump(sources, source_file, indent=2, sort_keys=True)

    def split_data(self, location, basename):
        noreref_location = os.path.join(location, 'noreref')
        if not os.path.exists(noreref_location):
            files.makedirs(noreref_location)
        logger.info("Splitting data into {}/{}".format(location, basename))
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
                raise UnSplittableEEGFileException('%s EEG2 Control blocks detected' % number_of_blocks)
            block_address = self.int32(f)
            _ = f.read(16)  # EEG control block name

            # Reading EEG2 Control block
            f.seek(block_address, 0)
            _ = f.read(1)  # Block ID
            _ = f.read(16)  # Data format
            number_of_blocks = self.uint8(f)
            if number_of_blocks > 1:
                raise UnSplittableEEGFileException('%d waveform blocks detected' % number_of_blocks)
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

        channel_order = range(10) + [22, 23] + range(10, 19) + [20, 21] + range(24, 37) + [74, 75] + \
                        range(100, 254) + [50, 51]

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
                raise UnSplittableEEGFileException('%s EEG2 Control blocks detected' % number_of_blocks)
            block_address = self.int32(f)
            _ = f.read(16)  # EEG control block name

            # Reading EEG2 Control block
            f.seek(block_address, 0)
            _ = f.read(1)  # Block ID
            _ = f.read(16)  # Data format
            number_of_blocks = self.uint8(f)
            if number_of_blocks > 1:
                raise UnSplittableEEGFileException('%d waveform blocks detected' % number_of_blocks)
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
                raise UnSplittableEEGFileException('Unknown sample rate')

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
                raise UnSplittableEEGFileException('Expecting old format, but 1 channel for 1 second')
            elif num_channels > 1 and new_format:
                raise UnSplittableEEGFileException('Expecting new format, but > 1 channel')

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
                raise UnSplittableEEGFileException('All channels do not have the same gain')
            self.gain = gain[0]

            if not (np.count_nonzero(data_num_to_21e_index != -1) == len(nums_21e_filtered)):
                bad_indices = [not i in data_num_to_21e_index for i in range(len(names_21e_filtered))]
                bad_names = names_21e_filtered[np.array(bad_indices)]
                good_names = names_21e_filtered[np.logical_not(np.array(bad_indices))]
                unique_bad_names = bad_names[np.array([bad_name not in good_names for bad_name in bad_names])]
                if len(unique_bad_names) > 0:
                    raise UnSplittableEEGFileException('Could not find recording for channels:\n%s' % unique_bad_names)

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
                raise UnSplittableEEGFileException("Cannot split EEG without jacksheet")
            self._data = self.get_data(self.jacksheet, self.channel_map)
        return self._data[channel]

    def _split_data(self, location, basename):
        if not os.path.exists(location):
            files.makedirs(location)
        if not self.jacksheet:
            raise UnSplittableEEGFileException('Jacksheet not specified')
        data = self.get_data(self.jacksheet, self.channel_map)
        if not self.sample_rate:
            raise UnSplittableEEGFileException('Sample rate not determined')

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
            raise UnSplittableEEGFileException('Cannot split files with multiple sample rates together: {}'.format(sample_rates))
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

        with files.open_with_perms(os.path.join(location, 'sources.json'), 'w') as source_file:
            json.dump(sources, source_file, indent=2, sort_keys=True)

    def _split_data(self, location, basename):
        for reader in self.readers:
            reader._split_data(location, basename)


class NSx_reader(EEG_reader):
    TIC_RATE = 30000

    N_CHANNELS = 128

    SAMPLE_RATES = {
        '.ns2': 1000
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
                raise UnSplittableEEGFileException("EEG File {} contains no data "
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
            raise UnSplittableEEGFileException('Could not determine number of samples in file %s' % self.raw_filename)
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
                    raise UnSplittableEEGFileException('Different sample rates for recorded channels')
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
                out_channel = self.jacksheet[label]
                used_jacksheet_labels.append(label)
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


class EGI_reader(EEG_reader):
    """
    Parses the EEG sample data from a .raw.bz2 file. The raw file begins with a header with a series of information
    such as the version number, start time, and sample rate of the recording. The data samples immediately follow the
    header. Because the data is in compressed .bz2 format, bytes must be decompressed as the file is read using Ptyhon's
    built-in BZ2File class. Full information on the format of EGI .raw files can be found online at:
    https://sccn.ucsd.edu/eeglab/testfiles/EGI/NEWTESTING/rawformat.pdf

    DATA FIELDS:
    raw_filename: The path to the .raw.bz2 file containing the EEG data for the session.
    basename: The string used to name the channel files once split (typically subj_DDMonYY_HHMM).
    noreref_loc: The path to the noreref directory.
    data: Holds the unpacked and split EEG data.
    header: A dictionary where the header info from the beginning of the raw file is stored.
    header_names: A mapping of header names to the format string required for unpacking that header item.
    amp_gain: The gain factor
    data_format: A string stating the format in which the output split channel files will be written.
    """
    def __init__(self, raw_filename, unused_jacksheet):
        self.raw_filename = raw_filename
        self.basename = ''
        self.noreref_loc = ''
        self.start_datetime = None
        self._data = None
        self.header = {}
        self.header_names = (('version', '>l'), ('year', '>h'), ('month', '>h'), ('day', '>h'), ('hour', '>h'),
                             ('minute', '>h'), ('second', '>h'), ('msec', '>l'), ('sample_rate', '>h'),
                             ('num_channels', '>h'), ('gain', '>h'), ('bits', '>h'), ('amp_range', '>h'),
                             ('num_samples', '>l'), ('num_events', '>h'))
        self.amp_gain = 1.
        self.data_format = 'int16'

    def read_header(self):
        """
        Parses the header information from an EGI file. The header consists of the first 36 + (4 * num_events) bytes of
        the file. The format of each header is specified in self.header_names: '>l' is a 4-byte big-endian signed long,
        and '>h' is a 2-byte big-endian signed short. The header fields are as follows (in order):

        Version Number: 2, 4, or 6 depending on whether the data is formatted as shorts, singles, or doubles
        Year: The start year of the recording
        Month: The start month of the recording
        Day: The start day of the recording
        Hour: The start hour of the recording
        Minute: The start minute of the recording
        Second: The start second of the recording
        Millisecond: The start millisecond of the recording
        Sample Rate: The sampling rate of the recording, in Hz
        Number of Channels: The number of electrodes being recorded from
        Gain: The board gain; can be either 1, 2, 4, or 8
        Bits: The number of conversion bits
        Amplifier Range: The range of the amplifier, in uV
        Number of Samples: The number of samples per channel in the recording
        Number of Events: The number of event channels, typically the sync pulse channel(s)
        Event Names: The four-character label for each event (e.g. 'DIN1', 'DI15')
        """
        with bz2.BZ2File(self.raw_filename, 'rb') as raw_file:
            # Read header info; each pair from self.header_names contains the name of the header and the format to be
            # used by struct.unpack(); '>l' unpacks a long, '>h' unpacks a short.
            for pair in self.header_names:
                bytes_to_read = 2 if pair[1] == '>h' else 4
                self.header[pair[0]] = struct.unpack(pair[1], raw_file.read(bytes_to_read))[0]

            # Read event codes
            self.header['event_codes'] = np.empty(self.header['num_events'], dtype='S4')
            for i in range(0, self.header['num_events']):
                # Each event code is four characters long
                code = struct.unpack('>4c', raw_file.read(4))
                code = (x.rstrip() for x in code)
                code = ''.join(code)
                self.header['event_codes'][i] = code

            self.start_datetime = datetime.datetime(self.header['year'], self.header['month'], self.header['day'],
                                                    self.header['hour'], self.header['minute'], self.header['second'])

            # Log various information about the file
            logger.debug('EEG File Information:')
            logger.debug('---------------------')
            logger.debug('Sample Rate = %d' % self.header['sample_rate'])
            logger.debug('Start of recording = %d/%d/%d %02d:%02d' % (self.header['month'], self.header['day'],
                                                                      self.header['year'], self.header['hour'],
                                                                      self.header['minute']))
            logger.debug('Number of Channels = %d' % self.header['num_channels'])
            logger.debug('Number of Events = %d' % self.header['num_events'])

    def get_data(self):
        """
        Parses the data samples from an EGI file. The first data sample begins at index 36 + (4 * num_events), and there
        are a total of (num_channels + num_events) * num_samples data samples in the file. Each sample is either 2, 4,
        or 8 bytes long, depending on the version number. Typically, for our data files, it is 4 bytes per sample. All
        data is signed and big-endian.
        """
        # Read header info if have not already done so
        if not self.header:
            self.read_header()

        # Determine whether the EEG data is formatted as shorts, singles, or doubles and set the unpacking format
        # accordingly. Version 2 = short, 4 = single, 6 = double.
        eeg_format_map = {2: ('>h', 2), 4: ('>f', 4), 6: ('>d', 8)}
        fmt, bytes_per_sample = eeg_format_map[self.header['version']] if self.header['version'] in eeg_format_map \
            else (None, None)
        if not fmt:
            raise Exception('Unknown EGI format %d' % self.header['version'])

        # Calculate total number of samples to read
        total_samples = (self.header['num_channels'] + self.header['num_events']) * self.header['num_samples']

        # Calculate the gain factor for converting raw EEG data to uV
        # amp_info = np.array(((-32767., 32767.), (-2.5, 2.5)))
        # amp_fact = 1000.
        # self.amp_gain = calc_gain(amp_info, amp_fact)

        self.amp_gain = .0762963  # Gain is always this value for EGI - no need to calculate it

        # Limit the number of samples that are copied at once, to reduce memory usage
        step_size = 1000000
        total_read = 0
        raw = np.zeros((total_samples, 1))
        logger.debug('Loading %d samples...' % total_samples)
        with bz2.BZ2File(self.raw_filename, 'rb') as raw_file:
            # Go to index for the beginning of the EEG samples in the raw file
            data_start_index = 36 + 4 * self.header['num_events']
            raw_file.seek(data_start_index)

            # Read samples in blocks of step_size, until all samples have been read
            while total_read < total_samples:
                samples_left = total_samples - total_read
                samples_to_read = samples_left if samples_left < step_size else step_size
                unpacked_samples = struct.unpack(fmt[0] + str(samples_to_read) + fmt[1],
                                                 raw_file.read(samples_to_read * bytes_per_sample))
                samples_array = np.array(unpacked_samples)
                raw[total_read:total_read+samples_to_read] = np.reshape(samples_array, (samples_to_read, 1))
                total_read += samples_to_read
            logger.debug('%d...Done' % total_read)

        # Organize the data into a matrix with each channel and event on its own row
        self._data = raw.reshape((self.header['num_channels'] + self.header['num_events'], self.header['num_samples']),
                                 order='F')

        # Run a first-order highpass butterworth filter on the EEG signal from each channel
        logger.debug('Running first-order .1 Hz highpass filter on all channels.')
        for i in range(self._data.shape[0]):
            self._data[i] = butter_filt(self._data[i], .1, self.header['sample_rate'], 'highpass', 1)
        logger.debug('Done')

        # Divide the signal by the amplifier gain
        self._data = self._data / self.amp_gain

        # Clip to within bounds of selected data format
        bounds = np.iinfo(self.DATA_FORMAT)
        self._data = self._data.clip(bounds.min, bounds.max)
        for i in range(self.header['num_events']):
            self._data[-1*i] = self._data[-1*i] / np.max(self._data[-1*i])
        self._data = np.around(self._data).astype(self.DATA_FORMAT)

    def _split_data(self, location, basename):
        """
        Splits the data extracted from the raw file into each channel and event, and writes the data for each channel
        into a separate file. Also writes two parameter files containing the sample rate, data format, and amp gain
        for the session.
        :param location: A string denoting the directory in which the channel files are to be written
        :param basename: The string used to name the channel files (typically subj_DDMonYY_HHMM)
        """
        self.basename = basename
        self.noreref_loc = location
        self.get_data()

        # Create directory if needed
        if not os.path.exists(location):
            files.makedirs(location)

        # Write EEG channel files
        for i in range(self.header['num_channels']):
            j = i+1
            filename = os.path.join(location, basename + ('.%03d' % j))
            logger.debug(str(i+1))
            sys.stdout.flush()
            # Each row of self._data contains all samples for one channel or event
            self._data[i].tofile(filename)

        # Write event channel files
        current_event = 0
        for i in range(self.header['num_channels'], self.header['num_channels'] + self.header['num_events']):
            filename = (basename, '.', self.header['event_codes'][current_event])
            filename = os.path.join(location, ''.join(filename))
            logger.debug(self.header['event_codes'][current_event])
            sys.stdout.flush()
            # Each row of self._data contains all samples for one channel or event, so write each row to its own file
            self._data[i].tofile(filename)
            current_event += 1

        # Write the sample rate, data format, and amplifier gain to two params.txt files in the noreref folder
        logger.debug('Writing param files.')
        paramfile = os.path.join(location, 'params.txt')
        params = 'samplerate ' + str(self.header['sample_rate']) + '\ndataformat ' + self.data_format + '\ngain ' + \
                 str(self.amp_gain) + '\nsystem EGI'
        with files.open_with_perms(paramfile, 'w') as f:
            f.write(params)
        paramfile = os.path.join(location, basename + '.params.txt')
        with files.open_with_perms(paramfile, 'w') as f:
            f.write(params)
        logger.debug('Done.')

    def reref(self, bad_chans, location):
        """
        Rereferences the EEG recordings and writes the referenced data to separate files for each channel. Rereferencing
        is performed by dividing each sample by the average voltage of that sample number across all good channels
        (excluding event channels).

        :param bad_chans: 1-D numpy array containing all channel numbers to be excluded from rereferencing
        :param location: A string denoting the directory to which the reref files will be written
        """
        # Create directory if needed
        if not os.path.exists(location):
            files.makedirs(location)

        logger.debug('Rerefencing data...')

        # Ignore bad channels for the purposes of calculating the averages for rereference
        all_chans = np.array(range(1, self.header['num_channels']+1))
        good_chans = np.setdiff1d(all_chans, np.array(bad_chans))

        # Find the average value of each sample across all good channels (index of each channel is channel number - 1)
        means = np.mean(self._data[good_chans-1], axis=0)

        # Rereference the data
        self._data = self._data - means

        # Clip to within bounds of selected data format
        bounds = np.iinfo(self.DATA_FORMAT)
        self._data = np.around(self._data.clip(bounds.min, bounds.max)).astype(self.DATA_FORMAT)

        logger.debug('Done.')

        # Write reref files
        logger.debug('Writing rereferenced channels...')
        for chan in all_chans:
            filename = os.path.join(location, self.basename + ('.%03d' % chan))
            # Write the rereferenced data from each channel to its own file
            self._data[chan-1].tofile(filename)
        logger.debug('Done.')

        # Copy the params.txt file from the noreref folder
        logger.debug('Copying param file...')
        copy(os.path.join(self.noreref_loc, 'params.txt'), location)
        logger.debug('Done.')

        # Write a bad_chans.txt file in the reref folder
        np.savetxt(os.path.join(location, 'bad_chans.txt'), bad_chans, fmt='%s')

    @staticmethod
    def find_bad_chans(log, threshold):
        """
        Reads an artifact log to determine whether there were any bad channels during a session. A channel is considered
        to be bad if it contained an artifact lasting longer than {threshold} milliseconds. Bad channels are excluded
        from the rereferencing process.
        :param log: The filepath to the artifact log
        :param threshold: The minimum number of milliseconds that an artifact must last in order for a channel to be
        declared "bad"
        :return: A 1D numpy array containing the numbers of all bad channels
        """
        # Read the log
        with open(log, 'r') as f:
            text = f.read()

        # Use regex and numpy to find all rows in the file that contain a channel number, and extract the numbers
        chans = np.array([re.findall(r'[0-9]+', s)[0] for s in re.findall(r'Channel number: [0-9]+', text)])
        # Find all rows in the file that contain an artifact duration, and extract the durations
        durs = np.array([int(re.findall(r'[0-9]+', s)[0]) for s in re.findall(r'Artifact\'s duration: [0-9]+', text)])

        # Get the channel numbers for any artifacts that lasted longer than the threshold
        bad_chans = np.unique(chans[np.where(durs > threshold)])

        return bad_chans

    def get_start_time(self):
        # Read header info if have not already done so, as the header contains the start time info
        if not self.header:
            self.read_header()
        return self.start_datetime

    def get_start_time_string(self):
        return self.get_start_time().strftime(self.STRFTIME)

    def get_start_time_ms(self):
        return int((self.get_start_time() - self.EPOCH).total_seconds() * 1000)

    def get_sample_rate(self):
        if not self.header:
            self.read_header()
        return self.header['sample_rate']

    def get_source_file(self):
        return self.raw_filename

    def get_n_samples(self):
        if not self.header:
            self.read_header()
        return self.header['num_samples']


class BDF_reader(EEG_reader):
    """
    Parses the EEG sample data from a Biosemi .bdf file. The file begins with a header with a series of information
    written in ASCII such as the version number, start time, and channel count of the recording. This is followed by
    24-bit binary for each of the EEG data samples. Full information on the format of .bdf files can be found
    on Biosemi's website at http://www.biosemi.com/faq/file_format.htm

    DATA FIELDS:
    raw_filename: The path to the .raw file containing the EEG data for the session.
    basename: The string used to name the channel files once split (typically subj_DDMonYY_HHMM).
    noreref_loc: The path to the noreref directory.
    start_datetime: The start time and date of the recording.
    data: Holds the unpacked and split EEG data.
    header: A dictionary where the header info from the beginning of the raw file is stored.
    header_names: A mapping of header names to the format string required for unpacking that header item.
    gain: The gain factor
    data_format: A string stating the format in which the output split channel files will be written.
    sample_rate: The sample rate of the recording
    """
    def __init__(self, raw_filename, unused_jacksheet):
        self.raw_filename = raw_filename
        self.basename = ''
        self.noreref_loc = ''
        self.start_datetime = None
        self._data = None
        self.sync = None
        self.header = {}
        self.header_names = (('ID2', 7, False), ('subject', 80, False), ('recording', 80, False), ('date', 8, False),
                             ('time', 8, False), ('num_header_bytes', 8, False), ('sample_format', 44, False),
                             ('num_records', 8, False), ('record_dur', 8, False), ('num_channels', 4, False),
                             ('channel_names', 16, True), ('transducer_type', 80, True), ('physical_dims', 8, True),
                             ('physical_min', 8, True), ('physical_max', 8, True), ('digital_min', 8, True),
                             ('digital_max', 8, True), ('prefiltering', 80, True), ('samps_per_record', 8, True),
                             ('reserved', 32, True))
        self.gain = None
        self.data_format = 'int16'
        self.sample_rate = None
        self.total_samples = None

    def read_header(self):
        """
        Loads the header information from the file and uses it to calculate the sample rate, gain, and start date/time.
        All headers except the first are written in ASCII, and do not need to be unpacked. The length of each header is
        specified in self.header_names, including a boolean which indicates whether or not the header contains an entry
        for each channel. The headers fields are as follows:

        ID: Always a one-byte '255' followed by seven ASCII characters spelling 'BIOSEMI'
        Subject ID: The subject's number
        Recording ID: The session/recording number
        Date: The start date of the recording, formatted as 'DD.MM.YY'
        Time: The start time of the recording, formatted as 'HH.MM.SS'
        Number of header bytes: The length of the header, in bytes. This is also the index of the first data sample.
        Sample format: Always '24BIT' written in ASCII
        Number of records: The number of records in the recording
        Record duration: The length of each record, in seconds (typically 1)
        Number of channels: The number of channels recorded from
        Channel names: The name of each channel
        Transducer types: The type of recording device used for each channel (e.g. 'active electrode')
        Physical dimensions: The units of eah channel's signal (e.g. uV)
        Physical min: The minimum possible value of the physical recording
        Physical max: The maximum possible value of the physical recording
        Digital min: The minimum possible digital value of a sample
        Digital max: The maximum possible digital value of a sample
        Prefiltering: Information about any prefiltering performed on each channel of the signal
        Samples per record: The number of samples per record for each channel; sample rate = samps_per_rec / record_dur
        Reserved: Unknown, but does not appear to be used
        """
        with bz2.BZ2File(self.raw_filename, 'rb') as raw_file:
            # Read header info; each triplet from self.header_names contains the name of the header, its length, and
            # whether it has a separate entry for each channel
            self.header['ID1'] = struct.unpack('B', raw_file.read(1))[0]
            for triplet in self.header_names:
                # Identify how many ASCII characters need to be read
                chars_to_read = triplet[1]
                if triplet[2]:
                    chars_to_read = chars_to_read * int(self.header['num_channels'])

                # Read the appropriate number of ASCII characters for each field of the header
                # Split any header fields that contain separate entries for each channel
                self.header[triplet[0]] = raw_file.read(chars_to_read)
                self.header[triplet[0]] = self.header[triplet[0]].strip() if not triplet[2] else \
                    np.array([self.header[triplet[0]][i:i+triplet[1]].strip() for i in range(0, chars_to_read, triplet[1])])

        # Reformat headers that require it
        for head in ('date', 'time'):
            self.header[head] = [int(x) for x in self.header[head].split('.')]
        self.header['date'][2] += 2000  # convert year from two digits to four... will need to be changed in 2100
        for head in ('num_header_bytes', 'num_records', 'record_dur', 'num_channels'):
            self.header[head] = int(self.header[head])
        for head in ('physical_min', 'physical_max', 'digital_min', 'digital_max', 'samps_per_record'):
            self.header[head] = self.header[head].astype(int)

        # Set the start date and time based on the header info
        self.start_datetime = datetime.datetime(self.header['date'][2], self.header['date'][1],
                                                self.header['date'][0], self.header['time'][0],
                                                self.header['time'][1], self.header['time'][2])

        # If different channels have different sample rates, the pipeline currently will not be able to support it
        if np.max(self.header['samps_per_record']) != np.min(self.header['samps_per_record']):
            raise('Some channels have different sampling rates from one another. Pipeline does not currently support '
                  'EEG data with multiple sample rates.')

        # Calculate the sample rate and gain for each channel based on the relevant headers
        self.sample_rate = self.header['samps_per_record'][0] / self.header['record_dur']
        self.gain = (self.header['physical_max'] - self.header['physical_min']).astype(float) / \
                    (self.header['digital_max'] - self.header['digital_min']).astype(float)

        # Log various information about the file
        logger.debug('EEG File Information:')
        logger.debug('---------------------')
        logger.debug('Sample Rate = %d' % self.sample_rate)
        logger.debug('Start of recording = %d/%d/%d %02d:%02d' % (self.header['date'][1], self.header['date'][0],
                                                           self.header['date'][2], self.header['time'][0],
                                                           self.header['time'][1]))
        logger.debug('Number of Channels and Events = %d' % self.header['num_channels'])

    def get_data(self):
        """
        Unpack the binary in the raw data file to get the voltage data from all channels.

        BDF files store data samples as signed 24-bit (3-byte) little-endian binary. Note that Python's struct.unpack()
        function cannot interpret 3 bytes as a signed integer (requires 4 bytes), so each number needs to be padded with
        an extra byte. Because the data is signed, padding can be performed by adding an extra byte of \x00 at the least
        significant end and right-shifting the result by 8 bits. Unfortunately, this is likely to slow down the data
        reading process.
        """
        # Read header info if have not already done so
        if not self.header:
            self.read_header()

        # Calculate total number of samples to read by summing the number of samples in all channels
        self.total_samples = self.header['num_records'] * np.sum(self.header['samps_per_record'])

        # Data will be stored in a matrix with one row for each channel and one column for each sample
        self._data = np.zeros((self.header['num_channels'], self.header['num_records'] *
                               self.header['samps_per_record'][0]))

        raw = np.zeros((self.total_samples, 1))
        logger.debug('Loading %d samples...' % self.total_samples)
        with bz2.BZ2File(self.raw_filename, 'rb') as raw_file:
            # Go to index for the beginning of the EEG samples in the raw file
            data_start_index = self.header['num_header_bytes']
            raw_file.seek(data_start_index)

            # Create the ranges that j and h will iterate over outside of the nested loop, so we don't end up
            # creating range(self.header['samps_per_record']) millions of times
            chan_range = range(self.header['num_channels']-1)  # List all non-sync channels
            samp_range = range(self.header['samps_per_record'][0])

            # Read samples into self._data
            for i in range(self.header['num_records']):
                c = (i-1) * self.header['samps_per_record'][0]
                for j in chan_range:
                    for h in samp_range:
                        # Unpack cannot read int24, so each sample must be read and padded independently
                        self._data[j, c + h] = struct.unpack('<i', '\x00' + raw_file.read(3))[0] >> 8
                # Read the sync pulse channel separately, as it is big-endian
                for h in samp_range:
                    self._data[-1, c + h] = struct.unpack('>i', raw_file.read(3) + '\x00')[0] >> 8

        logger.debug('Done.')

        # Run a first-order highpass butterworth filter on each channel
        logger.debug('Running first-order .1 Hz highpass filter on all channels.')
        for i in range(self._data.shape[0]):
            self._data[i] = butter_filt(self._data[i], .1, self.sample_rate, 'highpass', 1)

        logger.debug('Done')

    def _split_data(self, location, basename):
        """
        Splits the data extracted from the binary file into each channel, and writes the data for each into a separate
        file. Also writes two parameter files containing the sample rate, data format, and amp gain for the session.

        :param location: A string denoting the directory in which the channel files are to be written
        :param basename: The string used to name the channel files (typically subj_DDMonYY_HHMM)
        """
        self.basename = basename
        self.noreref_loc = location
        self.get_data()

        # Create directory if needed
        if not os.path.exists(location):
            files.makedirs(location)

        # Clip to within bounds of selected data format
        bounds = np.iinfo(self.DATA_FORMAT)
        self._data = self._data.clip(bounds.min, bounds.max)

        self.sync = self._data[-1] / np.max(self._data[-1])  # Clean up and separate sync pulse channel
        self._data = self._data[:-5]  # Drop EXG5 through EXG 8 and the sync channel from self._data
        self.header['channel_names'] = self.header['channel_names'][:-5]

        self.sync = np.around(self.sync).astype(self.DATA_FORMAT)
        self._data = np.around(self._data).astype(self.DATA_FORMAT)

        # Write EEG channel files
        for i in range(self._data.shape[0]):
            filename = os.path.join(location, basename + ('.' + self.header['channel_names'][i]))
            logger.debug(i+1)
            sys.stdout.flush()
            # Each row of self._data contains all samples for one channel or event
            self._data[i].tofile(filename)

        # Write sync pulse channel file
        filename = os.path.join(location, basename + '.Status')
        logger.debug(i + 1)
        sys.stdout.flush()
        # Each row of self._data contains all samples for one channel or event
        self.sync.tofile(filename)

        logger.debug('Saved.')

        # Write the sample rate, data format, and amplifier gain to two params.txt files in the noreref folder
        logger.debug('Writing param files.')
        paramfile = os.path.join(location, 'params.txt')
        params = 'samplerate ' + str(self.sample_rate) + '\ndataformat ' + self.DATA_FORMAT + '\ngain ' + \
                 str(self.gain[0]) + '\nsystem Biosemi'
        with files.open_with_perms(paramfile, 'w') as f:
            f.write(params)
        paramfile = os.path.join(location, basename + '.params.txt')
        with files.open_with_perms(paramfile, 'w') as f:
            f.write(params)
        logger.debug('Done.')

    def reref(self, bad_chans, location):
        """
        Rereferences the EEG recordings and writes the referenced data to separate files for each channel. Rereferencing
        is performed by dividing each sample by the average voltage of that sample number across all good channels
        (excluding event channels).

        :param bad_chans: A list containing all channel numbers to be excluded from rereferencing
        :param location: A string denoting the directory to which the reref files will be written
        """
        # Create directory if needed
        if not os.path.exists(location):
            files.makedirs(location)

        logger.debug('Rerefencing data...')

        # Ignore bad channels and the sync channel for the purposes of calculating the averages for rereference
        good_chans = np.where(np.array([(chan not in bad_chans) for chan in self.header['channel_names']]))[0]

        # Find the average value of each sample across all good channels (index of each channel is channel number - 1)
        means = np.mean(self._data[good_chans], axis=0)

        # Rereference the data
        self._data = self._data - means

        # Clip to within bounds of selected data format
        bounds = np.iinfo(self.DATA_FORMAT)
        self._data = np.around(self._data.clip(bounds.min, bounds.max)).astype(self.DATA_FORMAT)

        logger.debug('Done.')

        # Write reref files
        logger.debug('Writing rereferenced channels...')
        for i in range(self._data.shape[0]):
            filename = os.path.join(location, self.basename + '.' + self.header['channel_names'][i])
            # Write the rereferenced data from each channel to its own file
            self._data[i].tofile(filename)
        logger.debug('Done.')

        # Copy the params.txt file from the noreref folder
        logger.debug('Copying param file...')
        copy(os.path.join(self.noreref_loc, 'params.txt'), location)
        logger.debug('Done.')

    def get_start_time(self):
        # Read header info if have not already done so, as the header contains the start time info
        if not self.header:
            self.read_header()
        return self.start_datetime

    def get_start_time_string(self):
        return self.get_start_time().strftime(self.STRFTIME)

    def get_start_time_ms(self):
        return int((self.get_start_time() - self.EPOCH).total_seconds() * 1000)

    def get_sample_rate(self):
        if not self.header:
            self.read_header()
        return self.sample_rate

    def get_source_file(self):
        return self.raw_filename

    def get_n_samples(self):
        if not self.header:
            self.read_header()
        return self.total_samples


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
    '.raw': EGI_reader,
    '.bdf': BDF_reader
}


def get_eeg_reader(raw_filename, jacksheet_filename=None, **kwargs):
    if isinstance(raw_filename, list):
        return Multi_NSx_reader(raw_filename, jacksheet_filename)
    else:
        file_type = os.path.splitext(raw_filename)[1].lower()
        # If the data file is compressed, get the extension before the .bz2
        file_type = os.path.splitext(os.path.splitext(raw_filename)[0])[1].lower() if file_type == '.bz2' else file_type
        return READERS[file_type](raw_filename, jacksheet_filename, **kwargs)
