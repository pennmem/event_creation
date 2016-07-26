import pyedflib
import os
from sys import platform as _platform
import subprocess
import glob
import numpy as np
import json
import re
from nsx_utility.brpylib import NsxFile
import struct
import datetime


class UnSplittableEEGFileException(Exception):
    pass

class EEG_reader:

    DATA_FORMAT = 'int16'

    STRFTIME = '%d%b%y_%H%M'
    MAX_CHANNELS = 256

class NK_reader(EEG_reader):

    def __init__(self, nk_filename, jacksheet_filename=None):
        self.nk_filename = nk_filename
        if jacksheet_filename:
            self.jacksheet = {v['label']:k for k,v in read_jacksheet(jacksheet_filename).items()}
        else:
            self.jacksheet = None
        self.sample_rate = None
        self.start_datetime = None

    def set_jacksheet(self, jacksheet_filename):
        self.jacksheet = {v['label']:k for k,v in read_jacksheet(jacksheet_filename).items()}

    def get_start_time_string(self):
        with open(self.nk_filename, 'rb') as f:
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
            return dt.strftime(self.STRFTIME)

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
        return cls.fread(f, 'c', n, 1)

    @property
    def labels(self):
        return self.get_labels(elec_file = os.path.splitext(self.nk_filename)[0] + '.21E')

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

    def get_data(self, jacksheet_dict):
        eeg_file = self.nk_filename
        elec_file = os.path.splitext(eeg_file)[0] + '.21E'

        with open(eeg_file, 'rb') as f:
            # Skipping device block
            deviceBlockLen = 128
            f.seek(deviceBlockLen)

            # Reading EEG1 control block
            _ = f.read(1) # block ID
            device_type = self.char_(f, 16) # Device type
            new_format = device_type[:9] == 'EEG-1200A'
            number_of_blocks = self.uint8(f)
            if number_of_blocks > 1:
                raise UnSplittableEEGFileException('%s EEG2 Control blocks detected' % number_of_blocks)
            block_address = self.int32(f)
            _ = f.read(16) # EEG control block name

            # Reading EEG2 Control block
            f.seek(block_address, 0)
            _ = f.read(1) # Block ID
            _ = f.read(16) # Data format
            number_of_blocks = self.uint8(f)
            if number_of_blocks > 1:
                raise UnSplittableEEGFileException('%d waveform blocks detected' % number_of_blocks)
            block_address = self.int32(f)
            _ = f.read(16) # Name of waveform block

            # Reading waveform block
            f.seek(block_address, 0)
            _ = f.read(1) # Block ID
            _ = f.read(16) # Data format
            _ = f.read(1) # Data type
            L = self.uint8(f) # Byte length of one data
            M = self.uint8(f) # Mark/event flag

            T_year = self.uint8(f)
            T_month = self.uint8(f)
            T_day = self.uint8(f)
            T_hour = self.uint8(f)
            T_minute = self.uint8(f)
            T_second = self.uint8(f)

            print 'Date of session: %d/%d/%d\n' % (T_month, T_day, T_year)
            print 'Time of start: %02d:%02d:%02d\n' % (T_hour, T_minute, T_second)

            sample_rate = self.uint16(f)
            print sample_rate
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
            print 'Length of session: %2.2f hours\n' % (num_100_ms_blocks/10./3600.)
            num_samples = actual_sample_rate * num_100_ms_blocks / 10.
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
                print 'NEW FORMAT...'

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
                _ = self.char_(f, 20)  #  Start time string
                actual_sample_rate = self.uint32(f)  #  Data interval (sample rate)
                num_100_ms_blocks = self.uint64(f)  #  Length of session

                num_samples = actual_sample_rate * num_100_ms_blocks / 10
                ad_off = self.int16(f)  #  AD offset at 0V
                ad_val = self.uint16(f)  #  ACD val for 1 division
                bit_len = self.uint16(f)  #  Bit length of one sample
                com_flag = self.uint16(f)  #  Data compression
                reserve_l = self.uint16(f)  #  Reserve length
                _ = self.char_(f, reserve_l)  #  Reserve data

                num_channels = self.uint32(f) # Number of RAW recordings


            lines = [line.strip() for line in open(elec_file).readlines()]
            end_range = lines.index('[SD_DEF]')
            split_lines = [line.split('=') for line in lines[:end_range] if '=' in line ]
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

            channel_mask = np.array([name in jacksheet_dict.keys() for name in names_21e_ordered])
            nums_21e_filtered = nums_21e_ordered[channel_mask]
            names_21e_filtered = names_21e_ordered[channel_mask]
            jacksheet_filtered = [jacksheet_dict[name] for name in names_21e_filtered]

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
            print 'Reading...'
            data = np.array(self.uint16(f, int((num_channels + 1) * num_samples))).reshape((int(num_samples), int(num_channels + 1))).T
            print 'Done.'

            # Filter only for relevant channels
            data_num_mask = np.array(data_num_to_21e_index != -1)
            data = data[np.append(data_num_mask,False), :]
            data_num_to_21e_index = data_num_to_21e_index[data_num_mask]

            # Remove offset
            data = data + ad_off
            data_dict = {jacksheet_filtered[int(data_num_to_21e_index[i])]: data[i,:] for i in range(data.shape[0])}
            return data_dict

    def split_data(self, location, basename):
        if not self.jacksheet:
            raise UnSplittableEEGFileException('Jacksheet not specified')
        data = self.get_data(self.jacksheet)
        if not self.sample_rate:
            raise UnSplittableEEGFileException('Sample rate not determined')

        print 'saving:',
        for channel, channel_data in data.items():
            filename = os.path.join(location, basename + ('.%03d' % channel))

            print channel,
            channel_data.astype(self.DATA_FORMAT).tofile(filename)
        print 'Saved.'

        with open(os.path.join(location, basename + '.params.txt'),'w') as outfile:
            outfile.write('samplerate %.2f\ndataformat \'%s\'\ngain %e' % (self.sample_rate, self.DATA_FORMAT, self.gain))

class NSx_reader(EEG_reader):
    TIC_RATE = 30000


    SAMPLE_RATES = {
        '.ns2': 1000
    }


    def __init__(self, nsx_filename, jacksheet_filename=None):
        if jacksheet_filename:
            if os.path.splitext(jacksheet_filename)[1] == '.txt':
                self.jacksheet = read_jacksheet(jacksheet_filename)
            else:
                raise NotImplementedError('Non-txt jacksheet not implemented')
        else:
            self.jacksheet = None
        try:
            self.nsx_info = self.get_nsx_info(nsx_filename)
        except IOError:
            raise IOError("Could not read %s" % nsx_filename)

        self._data = None

    @property
    def data(self):
        if not self._data:
            self._data = self.nsx_info['reader'].getdata()
        return self._data

    @classmethod
    def get_nsx_info(cls, nsx_file):
        reader = NsxFile(nsx_file)
        _, extension = os.path.splitext(nsx_file)
        data = reader.getdata()
        start_time = reader.basic_header['TimeOrigin']
        total_data_points = sum([header['NumDataPoints'] for header in data['data_headers']])
        length_ms = total_data_points / float(NSx_reader.SAMPLE_RATES[extension]) * 1000.
        return {'filename': nsx_file, 'start_time': start_time, 'length_ms': length_ms, 'reader': reader}

    def get_start_time_string(self):
        return self.nsx_info['start_time'].strftime(self.STRFTIME)

    def get_sample_rate(self, channels=None):
        _, extension = os.path.splitext(self.nsx_info['filename'])
        return NSx_reader.SAMPLE_RATES[extension]

    def split_data(self, location, basename):
        sample_rate = self.get_sample_rate()
        channels = np.array(self.data['elec_ids'])
        for label, channel in self.jacksheet.items():
            filename = os.path.join(location, basename + '.%03d' % channel)
            data = self.data['data'][channels==channel, :].astype(self.DATA_FORMAT)
            data.tofile(filename)

        with open(os.path.join(location, basename + '.params.txt'), 'w') as outfile:
            outfile.write('samplerate %.2f\ndataformat \'%s\'\ngain %d' % (sample_rate, self.DATA_FORMAT, 1))

class EDF_reader(EEG_reader):

    def __init__(self, edf_filename, jacksheet_filename=None):
        if jacksheet_filename:
            self.jacksheet = {v['label']:k for k,v in read_jacksheet(jacksheet_filename).items()}
        else:
            self.jacksheet = None
        try:
            self.reader = pyedflib.EdfReader(edf_filename)
        except IOError:
            raise IOError("Could not read %s" % edf_filename)
        self.headers = self.get_channel_info()

    def set_jacksheet(self, jacksheet_filename):
        self.jacksheet = {v['label']:k for k,v in read_jacksheet(jacksheet_filename).items()}

    def get_channel_info(self):
        headers = {}
        for i in range(self.MAX_CHANNELS):
            header = self.reader.getSignalHeader(i)
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

    def get_start_time_string(self):
        return self.reader.getStartdatetime().strftime(self.STRFTIME)

    def split_data(self, location, basename):

        sample_rate = self.get_sample_rate()

        print 'Saving:',
        for channel, header in self.headers.items():
            if self.jacksheet:
                if header['label'] not in self.jacksheet:
                    continue
                out_channel = self.jacksheet[header['label']]
            else:
                out_channel = channel
            print channel ,'->', out_channel
            filename = os.path.join(location, basename + '.%03d' % (out_channel))

            print header['label'],
            data = self.reader.readSignal(channel).astype(self.DATA_FORMAT)
            data.tofile(filename)

        print 'Saved.'
        with open(os.path.join(location, basename + '.params.txt'), 'w') as outfile:
            outfile.write('samplerate %.2f\ndataformat \'%s\'\ngain %e' % (sample_rate, self.DATA_FORMAT, 1))

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
    return {int(line[0]): {'label': line[1]} for line in lines}

def read_json_jacksheet(filename):
    json_load = json.load(open(filename))
    jacksheet = {int(k):v for k,v in json_load.items()}
    return jacksheet

def convert_nk_to_edf(filename):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    if _platform == 'darwin': # OS X
        program = os.path.join(current_dir, 'nk2edf_mac')
    else:
        program = os.path.join(current_dir, 'nk2edf_linux')
    print 'Converting to edf'
    try:
        subprocess.check_call([program, filename])
    except subprocess.CalledProcessError:
        print 'Could not process annotat'
        subprocess.call([program, '-no-annotations', filename])
    edf_dir = os.path.dirname(filename)
    edf_file = glob.glob(os.path.join(edf_dir, '*.edf'))
    import stat
    try:
        os.chmod(edf_file, stat.S_IWGRP)
    except:
        print 'Could not chmod %s' % edf_file
    if not edf_file:
        raise UnSplittableEEGFileException("Could not convert %s to edf" % filename)
    else:
        print 'Success!'
        return edf_file[0]

def create_eeg_basename(info, subject, experiment, session):
    return '%s_%s_%d_%s' % (subject, experiment, session, info['start_time'].strftime(EEG_reader.STRFTIME))