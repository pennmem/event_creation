from ..submission.readers.eeg_reader import EDF_reader, NSx_reader, convert_nk_to_edf, NK_reader, read_text_jacksheet, EGI_reader
from ..submission.exc import EEGError
import numpy as np
import os
import glob

DATA_ROOT='/Volumes/rhino_mount/data/eeg'

subjects = (('R1001P', 'FR1_0/R1001P_2014-10-12A.edf', 'R1001P_12Oct14_1034'))

def xtest_edf_reader():

    for subject in subjects:
        filename = os.path.join(DATA_ROOT, subject[0], 'raw', subject[1])
        jacksheet = os.path.join(DATA_ROOT, subject[0], 'docs', 'jacksheet.txt')
        split_out = '../tests/test_output/%s' % subject[0]
        try:
            os.mkdir(split_out)
        except:
            pass


        reader = EDF_reader(filename, jacksheet )
        reader.split_data(split_out, 'test')

        old_files = glob.glob(os.path.join(DATA_ROOT, subject[0], 'eeg.noreref', subject[2]+'.*'))
        old_params = os.path.join(DATA_ROOT, subject[0], 'eeg.noreref', subject[2]+'.params.txt')
        old_gain = float(open(old_params,'r').readlines()[-1].split()[-1])

        for old_file in old_files:
            try:
                num = int(old_file.split('.')[-1])
            except:
                continue
            test_file = '../tests/test_output/%s/test.%03d' % (subject[0], num)
            print(('comparing %d' % num))

            old_data = np.fromfile(old_file, 'int16') * old_gain

            new_data = np.fromfile(test_file, 'int16')

            assert (abs(old_data-new_data) < 2).all(), 'old data does not match new data'




def xtest_nk_converter():
   for subject in nk_subjects:
        nk_name = os.path.join(DATA_ROOT, subject[0], 'raw', subject[1])
        edf_name = convert_nk_to_edf(nk_name)
        jacksheet = os.path.join(DATA_ROOT, subject[0], 'docs', 'jacksheet.txt')
        split_out = '../tests/test_output/%s' % subject[0]
        try:
            os.mkdir(split_out)
        except:
            pass


        reader = EDF_reader(edf_name, jacksheet )
        reader.split_data(split_out, 'test')

        leads = [int(x.strip()) for x in open(os.path.join(DATA_ROOT, subject[0], 'tal', 'leads.txt'))]

        old_files = os.path.join(DATA_ROOT, subject[0], 'eeg.noreref', subject[2]+'.%03d')
        old_params = os.path.join(DATA_ROOT, subject[0], 'eeg.noreref', subject[2]+'.params.txt')
        old_gain = float(open(old_params,'r').readlines()[-1].split()[-1])

        for lead in leads:
            old_file = old_files % lead
            test_file = '../tests/test_output/%s/test.%03d' % (subject[0], lead)
            print(('comparing %d' % lead))

            old_data = np.fromfile(old_file, 'int16') * old_gain

            new_data = np.fromfile(test_file, 'int16')

            if not (abs(old_data-new_data) < 2).all():
                raise EEGError('old data does not match new data : max diff %.1f' %
                               max(abs(old_data-new_data)))

ns2_subjects = (('R1170J_1', 'FR3_0/20160516-144520/20160516-144520-001.ns2', 'R1170J_1_FR3_0_16May16_1845'),
                ('R1196N', 'PAL1_2/20160712-102704/20160712-102704-001.ns2', 'R1196N_PAL1_2_12Jul16_1427'),)

def xtest_nsx_split():
    for subject in ns2_subjects:
        ns2_name = os.path.join(DATA_ROOT, subject[0], 'raw', subject[1])
        jacksheet = os.path.join(DATA_ROOT, subject[0], 'docs', 'jacksheet.txt')
        split_out = '../tests/test_output/%s/' % subject[0]
        try:
            os.mkdir(split_out)
        except:
            pass

        reader = NSx_reader(ns2_name, jacksheet)
        reader.split_data(split_out, 'test')

        leads = [int(x.strip()) for x in open(os.path.join(DATA_ROOT, subject[0], 'tal', 'leads.txt'))]

        old_files = os.path.join(DATA_ROOT, subject[0], 'eeg.noreref', subject[2] + '.%03d')
        old_params = os.path.join(DATA_ROOT, subject[0], 'eeg.noreref', subject[2] + '.params.txt')
        old_gain = float(open(old_params, 'r').readlines()[-1].split()[-1])

        for lead in leads:
            old_file = old_files % lead
            test_file = '../tests/test_output/%s/test.%03d' % (subject[0], lead)
            print(('comparing %d' % lead))

            old_data = np.fromfile(old_file, 'int16') * old_gain

            new_data = np.fromfile(test_file, 'int16')

            if not (abs(old_data - new_data) < 2).all():
                raise EEGError('old data does not match new data : max diff %.1f' %
                               max(abs(old_data - new_data)))

nk_subjects = (('R1083J', 'FR1_0/RA2350UO.EEG', ''),
               ('R1008J', 'YC1_0/CA2417MP.EEG', 'R1008J_21Nov14_1037'),
               ('R1010J', 'FR1_0/CA2417SF.EEG', 'R1010J_08Dec14_1407'),
               ('R1010J', 'FR1_1/CA2417SY.EEG', 'R1010J_09Dec14_1345'),
               ('R1113T', 'FR1_0/EA4170GY.EEG', 'R1113T_FR1_0_19Nov15_0835'),)
def test_nk_split():
    for subject in nk_subjects:
        filename = os.path.join(DATA_ROOT, subject[0], 'raw', subject[1])
        jacksheet = os.path.join(DATA_ROOT, subject[0], 'docs', 'jacksheet.txt')
        split_out = '../tests/test_output/%s' % subject[0]
        try:
            os.mkdir(split_out)
        except:
            pass

        reader = NK_reader(filename, jacksheet)
        reader.split_data(split_out, 'test')

        old_files = glob.glob(os.path.join(DATA_ROOT, subject[0], 'eeg.noreref', subject[2] + '.*'))
        old_params = os.path.join(DATA_ROOT, subject[0], 'eeg.noreref', subject[2] + '.params.txt')
        old_gain = float(open(old_params, 'r').readlines()[-1].split()[-1])

        new_params = '../tests/test_output/%s/test.params.txt' % subject[0]
        new_gain = float(open(new_params, 'r').readlines()[-1].split()[-1])

        old_files.sort()
        for old_file in old_files:
            try:
                num = int(old_file.split('.')[-1])
            except:
                continue
            test_file = '../tests/test_output/%s/test.%03d' % (subject[0], num)
            print(('comparing %d' % num))

            old_data = np.fromfile(old_file, 'int16') * old_gain

            new_data = np.fromfile(test_file, 'int16') * new_gain

            try:
                if not (abs(old_data - new_data) < 2).all():
                    raise Exception('old data does not match new data')
            except:
                raise Exception('Could not compare new and old data')

def test_egi_split():
    split = True
    reref = True
    path = os.path.expanduser('~/LTP117 20140715 1152.2.raw.bz2')
    reader = EGI_reader(path, None)
    if split:
        reader.split_data(os.path.expanduser('~/noreref'), 'TEST')
    for i in range(1, 10):
        print('Channel', i, 'noreref')
        old_file = os.path.expanduser('~/eeg.noreref/LTP117_15Jul14_1225.00'+str(i))
        new_file = os.path.expanduser('~/noreref/TEST.00'+str(i))
        old_data = np.fromfile(old_file, 'int16')
        new_data = np.fromfile(new_file, 'int16')
        print(old_data[0:10])
        print(new_data[0:10])
    if reref:
        good_chans = np.array(list(range(1, 130)))
        reader.reref(good_chans, os.path.expanduser('~/reref'))
    for i in range(1, 10):
        print('Channel', i, 'reref')
        old_file = os.path.expanduser('~/eeg.reref/LTP117_15Jul14_1225.00' + str(i))
        new_file = os.path.expanduser('~/reref/TEST.00' + str(i))
        old_data = np.fromfile(old_file, 'int16')
        new_data = np.fromfile(new_file, 'int16')
        print(old_data[0:10])
        print(new_data[0:10])

if __name__ == '__main__':
    test_egi_split()
