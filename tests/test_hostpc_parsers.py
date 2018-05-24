from submission.parsers.hostpc_parsers import FRHostPCLogParser,PS5FRLogParser
import numpy as np
import glob
import pandas as pd

if __name__ == '__main__':
    fr_files = {'event_log':'/Users/leond/Documents/unityepl_logs/R1350D/behavioral/catFR6/session_6/host_pc/20171102_150753/event_log.json',
                'experiment_config':'/Users/leond/Documents/unityepl_logs/R1350D/behavioral/catFR6/session_6/host_pc/20171102_150753/experiment_config.json',
                'electrode_config': ['/Users/leond/Documents/unityepl_logs/R1350D/behavioral/catFR6/session_6/host_pc/20171102_150753/config_files/R1350D_18OCT2017L0M0STIM.csv'],
                'wordpool':'/Users/leond/Documents/unityepl_logs/R1350D/behavioral/RAM_wordpool.txt',
                'annotations':list(glob.glob('/Users/leond/Documents/unityepl_logs/R1350D/behavioral/catFR6/session_6/*.ann'))
                }

    ps5_files = {'event_log':'/Users/leond/Desktop/R1378T/behavioral/PS5_FR/session_1/host_pc/20180127_145319/event_log.json',
                 'experiment_config':'/Users/leond/Desktop/R1378T/behavioral/PS5_FR/session_1/host_pc/20180127_145319/experiment_config.json',
                 'electrode_config':['/Users/leond/Desktop/R1378T/behavioral/PS5_FR/session_1/host_pc/20180127_145319/config_files/R1378T_18DEC2017L0M0STIM.csv'],
                 'wordpool':'/Users/leond/Desktop/R1378T/behavioral/PS5_FR/short_ram_wordpool_en.txt',
                 'annotations':list(glob.glob('/Users/leond/Desktop/R1378T/behavioral/PS5_FR/session_1/*.ann'))}
    parser = PS5FRLogParser('r1','none','0','test',0,ps5_files)
    ps5_events = parser.parse()

    parser = FRHostPCLogParser('r1','none',0,'test',0,fr_files)

    events = parser.parse()
    pass