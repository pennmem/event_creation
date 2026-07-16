
import os, socket, getpass
import runpy
import sys

__version__ = "1.2.4"

if sys.version_info[0] < 3:
    input = raw_input


def confirm(prompt):
    while True:
        resp = eval(input(prompt))
        if resp.lower() in ('y', 'n', 'yes', 'no'):
            return resp.lower() == 'y' or resp.lower() == 'yes'
        print('Please enter y or n')


def submit():
    usr = getpass.getuser()
    # scalp/LTP data is owned by maint; RAM/iEEG data is owned by RAM_maint.
    # Run as the account that owns the data you are processing so output files
    # get the correct owner.
    expected_users = ["maint", "RAM_maint", "RAM_clinical"]
    if usr not in expected_users:
        print("This script is meant to be run from maint (scalp/LTP) or "
              "RAM_maint (RAM/iEEG), not", usr)
        confirm("Are you sure you want to continue? ")
    host = socket.gethostname()
    if "node" not in host:
        print("This script is best run from a node not the headnode (use qlogin)")
        confirm("Are you sure you want to continue? ")
    runpy._run_module_as_main("event_creation.submission.convenience")


def split():
    runpy._run_module_as_main("event_creation.submission.readers.eeg_splitter")


