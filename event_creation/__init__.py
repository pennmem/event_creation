from __future__ import print_function
import os,socket,getpass
import runpy
import sys

__version__ = "1.2.3"

if sys.version_info[0] < 3:
    input = raw_input


def confirm(prompt):
    while True:
        resp = input(prompt)
        if resp.lower() in ('y', 'n', 'yes', 'no'):
            return resp.lower() == 'y' or resp.lower() == 'yes'
        print('Please enter y or n')


def submit():
    usr = getpass.getuser()
    expected_users =["RAM_maint","RAM_clinical:"]
    if usr not in expected_users:
        print("This script is meant to be run from RAM_maint not", usr)
        confirm("Are you sure you want to continue? ")
    host = socket.gethostname()
    if "node" not in host:
        print("This script is best run from a node not the headnode (use qlogin)")
        confirm("Are you sure you want to continue? ")
    runpy._run_module_as_main("event_creation.submission.convenience")


def split():
    runpy._run_module_as_main("event_creation.submission.readers.eeg_splitter")


