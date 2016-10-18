import os

RHINO_ROOT = os.path.join(os.environ['HOME'], 'rhino_mount')
DATA_ROOT=os.path.join(RHINO_ROOT, 'data/eeg')
LOC_DB_ROOT=RHINO_ROOT
#DB_ROOT='/Volumes/db_root/' #os.path.join(RHINO_ROOT, 'data', 'eeg')
DB_ROOT=os.path.expanduser('/Volumes/db_root/')
EVENTS_ROOT=os.path.join(RHINO_ROOT, 'data/events')

