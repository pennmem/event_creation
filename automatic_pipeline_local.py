#!/usr/bin/env python
"""
Run event creation for recently modified, not-yet-processed scalp (ltp) sessions,
one after another, on a single machine with no job scheduler.

This is the stand-in for automatic_pipeline_ltp.py, which submits an sbatch array
to Slurm. The session-selection logic is the same:

  * experiments come from <data root>/ACTIVE_EXPERIMENTS.txt
  * candidate sessions come from <data root>/<experiment>/recently_modified.json
    (written by scalp_lab_reporting/identify_modified_participants.py)
  * sessions already present in <db root>/protocols/ltp.json are skipped
  * sessions flagged AUTOMATED_ANNOT (uncorrected automatic annotations) are skipped

Each session runs in its own subprocess (python -m event_creation.submission.convenience),
so one failure cannot take the rest down, and each gets its own log in ~/logs/.
Afterwards the ltp index is re-aggregated, as automatic_pipeline_ltp.py does.

Usage:
    automatic_pipeline_local.py            # run everything that needs running
    automatic_pipeline_local.py --dry-run  # only list what would run
    automatic_pipeline_local.py --all      # ignore the index; re-run every recently modified session

Environment overrides (for testing on a copy of the data tree):
    LTP_RHINO_ROOT   root that contains data/eeg and protocols (default '/')
"""
from __future__ import print_function
import datetime
import json
import os
import subprocess
import sys

RHINO_ROOT = os.environ.get('LTP_RHINO_ROOT', '/')
DATA_ROOT = os.path.join(RHINO_ROOT, 'data', 'eeg', 'scalp', 'ltp')
INDEX_FILE = os.path.join(RHINO_ROOT, 'protocols', 'ltp.json')
LOG_DIR = os.path.join(os.environ['HOME'], 'logs')


def now():
    return datetime.datetime.now().strftime('%F %T')


def load_index():
    try:
        with open(INDEX_FILE) as f:
            db = json.load(f)
        return db['protocols']['ltp']['subjects'] if db else {}
    except (IOError, KeyError, ValueError):
        return {}


def already_processed(db, exp, subj, sess):
    try:
        return str(sess) in db[subj]['experiments'][exp]['sessions']
    except KeyError:
        return False


def sessions_to_run(check_index=True):
    with open(os.path.join(DATA_ROOT, 'ACTIVE_EXPERIMENTS.txt')) as f:
        experiments = [s.strip() for s in f if s.strip()]
    db = load_index()
    todo = []
    for exp in experiments:
        rm_file = os.path.join(DATA_ROOT, exp, 'recently_modified.json')
        try:
            with open(rm_file) as f:
                recent = json.load(f)
        except IOError:
            print('{}: no recently_modified.json (run identify_modified_participants.py first), skipping'.format(exp))
            continue
        for subj, sessions in recent.items():
            for sess in sessions:
                sess_dir = os.path.join(DATA_ROOT, exp, subj, 'session_{}'.format(sess))
                if os.path.exists(os.path.join(sess_dir, 'AUTOMATED_ANNOT')):
                    print('skip {} {} {}: uncorrected automatic annotations'.format(exp, subj, sess))
                    continue
                if check_index and already_processed(db, exp, subj, sess):
                    continue
                todo.append((exp, subj, sess))
    return todo


def run_one(exp, subj, sess, script_dir):
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, 'automatic_run_{}_{}_{}.log'.format(exp, subj, sess))
    cmd = [sys.executable, '-m', 'event_creation.submission.convenience', '--set-input',
           'protocol=ltp:code={s}:subject={s}:experiment={e}:session={n}:original_session={n}:montage=0.0'
           .format(s=subj, e=exp, n=sess)]
    if RHINO_ROOT != '/':
        cmd += ['--path', 'rhino_root={r}:data_root={r}/data/eeg:db_root={r}'.format(r=RHINO_ROOT)]
    with open(log, 'a') as fw:
        fw.write('{} - Beginning run of {} {} {}\n'.format(now(), exp, subj, sess))
        fw.flush()
        subprocess.call(cmd, cwd=script_dir, stdout=fw, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        fw.write('{} - Completed run of {} {} {}\n'.format(now(), exp, subj, sess))
    # convenience.py exits 0 either way; its last lines say "Success:" or "Failed:"
    with open(log) as fr:
        tail = fr.read()[-4000:]
    return 'Success:' in tail and 'Failed:' not in tail.split('Success:')[-1], log


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    check_index = '--all' not in sys.argv
    script_dir = os.path.dirname(os.path.realpath(__file__))

    todo = sessions_to_run(check_index)
    print('{} - {} session(s) to run'.format(now(), len(todo)))
    for exp, subj, sess in todo:
        print('  {} {} session {}'.format(exp, subj, sess))
    if dry_run:
        sys.exit(0)

    n_ok = 0
    for exp, subj, sess in todo:
        ok, log = run_one(exp, subj, sess, script_dir)
        n_ok += ok
        print('{} - {} {} {}: {}'.format(now(), exp, subj, sess, 'ok' if ok else 'FAILED, see ' + log))

    if todo:
        sys.path.insert(0, script_dir)
        from event_creation.submission.configuration import paths
        from event_creation.submission.tasks import IndexAggregatorTask
        if RHINO_ROOT != '/':
            paths.set('rhino_root', RHINO_ROOT)
            paths.set('db_root', RHINO_ROOT)
            paths.set('data_root', os.path.join(RHINO_ROOT, 'data', 'eeg'))
        IndexAggregatorTask().run(protocols='ltp')
    print('{} - done: {} ok, {} failed'.format(now(), n_ok, len(todo) - n_ok))
