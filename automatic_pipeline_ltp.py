#!/usr/global/shared/runvenv workshop

from __future__ import print_function
import json
import sys
import os
import random
from event_creation.submission.tasks import IndexAggregatorTask

from .log import logger

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)

##########
#
# Event creation management
#
##########
def automatic_event_creator(check_index=True):

    # Load list of supported active experiments
    with open('/data/eeg/scalp/ltp/ACTIVE_EXPERIMENTS.txt', 'r') as f:
        experiments = [s.strip() for s in f.readlines()]

    inputs = []
    for exp in experiments:
        exp = exp.strip()
        if not exp:
            continue
        try:
            # Get dictionary of new/recently modified sessions
            with open('/data/eeg/scalp/ltp/%s/recently_modified.json' % exp, 'r') as f:
                new_sess = json.load(f)
            # Get index for ltp database to identify which sessions have been proccessed
            with open('/protocols/ltp.json', 'r') as f:
                db_index = json.load(f)
                if db_index != {}:
                    db_index = db_index['protocols']['ltp']['subjects']

                logger.debug(f'Loaded index for LTP database with {len(db_index)} subjects')
                logger.debug(f"db_index keys: {db_index.keys()} db_keys values: {db_index.values()}")
        except IOError:
            print('Unable to load necessary session information for experiment: ', exp)
            print('Skipping...')
            continue
        
        # Create an input structure for each new session that has not been processed
        logger.debug(f'{"Checking index for" if check_index else "Processing"} {exp} sessions: {new_sess}')
        for subject in new_sess:
            for session in new_sess[subject]:
                if check_index:
                    if (db_index == {}) or (subject not in db_index) or (exp not in db_index[subject]['experiments']) or (str(session) not in db_index[subject]['experiments'][exp]['sessions']):
                        inputs.append(f'{exp}:{subject}:{session}')
                        print('Session to run: ', exp, subject, session)
                else:
                    inputs.append(f'{exp}:{subject}:{session}')
                    print('Session to run: ', exp, subject, session)

    # Submit a job to the cluster for each session that needs to be processed
    n_jobs = len(inputs)
    if n_jobs > 0:
        # Minimize consistent problems by shuffling the list each time
        random.shuffle(inputs)

        outdir = os.path.join(os.environ['HOME'], 'logs', 'stdouterr')
        # Use sbatch to launch automatic_run.py,
        # which in turn will call ./submit
        os.system(f'sbatch --mem-per-cpu=60G -t 23:00:00 ' +
            f'-o {outdir}/slurm-%A_%a.out -e {outdir}/slurm-%A_%a.err ' +
            f'-a 0-{n_jobs-1}%16 ' +
            f'{script_dir}/automatic_run.py {script_dir} ' +
            ' '.join(inputs))

if __name__=='__main__':
    try:
        automatic_event_creator(check_index=True)
    except:
        pass
    IndexAggregatorTask().run(protocols='ltp')

