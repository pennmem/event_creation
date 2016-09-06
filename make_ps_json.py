from loggers import log
import re
import os
from submission.transferer import DATA_ROOT
import glob
from ptsa.data.readers import BaseEventReader
import numpy as np
import json
import pprint

def get_subject_sessions_by_experiment(experiment, protocol='r1'):
    if re.match('catFR[0-4]', experiment):
        ram_exp = 'RAM_{}'.format(experiment[0].capitalize() + experiment[1:])
    else:
        ram_exp = 'RAM_{}'.format(experiment)
    events_dir = os.path.join(DATA_ROOT, '..', 'events', ram_exp)
    events_files = sorted(glob.glob(os.path.join(events_dir, '{}*_events.mat'.format(protocol.upper()))))
    for events_file in events_files:
        subject = '_'.join(os.path.basename(events_file).split('_')[:-1])
        if '_' in subject:
            continue
        mat_events_reader = BaseEventReader(filename=events_file, common_root=DATA_ROOT)
        log('Loading matlab events {exp}: {subj}'.format(exp=experiment, subj=subject))
        try:
            mat_events = mat_events_reader.read()
            sessions = np.unique(mat_events['session'])
            version_str = mat_events[20]['expVersion'] if 'expVersion' in mat_events.dtype.names else '0'
            version = 0
            seen_experiments = []
            try:
                version = float(version_str.split('_')[-1])
            except:
                try:
                    version = float(version_str.split('v')[-1])
                except:
                    pass
            for i, session in enumerate(sessions):
                experiments = np.unique(mat_events[mat_events['session'] == session]['experiment'])
                for this_experiment in experiments:
                    #this_experiment = 'PS2' if this_experiment == 'PS2.1' else this_experiment
                    yield (subject, seen_experiments.count(this_experiment), session, this_experiment, version)
                    seen_experiments.append(this_experiment)


        except AttributeError:
            log('Failed.')

from collections import defaultdict

infinite_dict = lambda: defaultdict(lambda: infinite_dict())

subjects = infinite_dict()

for subject, session, orig_session, experiment, version in get_subject_sessions_by_experiment('PS'):
    print subject, session, experiment, version

    if experiment == 'PS0':
        continue
    if experiment == 'PS2.1':
        version = 2.1
        sys = 'system_2'
        orig_exp = 'PS'
        experiment = 'PS2'
    elif version >= 2:
        orig_exp='PS'
        sys = 'system_2'
    else:
        orig_exp='PS'
        sys = 'system_1'

    subjects[subject][experiment][session]['original_experiment'] = orig_exp
    subjects[subject][experiment][session]['original_session'] = orig_session


    subjects[subject][experiment][session][sys] = True

    print json.dumps(subjects[subject][experiment], indent=2)

json.dump(subjects, open('ps_sessions.json','w'), indent=2, sort_keys=True)