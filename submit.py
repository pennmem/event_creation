from event_creation.submission.convenience import run_session_import, IndexAggregatorTask
from event_creation.submission.log import logger
import os
import matplotlib

matplotlib.use('agg')
logger.set_stdout_level(0)

def build_inputs(experiment, subject, session):
    inputs = dict(
        protocol='r1',
        subject=subject,
        montage='0.0',
        montage_num='0',
        localization='0',
        experiment=experiment,
        new_experiment=experiment,
        ram_experiment='RAM_%s' % experiment,
        force=False,
        do_compare=False,
        code=subject,
        session=session,
        original_session=session,
        groups=('r1',),
        attempt_import=True,
        attempt_conversion=False,
        PS4=False
    )
    return inputs
# System 4 test
subjects = [
    # ('R1556J_1', 'RepFR1', 0),
    ('FBG490', 'EFRCourierOpenLoop', 2),
    # ('R1384J_0', 'CPS', 0)
]


#subjects = [('R1505J', 'RepFR1', 0), 
#            ('R1316T', 'FR1', 0), 
#            ('R1316T', 'FR1', 1), 
#            ('R1195E', 'FR3', 0),
#            ('R1195E', 'FR3', 1),
#            ('R1271P', 'catFR1', 0),
#            ('R1271P', 'catFR1', 1),
#            ('R1406M', 'catFR5', 0),
#            ('R1406M', 'FR1', 0),
#            ('R1406M', 'FR1', 3),
#            ('R1515T', 'catFR1', 0),
#            ('R1515T', 'catFR1', 1),
#            ('R1515T', 'catFR1', 2),]

# sandbox testing
username = "rdehaan"
sandbox = f'/scratch/{username}/event_creation_sandbox'
# import pdb; pdb.set_trace()
for sub, exp, sess in subjects:
#     src = "/protocols/r1/subjects/{sub}/localizations".format(sub=sub.split('_')[0])
#     print(src)
#     try:
#         if not os.access(sandbox+src, os.F_OK):
#             print(sandbox+os.path.split(src)[0])
#             os.makedirs(sandbox+os.path.split(src)[0])
#             os.symlink(src, sandbox+src, target_is_directory=True)
#     except:
#         print('could not make new dirs')

#     src = "/data/eeg/{sub}".format(sub=sub.split('_')[0])
#     print(src)
#     try:
#         if not os.access(sandbox+src, os.F_OK):
#             print(sandbox+os.path.split(src)[0])
#             os.makedirs(sandbox+os.path.split(src)[0])
#             os.symlink(src, sandbox+src, target_is_directory=True)
#     except:
#         print('could not make new dirs')

    try:
        inputs = build_inputs(exp, sub, sess)

        # attempt import, attempt convert, force_events, force_eeg
        success, importers = run_session_import(inputs, True, False, True, True)
        if success:
            IndexAggregatorTask().run_single_subject(inputs['subject'], inputs['protocol'])
    except:
        with open(f"/scratch/{username}/submit_log.txt", 'a') as f:
            f.write("Error with {sub}:{exp}:{sess}".format(sub=sub, exp=exp, sess=sess))
