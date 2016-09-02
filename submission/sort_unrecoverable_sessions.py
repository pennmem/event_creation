import json
subjects = json.load(open('unrecoverable_sessions.json'))
counter = 0
for subject, experiments in subjects.items():
    for experiment, sessions in experiments.items():
        counter += len(sessions)
print counter
json.dump(subjects, open('unrecoverable_sessions_sorted.json', 'w'), indent=2, sort_keys=True)