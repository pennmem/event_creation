from submission import convenience
from submission.configuration import config
import os,tempfile,shutil,json,traceback
from ptsa.data.readers import JsonIndexReader,CMLEventReader


class init_db_root(object):
    """Context generator that makes a temporary directory,
    and removes it if no error occurs"""
    def __init__(self):
        self.db_root = tempfile.mkdtemp()

    def __enter__(self):
        return self.db_root

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            shutil.rmtree(self.db_root)
        return False


def run_session_import(subject_code,experiment,session,db_root):
    config.parse_args('--path', 'db_root=%s' % db_root,
                      '--set-input', 'code={}:experiment={}:session={}'.format(subject_code, experiment, session))
    subject,montage = subject_code.split('_')
    if not montage:
        montage = '0.0'
    else:
        montage = '0.%s'%montage
    montage_inputs = {'code':subject_code,'montage':montage,'protocol':'r1','subject':subject,'reference_scheme':'monopolar'}
    convenience.run_montage_import(montage_inputs,do_convert=True)
    session_inputs = convenience.prompt_for_session_inputs(**config.options)
    return convenience.run_session_import(session_inputs)


def compare_equal(subject,experiment,session,db_root=None):
    if db_root is  None:
        db_root = init_db_root().db_root
    run_session_import(subject, experiment, session, db_root)
    new_jr = JsonIndexReader(os.path.join(db_root,'protocols','r1.json'))
    new_events = CMLEventReader(filename=new_jr.get_value('all_events',subject=subject,
                                                          experiment=experiment,session=session)).read()
    old_jr = JsonIndexReader(os.path.join(config.paths.rhino_root,'protocols','r1.json'))
    old_events = CMLEventReader(filename=old_jr.get_value('all_events',subject=subject,
                                                          experiment=experiment,session=session)).read()
    return (old_events==new_events).all()


def compare_contains(subject,experiment,session,db_root=None):
    if db_root is  None:
        db_root = init_db_root().db_root

    run_session_import(subject, experiment, session,db_root)
    new_jr = JsonIndexReader(os.path.join(db_root, 'protocols', 'r1.json'))
    new_events = CMLEventReader(filename=new_jr.get_value('all_events', subject=subject,
                                                          experiment=experiment, session=session)).read()
    old_jr = JsonIndexReader(os.path.join(config.paths.rhino_root, 'protocols', 'r1.json'))
    old_events = CMLEventReader(filename=old_jr.get_value('all_events', subject=subject,
                                                          experiment=experiment, session=session)).read()

    fields_match = all([n in new_events.dtype.names for n in old_events.dtype.names])
    if fields_match:
        return all([(old_events[f]==new_events[f]).all() for f in old_events.dtype.names])
    else:
        return False


def test():
    with init_db_root():
        with open('regression.json') as jf:
            test_cases = json.load(jf)
        successes = []
        failures = []
        crashes = []
        for case in test_cases:
            try:
                if compare_equal(**case):
                    successes.append(case)
                else: failures.append(case)
            except Exception :
                tb = traceback.print_exc()
                case['error'] = tb
                crashes.append(case)
    return successes,failures,crashes

if __name__ == '__main__':
    import smtplib
    from email.mime.text import MIMEText

    successes,failures,crashes = test()
    result_summary = MIMEText('Successes: \n%s\nFailures: \n%s\nCrashes:\n%s\n'%(successes,failures,crashes))
    from_ =  'leond@rhino2.psych.upenn.edu'
    to_ = 'leond@sas.upenn.edu'
    result_summary['Subject'] = 'Event_creation Regression Tests'
    result_summary['From'] = from_
    result_summary['To'] = to_
    s = smtplib.SMTP('localhost')
    s.sendmail(from_, [to_], result_summary.as_string())
    s.quit()
