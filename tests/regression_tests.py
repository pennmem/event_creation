from __future__ import print_function
import os,tempfile,shutil,json,traceback
from ptsa.data.readers import JsonIndexReader,CMLEventReader,LocReader
from submission.parsers.base_log_parser import EventComparator,StimComparator
import argparse
import matplotlib
import contextlib
matplotlib.use('agg')

@contextlib.contextmanager
def init_db_root(db_root = None,whitelist = (KeyboardInterrupt,)):
    """
    Context generator that makes a temporary directory,
    and removes it if no unexpected error occurs.
    :param whitelist: A tuple of exception classes that still let us clean up the directory
    """

    erase = False
    if db_root is None:
        db_root = tempfile.mkdtemp()
        erase = True
    else:
        db_root = db_root
    whitelist = whitelist
    try:
        yield db_root
    except Exception as e:
        if issubclass(type(e),whitelist):
            erase = True
        else:
            erase = False
            raise
    finally:
        if erase:
            shutil.rmtree(db_root)

def run_localization_import(subject_code,localization_number):
    from submission import convenience
    subject = subject_code.split('_')[0]
    localization_inputs = {'subject':subject,'code':subject_code,'localization':localization_number,'protocol':'r1'}
    return convenience.run_localization_import(localization_inputs)


def run_session_import(subject_code,experiment,session):
    from submission import convenience
    subject,montage = subject_code.split('_') if '_' in subject_code else (subject_code,'')

    if not montage:
        montage = '0.0'
    else:
        montage = '0.%s'%montage

    config.parse_args(['--set-input', 'code={}:experiment={}:session={}:montage={}'.format(
        subject_code, experiment, session,montage)])

    montage_inputs = {'code':subject_code,'montage':montage,'protocol':'r1','subject':subject,'reference_scheme':'monopolar'}
    success, _ = convenience.run_montage_import(montage_inputs,do_convert=True)
    if not success:
        localization_root = os.path.join('protocols','r1','subjects',subject,
                                 'localizations','0')
        if not os.path.isdir(localization_root):
            shutil.copytree(os.path.join(config.paths.rhino_root,localization_root),
                            os.path.join(db_root,localization_root),
                            symlinks=True)

    session_inputs = convenience.prompt_for_session_inputs(config.inputs,)
    success,_ = convenience.run_session_import(session_inputs)
    convenience.IndexAggregatorTask().run_single_subject(subject,'r1')
    return success

def compare_equal_localizations(subject_code,localization_number):
    run_localization_import(subject_code,localization_number)
    new_localization  = LocReader(filename=os.path.join(db_root,'protocols','r1','subjects',subject_code.split('_')[0],
                                                        'localizations',localization_number,
                                                        'neuroradiology','current_processed','localization.json')).read()
    old_localization = LocReader(
        filename=os.path.join(config.paths.rhino_root, 'protocols', 'r1', 'subjects', subject_code.split('_')[0],
                              'localizations', localization_number,
                              'neuroradiology', 'current_processed', 'localization.json')).read()

    return (old_localization==new_localization).all()


def compare_equal_events(subject, experiment, session):
    assert run_session_import(subject, experiment, session)
    new_jr = JsonIndexReader(os.path.join(db_root,'protocols','r1.json'))
    new_events = CMLEventReader(filename=new_jr.get_value('all_events',subject=subject,
                                                          experiment=experiment,session=session)).read()
    old_jr = JsonIndexReader(os.path.join(config.paths.rhino_root,'protocols','r1.json'))
    old_events = CMLEventReader(filename=old_jr.get_value('all_events',subject=subject,
                                                          experiment=experiment,session=session)).read()
    flat_comparison = EventComparator(events1=old_events,events2=new_events,
                                      field_ignore=('stim_params','test','eegfile','msoffset'))

    return flat_comparison.compare()


def compare_contains(subject,experiment,session):
    assert run_session_import(subject, experiment, session)
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


def run_test(test_function,input_file,experiments = tuple()):
    with open(input_file) as ifj:
        test_cases = json.load(ifj)
    successes = []
    failures = []
    for case in test_cases:
        if not experiments or case['experiment'] in experiments:
            success, msg = test_function(**case)
            if success:
                successes.append(case)
            else:
                case['error']=msg
                failures.append(case)
    return successes,failures

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-root')
    parser.add_argument('--experiments',action = 'append')
    return parser

if __name__ == '__main__':
    args = parser().parse_args()
    from submission.configuration import config
    import smtplib
    from email.mime.text import MIMEText
    successes = {}
    failures = {}
    crashes = {}
    with init_db_root(db_root=args.db_root) as db_root:
        config.parse_args(['--path','db_root=%s'%db_root])
        successes['events'],failures['events'] = run_test(compare_equal_events,
                                                                            os.path.join(os.path.dirname(__file__),
                                                                                         'regression_sessions.json'),
                                                                            experiments=args.experiments)
        # TODO: Add this back in
        # successes['localization'],failures['localization'],crashes['localization'] = run_test(compare_equal_localizations,
        #                                                                                       'regression.json',db_root)
    result_summary = MIMEText('Successes: \n%s\nFailures: \n%s\ndb_root:%s'%(
        json.dumps(successes,indent=2),json.dumps(failures,indent=2),db_root))

    # Email addresses should be configurable?

    from_ = 'RAM_maint@rhino2.psych.upenn.edu'
    to_ = 'leond@sas.upenn.edu'
    result_summary['Subject'] = 'Post_Processing Regression Tests'
    result_summary['From'] = from_
    result_summary['To'] = to_
    s = smtplib.SMTP('localhost')
    s.sendmail(from_, [to_], result_summary.as_string())
    s.quit()
