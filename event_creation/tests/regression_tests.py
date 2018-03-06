from __future__ import print_function
import os,tempfile,shutil,json,traceback
from ptsa.data.readers import JsonIndexReader,CMLEventReader
from event_creation.submission.parsers.base_log_parser import EventComparator
import argparse
import matplotlib
import contextlib
matplotlib.use('agg')

debug = False

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
    from event_creation.submission import convenience
    subject = subject_code.split('_')[0]
    localization_inputs = {'subject':subject,'code':subject_code,'localization':localization_number,'protocol':'r1'}
    return convenience.run_localization_import(localization_inputs)


def run_session_import(subject_code,experiment,session):
    from event_creation.submission import convenience
    subject,montage = subject_code.split('_') if '_' in subject_code else (subject_code,'')

    if not montage:
        montage = '0.0'
    else:
        montage = '0.%s'%montage

    config.parse_args(['--set-input', 'code={}:experiment={}:session={}:montage={}'.format(
        subject_code, experiment, session,montage)])


    localization_root = os.path.join('protocols','r1','subjects',subject,
                             'localizations','0')
    if not os.path.isdir(os.path.join(db_root,localization_root)):
        shutil.copytree(os.path.join(config.paths.rhino_root,localization_root),
                        os.path.join(db_root,localization_root),
                        symlinks=True)

    session_inputs = convenience.prompt_for_session_inputs(config.inputs,)
    session_inputs['exp_version']=-1
    success, importer_collection = convenience.run_session_import(session_inputs,do_import=True,force_events=True)
    convenience.IndexAggregatorTask().run_single_subject(subject,'r1')
    return success , importer_collection.describe_errors()

# def compare_equal_localizations(subject_code,localization_number):
#     run_localization_import(subject_code,localization_number)
#     new_localization  = LocReader(filename=os.path.join(db_root,'protocols','r1','subjects',subject_code.split('_')[0],
#                                                         'localizations',localization_number,
#                                                         'neuroradiology','current_processed','localization.json')).read()
#     old_localization = LocReader(
#         filename=os.path.join(config.paths.rhino_root, 'protocols', 'r1', 'subjects', subject_code.split('_')[0],
#                               'localizations', localization_number,
#                               'neuroradiology', 'current_processed', 'localization.json')).read()
#
#     return (old_localization==new_localization).all()
#

def compare_equal_events(subject, experiment, session):
    success, msg = run_session_import(subject, experiment, session)
    if not success:
            return False, 'Import Failed for %s %s_%s : %s'%(subject,experiment,session,msg)
    new_jr = JsonIndexReader(os.path.join(db_root,'protocols','r1.json'))
    new_events = CMLEventReader(filename=new_jr.get_value('task_events',subject=subject,
                                                          experiment=experiment,session=session)).read()
    old_jr = JsonIndexReader(os.path.join(config.paths.rhino_root,'protocols','r1.json'))
    old_events = CMLEventReader(filename=old_jr.get_value('task_events',subject=subject,
                                                          experiment=experiment,session=session)).read()
    flat_comparison = EventComparator(events1=old_events,events2=new_events,same_fields=False,
                                      field_ignore=['stim_params','test','eegfile','msoffset','exp_version'],
                                      verbose=True)

    failure,msg= flat_comparison.compare()
    return not failure, msg


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


def run_test(test_function,input_file,experiments = tuple(),subjects=tuple()):
    with open(input_file) as ifj:
        test_cases = json.load(ifj)
    successes = []
    failures = []
    for case in sorted(test_cases):
        if (not experiments or case['experiment'] in experiments) and (not subjects or case['subject'] in subjects):
            try:
                success, msg = test_function(**case)
                if success:
                    successes.append(case)
                else:
                    case['error']=msg
                    failures.append(case)
            except Exception as e:
                if debug:
                    raise
                else:
                    case['error'] = traceback.format_exc()
                    failures.append(case)
    return successes,failures

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-root')
    parser.add_argument('--experiments',nargs = '*')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--subjects', nargs = '*')
    return parser


def format_result_summary(success_records, failure_records):
    full_template = """
    Regression Test Results:

    %d Successes:
    ------------------------------
    %s
    ==============================
    %d Failures:
    ------------------------------
    %s

    ===============================
    db_root = %s
    """
    success_str = '\n'.join(['{subject}, {experiment}_{session}'.format(**s) for s in success_records])
    failure_str = '\n'.join(['{subject}, {experiment}_{session}'.format(**f) for f in failure_records])
    return full_template%(len(success_records),success_str,len(failure_records),failure_str,db_root)


def format_errors(error_records):
    """
    :param error_records: {List[Dict["subject","experiment","session","error"]]}
    :return:
    """
    err_template = """
    {subject}\t{experiment}\t{session}\t
    Error:
    {error}
    ###
    """
    return '\n'.join([err_template.format(**case) for case in error_records])

if __name__ == '__main__':
    args = parser().parse_args()
    debug = args.debug
    from event_creation.submission.configuration import config
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.application import MIMEApplication
    successes = {}
    failures = {}
    crashes = {}
    here = os.path.dirname(__file__)
    with init_db_root(db_root=args.db_root) as db_root:
        config.parse_args(['--path','db_root=%s'%db_root])
        successes,failures = run_test(compare_equal_events,os.path.join(here,'regression_sessions.json'),
                                      experiments=args.experiments,subjects=args.subjects)
        # TODO: Add this back in
        # successes['localization'],failures['localization'],crashes['localization'] = run_test(compare_equal_localizations,
        #                                                                                       'regression.json',db_root)
    result_summary = MIMEText(format_result_summary(successes, failures))
    err_msg = format_errors(failures)
    with open(os.path.join(here,'regression_errors.txt'),'w') as results:
        results.write(err_msg)
    err_attatchment = MIMEApplication(err_msg,Name='regression_errors.txt')
    err_attatchment['Content-Disposition'] = 'attachment; filename=regression_errors.txt'
    # Email addresses should be configurable?
    message = MIMEMultipart()
    from_ = 'RAM_maint@rhino2.psych.upenn.edu'
    to_ = 'leond@sas.upenn.edu'
    message['Subject'] = 'Post_Processing Regression Tests'
    message['From'] = from_
    message['To'] = to_
    message.attach(result_summary)
    message.attach(err_attatchment)
    s = smtplib.SMTP('localhost')
    s.sendmail(from_, [to_], message.as_string())
    s.quit()
