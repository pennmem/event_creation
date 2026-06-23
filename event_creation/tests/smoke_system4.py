"""
Distributed full-import smoke test over System-4 (Elemem) sessions, via CML Dask (SLURM).

Imports each discovered System-4 session through the real pipeline (EEG split + events +
heartbeat correction) on a SLURM worker, classifies the outcome, and prints a summary.
Each session runs in its own worker (clean process state, own temp db_root deleted
immediately), so there is no shared-singleton pollution and disk stays bounded.

Must run as the maintenance account that owns /data/eeg (e.g. RAM_maint), from the repo root:

    PY=/usr/global/ubuntu/miniforge3/25.3.1/envs/workshop_311_rhino2b/bin/python
    $PY -m event_creation.tests.smoke_system4 --max-n-jobs 50 --out /tmp/sys4_smoke.json

Options:
    --job-name NAME        SLURM job name (default sys4_smoke)
    --memory-per-job MEM   memory per worker (default 20GB)
    --max-n-jobs N         max concurrent SLURM workers (default 50)
    --threads-per-job N    threads per worker (default 1)
    --queue Q              SLURM partition (default RAM)
    --log-directory DIR    dask worker logs (default ~/logs/)
    --no-adapt             disable adaptive scaling (fixed max-n-jobs workers)
    --experiments ...      restrict to these experiments (default: all sys4)
    --limit N              only run the first N discovered sessions (quick trial)
    --out PATH             write the full per-session results as JSON

Outcome categories per session:
    pass               import succeeded and heartbeat A/B checks held
    hb_fail            import succeeded but a heartbeat invariant was violated
    xfail_missing      import failed due to missing/unreadable source data (not a code break)
    fail               import failed with a real (code) error
    error              the smoke harness itself errored on this session
"""
import argparse
import json
import os
import shutil
import tempfile
import traceback

STIM_TYPES = {'STIM', 'STIM_ON', 'STIM_OFF', 'STIMMING'}
_MISSING_DATA_MARKERS = (
    'is required, but cannot be found', 'ConfigurationError', 'Permission denied',
    # Incomplete annotation: a recall trial has a .lst but no .ann yet — the
    # pipeline can't score recalls, equivalent to missing data (not a code bug).
    'NoAnnotationError')


def _is_missing_data(errors):
    text = errors or ''
    return any(marker in text for marker in _MISSING_DATA_MARKERS)


def _source_data_missing(case, config):
    """True if the subject's raw data dir is absent (e.g. a dangling symlink) -
    no system can match, so it's genuinely missing data, not a code bug.
    os.path.exists follows symlinks and returns False for a dangling target."""
    try:
        code = case['subject_code'].split('_')[0]
        return not os.path.exists(os.path.join(config.paths.data_root, code))
    except Exception:
        return False


def _find_task_events(db_root, case):
    """os.walk (followlinks) for the produced task_events.json for this session."""
    from ..submission.viewers.recarray import from_json
    base = os.path.join(db_root, 'protocols', 'r1', 'subjects')
    sess = str(case['session'])
    seen = set()
    found = []
    for root, _dirs, files in os.walk(base, followlinks=True):
        if 'task_events.json' in files:
            real = os.path.realpath(os.path.join(root, 'task_events.json'))
            if real in seen:
                continue
            seen.add(real)
            found.append(os.path.join(root, 'task_events.json'))
    for p in found:
        parts = p.split(os.sep)
        if case['subject'] in parts and (sess in parts or ('session_%s' % sess) in parts):
            return from_json(p)
    return from_json(found[0]) if found else None


def _heartbeat_report(db_root, case):
    """Data-level A/B check on the produced events (no pytest). Returns a dict with ok flag."""
    import numpy as np
    from ..submission.alignment.system4 import _elemem_originated_for

    events = _find_task_events(db_root, case)
    if events is None or events.shape == () or len(events) == 0:
        return dict(ok=False, reason='no task_events produced')
    for f in ('mstime_uncorrected', 'eegoffset_uncorrected'):
        if f not in events.dtype.names:
            return dict(ok=False, reason='%s missing from events' % f)

    deny = {t.upper() for t in _elemem_originated_for(case['experiment'])} | \
           {t.upper() for t in STIM_TYPES}
    types_u = np.array([str(t).upper() for t in events['type']])
    is_locked = np.isin(types_u, list(deny))
    is_task = ~is_locked

    report = dict(n_events=int(len(events)), n_task=int(is_task.sum()),
                  n_locked=int(is_locked.sum()))

    # A: task events' mstime changed (the inverse fit remaps every event's mstime).
    if is_task.any():
        report['task_mstime_unchanged'] = int(
            (is_task & (events['mstime'] == events['mstime_uncorrected'])).sum())

    # A2: the correction must actually move eegoffset. Near-EEGSTART task events legitimately
    #     round to 0 samples, so only a TOTALLY static eegoffset (the original
    #     "eegoffset == eegoffset_uncorrected everywhere" bug) is a failure.
    if is_task.any():
        report['task_eegoffset_static'] = int(
            (is_task & (events['eegoffset'] == events['eegoffset_uncorrected'])).sum())

    # B2: locked (STIM/Elemem-originated) events keep the plain host eegoffset (unchanged).
    #     Their mstime IS remapped onto the task clock by design, so it is NOT checked.
    if is_locked.any():
        report['locked_eegoffset_moved'] = int(
            (is_locked & (events['eegoffset'] != events['eegoffset_uncorrected'])).sum())

    # C2: eegoffset is the CORRECTED mstime converted to EEG samples. Both are exact affine
    #     functions of the host mstime, so eegoffset is an exact affine function of the corrected
    #     (task-clock) mstime -- fit that line over TASK events ONLY and require a small residual.
    #     Tolerance scales with the fitted slope a (samples/ms ~= sr/1000): integer mstime is
    #     quantized to +-0.5 ms = +-0.5*a samples, plus eegoffset's +-0.5 rounding. A wrong-ms-
    #     channel eegoffset falls off it. Do NOT mix in the uncorrected (host-clock) pair.
    if is_task.sum() >= 2:
        xt = events['mstime'][is_task].astype(float)
        yt = events['eegoffset'][is_task].astype(float)
        if np.unique(xt).size >= 2:
            a_s, c_s = np.polyfit(xt, yt, 1)
            report['eegoffset_vs_corrected_mstime_resid_max'] = float(
                np.abs(yt - (a_s * xt + c_s)).max())
            report['eegoffset_resid_tol'] = max(2.0, 2.0 * abs(float(a_s)))

    n_task = report.get('n_task', 0)
    correction_applied = (n_task == 0) or (report.get('task_eegoffset_static', 0) < n_task)
    resid = report.get('eegoffset_vs_corrected_mstime_resid_max', 0.0)
    tol = report.get('eegoffset_resid_tol', 2.0)
    report['ok'] = (report.get('task_mstime_unchanged', 0) == 0
                    and report.get('locked_eegoffset_moved', 0) == 0
                    and correction_applied
                    and resid <= tol)
    return report


def _import_session(case, config, db_root):
    """Import one session via convenience, WITHOUT IndexAggregatorTask.

    IndexAggregatorTask.PROTOCOLS_DIR is frozen at import time from paths.db_root (default
    '/'), so run_single_subject would scan the real /protocols and fail under a per-session
    temp db_root. The smoke test reads task_events.json directly (via glob), so it doesn't
    need the aggregated index. Mirrors regression_tests.run_session_import minus that step.
    """
    import shutil
    from event_creation.submission import convenience

    subject_code = case['subject_code']
    subject, montage = subject_code.split('_') if '_' in subject_code else (subject_code, '')
    montage = '0.0' if not montage else '0.%s' % montage
    orig = case.get('original_session', case['session'])
    config.parse_args(['--set-input',
        'code={}:experiment={}:session={}:original_session={}:montage={}'.format(
            subject_code, case['experiment'], case['session'], orig, montage)])

    loc_root = os.path.join('protocols', 'r1', 'subjects', subject, 'localizations', '0')
    src = os.path.join(config.paths.rhino_root, loc_root)
    if not os.path.isdir(os.path.join(db_root, loc_root)) and os.path.isdir(src):
        shutil.copytree(src, os.path.join(db_root, loc_root), symlinks=True)

    session_inputs = convenience.prompt_for_session_inputs(config.inputs)
    session_inputs['exp_version'] = -1
    success, importer_collection = convenience.run_session_import(
        session_inputs, do_import=True, force_events=True)
    return success, importer_collection.describe_errors()


def smoke_one_session(case):
    """Import one sys4 session on a (clean) worker and classify the result.

    Top-level so it is picklable for dask ``client.map``. All heavy imports are done inside
    so the function ships cheaply to workers.
    """
    result = dict(case=case, status='error', detail='')
    db_root = tempfile.mkdtemp(prefix='evcreate_smoke_')
    try:
        from event_creation.submission.configuration import config
        from event_creation.submission import convenience

        config.parse_args(['--path', 'db_root=%s' % db_root])
        for field in list(config.inputs.options.keys()):
            config.inputs.set(field, None)
        convenience.LOADED_INDEXES.clear()

        try:
            success, errors = _import_session(case, config, db_root)
        except Exception:
            success, errors = False, traceback.format_exc()

        if success:
            hb = _heartbeat_report(db_root, case)
            if hb.get('reason') == 'no task_events produced':
                # Import reported success but the events builder didn't produce events
                # (success-flag masking). Surface the importer errors so we can see why,
                # and bucket as missing-data if that's the cause.
                if _is_missing_data(errors):
                    result['status'] = 'xfail_missing'
                else:
                    result['status'] = 'no_events'
                result['detail'] = {'hb': hb, 'errors': (errors or '')[:1500]}
            else:
                result['status'] = 'pass' if hb.get('ok') else 'hb_fail'
                result['detail'] = hb
        elif _is_missing_data(errors):
            result['status'] = 'xfail_missing'
            result['detail'] = (errors or '')[:400]
        elif 'determination failed' in (errors or '') and _source_data_missing(case, config):
            result['status'] = 'xfail_missing'
            result['detail'] = 'subject data directory missing (dangling symlink): ' + (errors or '')[:300]
        else:
            result['status'] = 'fail'
            result['detail'] = (errors or '')[:2000]
    except Exception:
        result['status'] = 'error'
        result['detail'] = traceback.format_exc()[:2000]
    finally:
        shutil.rmtree(db_root, ignore_errors=True)
    return result


def _label(case):
    return '{subject}_{experiment}_{session}'.format(**case)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Distributed System-4 import smoke test (CML Dask/SLURM)')
    parser.add_argument('--job-name', default='sys4_smoke')
    parser.add_argument('--memory-per-job', default='20GB')
    parser.add_argument('--max-n-jobs', type=int, default=50)
    parser.add_argument('--threads-per-job', type=int, default=1)
    parser.add_argument('--queue', default='RAM')
    parser.add_argument('--log-directory', default='~/logs/')
    parser.add_argument('--adapt', action='store_true', default=True)
    parser.add_argument('--no-adapt', dest='adapt', action='store_false')
    parser.add_argument('--experiments', nargs='*')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--out')
    args = parser.parse_args(argv)

    import cmldask.CMLDask as da
    from dask.distributed import as_completed
    from event_creation.submission.configuration import config
    from event_creation.tests.conftest import discover_system4_sessions

    cases = discover_system4_sessions(config)
    if args.experiments:
        cases = [c for c in cases if c['experiment'] in args.experiments]
    if args.limit:
        cases = cases[:args.limit]

    print('Discovered %d System-4 sessions.' % len(cases))
    if not cases:
        return

    log_dir = os.path.expanduser(args.log_directory)
    os.makedirs(log_dir, exist_ok=True)

    client = da.new_dask_client_slurm(
        job_name=args.job_name,
        memory_per_job=args.memory_per_job,
        max_n_jobs=args.max_n_jobs,
        threads_per_job=args.threads_per_job,
        adapt=args.adapt,
        queue=args.queue,
        log_directory=log_dir,
    )
    print('Dask client up; submitting %d jobs (queue=%s, max_n_jobs=%d)...'
          % (len(cases), args.queue, args.max_n_jobs))

    results = []
    counts = {}
    futures = client.map(smoke_one_session, cases)
    done = 0
    for fut in as_completed(futures):
        try:
            res = fut.result()
        except Exception:
            res = dict(case={'note': 'future raised'}, status='error',
                       detail=traceback.format_exc()[:2000])
        results.append(res)
        counts[res['status']] = counts.get(res['status'], 0) + 1
        done += 1
        label = _label(res['case']) if 'subject' in res.get('case', {}) else fut.key
        print('[%d/%d] %-40s %s' % (done, len(cases), label, res['status']))

    print('\n==================== SUMMARY ====================')
    for status in ('pass', 'hb_fail', 'fail', 'xfail_missing', 'error'):
        if status in counts:
            print('  %-14s %d' % (status, counts[status]))
    print('  total          %d' % len(results))

    # Surface the real failures (the "what breaks" signal).
    for res in results:
        if res['status'] in ('fail', 'hb_fail', 'error'):
            detail = res['detail'] if isinstance(res['detail'], str) else json.dumps(res['detail'])
            print('\n--- %s [%s] ---\n%s' % (_label(res['case']) if 'subject' in res.get('case', {})
                                             else '?', res['status'], detail))

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print('\nFull results written to %s' % args.out)

    try:
        client.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
