def ClusterRunSlurm(function, parameter_list, max_jobs=64, procs_per_job=1,
    mem='5GB'):
  '''function: The routine run in parallel, which must contain all necessary
     imports internally.

     parameter_list: should be an iterable of elements, for which each
     element will be passed as the parameter to function for each parallel
     execution.

     max_jobs: Standard Rhino cluster etiquette is to stay within 100 jobs
     running at a time.  Please ask for permission before using more.

     procs_per_job: The number of concurrent processes to reserve per job.

     mem: A string specifying the amount of RAM required per job, formatted
     like '5GB'.  Standard Rhino cluster etiquette is to stay within 320GB
     total across all jobs.

     In jupyterlab, the number of engines reported as initially running may
     be smaller than the number actually running.  Check usage from an ssh
     terminal using:  "squeue" or "squeue -u $USER"

     Undesired running jobs can be killed by reading the JOBID at the left
     of that squeue command, then doing:  scancel JOBID
  '''
  import cmldask.CMLDask as da
  from dask.distributed import wait, as_completed, progress
  import os

  num_jobs = len(parameter_list)
  num_jobs = min(num_jobs, max_jobs)

  with da.new_dask_client_slurm(function.__name__, mem, max_n_jobs=num_jobs,
      processes_per_job=procs_per_job, walltime='23:00:00',
      log_directory=os.path.join(os.environ['HOME'], 'logs',
        'event_creation_outputs')) as client:
    futures = client.map(function, parameter_list)
    wait(futures)
    res = client.gather(futures)

  return res

def ClusterChecked(function, parameter_list, *args, **kwargs):
  '''Parallelizes and raises an exception if any results are False.'''
  res = ClusterRunSlurm(function, parameter_list, *args, **kwargs)
  if all(res):
    print('All', len(res), 'jobs successful.')
  else:
    failed = sum([not bool(b) for b in res])
    if failed == len(res):
      raise RuntimeError('All '+str(failed)+' jobs failed!')
    else:
      print('Error on job parameters:\n  ' + 
          '\n  '.join(str(parameter_list[i]) for i in range(len(res))
            if not bool(res[i])))
      raise RuntimeError(str(failed)+' of '+str(len(res))+' jobs failed!')

def ClusterCheckedTup(function, parameter_list, *args, **kwargs):
  '''Parallelizes and raises an exception if any results have False
     as the first value of the returned list/tuple.'''
  res = ClusterRunSlurm(function, parameter_list, *args, **kwargs)
  if all([bool(b[0]) for b in res]):
    print('All', len(res), 'jobs successful.')
  else:
    failed = sum([not bool(b[0]) for b in res])
    if failed == len(res):
      raise RuntimeError('All '+str(failed)+' jobs failed!')
    else:
      print('Error on job parameters:\n  ' + 
          '\n  '.join(str(parameter_list[i]) for i in range(len(res))
            if not bool(res[i][0])))
      raise RuntimeError(str(failed)+' of '+str(len(res))+' jobs failed!')

