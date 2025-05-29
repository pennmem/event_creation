#!/usr/global/shared/runvenv workshop

import os
import sys
import datetime

if len(sys.argv) < 2 or '--help' in sys.argv[1:]:
  print(f'{sys.argv[0]} <script_dir> <exp:sub:sess> [exp:sub:sess] ...')
  print('  - With environment variable SLURM_ARRAY_TASK_ID indicating which index of')
  print('    exp:sub:sess to process.  This is used for sbatch runs.')
  print(f'{sys.argv[0]} <exp:sub:sess>')
  print('  - For manual running, defaults to SLURM_ARRAY_TASK_ID of 0 if not set.')
  print('    This also finds its own script directory the normal way.')
  sys.exit(-1)

if len(sys.argv) > 2:
  script_dir = sys.argv.pop(1)
else:
  script_dir = os.path.dirname(os.path.realpath(__file__))

array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
target = sys.argv[1+array_id]
exp,sub,sess = target.split(':')
logfile = os.path.join(os.environ['HOME'], 'logs',
    f'automatic_run_{exp}_{sub}_{sess}.log')
slurm_jobid = os.environ.get('SLURM_JOB_ID', 'None')

with open(logfile, 'w') as fw:
  start_time = datetime.datetime.now()
  datestr = start_time.strftime('%F %H:%M:%S')
  fw.write(f'{datestr} - Beginning run of {exp} {sub} {sess}\n')
  fw.write(f'Slurm job info:\n')
  info = {
      'hostname': os.environ['HOSTNAME'], 'pid':os.getpid(),
      'array ID': array_id,
      'slurm jobid': slurm_jobid}
  for k,v in info.items():
    fw.write(f'  {k}: {v}\n')
  fw.write(f'Running in: {script_dir}\n')

os.chdir(script_dir)

os.system(f'squeue -o "%.11i %.5P %.8j %.8u %.2t %.11M %.11L %.2c %.5m %.6N"'
    + f' -j {slurm_jobid} &>>{logfile}')

if exp == 'CourierReinstate1':
  fix_script = os.path.join(os.environ['HOME'], 'reinstatement_fix_scripts',
      'fix_one_session_jsonl.sh')
  os.system(f'{fix_script} {sub} {sess} &>>{logfile}')

os.system('./submit --set-input ' +
    f'"protocol=ltp:code={sub}:experiment={exp}:session={sess}:montage=0.0"' +
    f' &>>{logfile}')

def TotalTimeStr(time_delta):
  total = time_delta.total_seconds()
  sparts = []
  if total > 60*60*24:
    days = total // (60*60*24)
    total -= days * 60*60*24
    label = 'days' if days > 1 else 'day'
    sparts.append(f'{days:0.0f} {label}')
  if total > 60*60:
    hours = total // (60*60)
    total -= hours * 60*60
    sparts.append(f'{hours:0.0f} hr')
  if total > 60:
    minutes = total // 60
    total -= minutes * 60
    sparts.append(f'{minutes:0.0f} min')
  sparts.append(f'{total:0.3f} s')
  return ', '.join(sparts)

with open(logfile, 'a') as fw:
  end_time = datetime.datetime.now()
  run_time = TotalTimeStr(end_time - start_time)
  datestr = end_time.strftime('%F %H:%M:%S')
  fw.write(f'{datestr} - Completed run of {exp} {sub} {sess} in {run_time}.\n')

