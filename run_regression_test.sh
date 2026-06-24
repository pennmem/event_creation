#!/usr/bin/env bash
#$ -q RAM.q
#$ -cwd
#$ -N regression_events
#$ -o tests/regression_tests.log
source ~/.bashrc
cd ~/event_creation
rm *.png
source /usr/global/ubuntu/miniforge3/25.3.1/etc/profile.d/conda.sh
conda activate workshop_311_rhino2b

EC_PYTHON="${CONDA_PREFIX:-/usr/global/ubuntu/miniforge3/25.3.1/envs/workshop_311_rhino2b}/bin/python"
"$EC_PYTHON" -m event_creation.tests.regression_tests --db-root=/scratch/db_root
