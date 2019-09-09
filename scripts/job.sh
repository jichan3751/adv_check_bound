#!/bin/bash

mkdir -p slurm

#SBATCH --job-name=LT_simulation
#SBATCH --output=slurm/arrayJob_%A_%a.out
#SBATCH --error=slurm/arrayJob_%A_%a.err
#SBATCH --array=0-49
#SBATCH --cpus-per-task=2

######################
# Begin work section #
######################

# Print this sub-job's task ID
# echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
# echo "My SLURM_ARRAY_TASK_COUNT: " $SLURM_ARRAY_TASK_COUNT
python3 -u src/test.py $SLURM_ARRAY_TASK_ID 50 test  10.0
python3 -u src/test.py $SLURM_ARRAY_TASK_ID 50 train 10.0
python3 -u src/test.py $SLURM_ARRAY_TASK_ID 50 test  5.0
python3 -u src/test.py $SLURM_ARRAY_TASK_ID 50 train 5.0
python3 -u src/test.py $SLURM_ARRAY_TASK_ID 50 test  20.0
python3 -u src/test.py $SLURM_ARRAY_TASK_ID 50 train 20.0
python3 -u src/test.py $SLURM_ARRAY_TASK_ID 50 test  15.0
python3 -u src/test.py $SLURM_ARRAY_TASK_ID 50 train 15.0
python3 -u src/test.py $SLURM_ARRAY_TASK_ID 50 test  25.0
python3 -u src/test.py $SLURM_ARRAY_TASK_ID 50 train 25.0


# Do some work based on the SLURM_ARRAY_TASK_ID
# For example:
# ./my_process $SLURM_ARRAY_TASK_ID
#
# where my_process is you executable

######### my memo ###############
# run slurm batch file by sbatch slurm_batch.sh
# $SLURM_ARRAY_TASK_COUNT does not work in icsi cluster!!!!! manually feed in num task!


