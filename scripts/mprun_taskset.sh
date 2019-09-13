#!/bin/bash

NUM_PROC=32
CPUS_PER_TASK=1


# kills all child if signal received
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# place for logs
mkdir -p 'out'

echo "start running $NUM_PROC processes..."
for i in $(seq 0 $(($NUM_PROC-1)))
do

    CPU_START=$(( $i * $CPUS_PER_TASK ))
    CPU_END=$(( $i * $CPUS_PER_TASK + $CPUS_PER_TASK -1 ))
    TASKSET_CPU="$CPU_START-$CPU_END"


    # echo $i / $NUM_PROC
    # sleep 3 && echo $i & # runs background process called

    # python -u script_mp.py $i $NUM_PROC > out/out_rank$i.txt 2> out/err_rank$i.txt
    OMP_NUM_THREADS=1 TASKSET_CPU=$TASKSET_CPU ./scripts/run.sh $i $NUM_PROC > out/out_rank$i.txt 2>&1 &

    # taskset -c $TASKSET_CPU python -u src/test.py $i $NUM_PROC test 5.0

done

wait # waits for background process called in this script
echo "$NUM_PROC processes done"


### MEMO

## easy to remember for loop
# END=4; for i in $(seq 1 $END); do echo $i; done

# keep looking at constantly changing text(log)

# tail -f out/out_rank0.txt

## TODO
# need to put timestamp in <time>_out_rank0.out
