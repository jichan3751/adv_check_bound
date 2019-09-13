#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Provide two arg: rank world size"
    exit -1
fi

source activate tensorflow_p36
export CUDA_VISIBLE_DEVICES="-1"

# python -u src/test.py $1 $2 | tee -i out_test_$1of$2.txt


# python -u src/test.py $1 $2 test 5.0

# python -u src/test.py $1 $2 test 5.0

# python -u src/test.py $1 $2 test 0.3

# declare -a arr=("1.0")
declare -a arr=("1.0" "2.0" "3.0" "4.0")
# declare -a arr=("0.3")

## now loop through the above array
for eps in "${arr[@]}"
do
    # python -u src/test.py $1 $2 test $eps
    # python -u src/test2.py $1 $2 test $eps # currently adversarial
    taskset -c $TASKSET_CPU python -u src/test.py $1 $2 train $eps
    taskset -c $TASKSET_CPU python -u src/test2.py $1 $2 train $eps
    # or do whatever with individual element of the array
done
