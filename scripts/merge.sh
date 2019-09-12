#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Provide two arg: rank world size"
    exit -1
fi

# python -u src/test.py $1 $2 | tee -i out_test_$1of$2.txt


# python -u src/merge_results.py $1 $2 train 5.0
# python -u src/merge_results.py $1 $2 train 10.0
# python -u src/merge_results.py $1 $2 train 15.0
# python -u src/merge_results.py $1 $2 train 20.0
# python -u src/merge_results.py $1 $2 train 25.0


python -u src/merge_results.py $1 $2 test 5.0
# python -u src/merge_results.py $1 $2 test 10.0
# python -u src/merge_results.py $1 $2 test 15.0
# python -u src/merge_results.py $1 $2 test 20.0
# python -u src/merge_results.py $1 $2 test 25.0