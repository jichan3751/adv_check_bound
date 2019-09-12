import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# from tensorflow.examples.tutorials.mnist import input_data

from test import *
# from test_adv import *

def main(RANK, WSIZE,dataset, eps):

    config = {
            "dataset":dataset, # 'train' or 'test'
            "max_epoch": 600,
            "lr": 0.01, # base lr : 0.1, 0.05
            "norm": "l2", # L2, linf
            "load_weight": "adv", # nat, adv, sec
            "eps": eps,
            "num_reruns": 10, # number of different seed run

            # optional
            "img_indices": None # None is run with all dataset
            # "img_indices": range(10,15)
        }

    print('Merge results for:')
    print(config)

    AA = Trainer(config)

    # with Timer(name = 'initialize') as t:
    #     AA.setup()

    # with Timer(name = 'run_train') as t:
    #     AA.run_train_save(rank=RANK, wsize = WSIZE)

    # merging
    if RANK ==0:
        with Timer(name = 'merge_results') as t:
            AA.merge_results(wsize = WSIZE)
    else:
        print("RANK not zero")


if __name__ == "__main__":
    import sys
    assert len(sys.argv)>=3
    RANK = int(sys.argv[1])
    WSIZE = int(sys.argv[2])
    dataset = str(sys.argv[3])
    eps = float(sys.argv[4])
    generate_dirs(['./plots', './results'])
    start_time = time.time()
    main(RANK, WSIZE, dataset ,eps)
    duration = (time.time() - start_time)

    print("---Program Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))



