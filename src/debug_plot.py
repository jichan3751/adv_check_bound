import matplotlib; matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend; instead, writes files
from matplotlib import pyplot as plt
import time

import numpy as np
import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data

from util import *
from model import *
from load_model import *



def main():
    import pickle

    num_rerun = 10
    # img_indices = range(10,15)
    img_indices = range(2000,2005)


    train_stats = pickle.load( open('train_stats_tmp.pkl', "rb" ))
    files = []
    for i, v in enumerate(img_indices):
        for sd in range(num_rerun):
            stat = train_stats[ i * num_rerun + sd]
            data = [st['loss'] for st in stat['epoch_stats']]

            fname = plot1( img_indices[i], sd,data)

            files.append(fname)

    merge_plots(files)

def plot1(img_idx,sd,data):

    plt.clf()
    plt.plot(range(len(data)),data)
    plt.title("img_idx %d sd %d"%(img_idx,sd ))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.ylim((0, 30))

    filename = "./plots/img_idx_%d_sd_%d_loss.eps"%(img_idx,sd )
    plt.savefig(filename, dpi=1000)

    return filename


def merge_plots(files):


    import subprocess
    cmd = ["gs","-sDEVICE=pdfwrite", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dEPSCrop", "-sOutputFile=merged.pdf"]
    cmd = cmd + files
    subprocess.call(cmd)



if __name__ == "__main__":
    import sys
    generate_dirs(['./plots'])
    start_time = time.time()
    main()
    duration = (time.time() - start_time)

    print("---Program Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))



