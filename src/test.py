import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# from tensorflow.examples.tutorials.mnist import input_data

from util import *
from model import *
from load_model import *

def main(RANK, WSIZE, dataset, eps):

    config = {
            "dataset":dataset, # 'train' or 'test'
            "max_epoch": 600,
            "lr": 0.1, # base lr : 0.1, 0.05
            "norm": "l2", # l2, linf
            "load_weight": "nat", # nat, adv, sec
            "eps": eps,
            "num_reruns": 10, # number of different seed run

            # optional
            "img_indices": None # None is run with all dataset
            # "img_indices": range(10,15)
        }
    ## overrides
    config["img_indices"]= range(2000,2100) # None is run with all dataset
    # config["img_indices"]= range(10,12) # None is run with all dataset


    print('Running EXP for:')
    print(config)

    AA = Trainer(config)

    with Timer(name = 'initialize') as t:
        AA.setup()

    with Timer(name = 'run_train') as t:
        AA.run_train_save(rank=RANK, wsize = WSIZE)

    # merging
    # if RANK ==0:
    #     with Timer(name = 'run_train') as t:
    #         AA.merge_results(wsize = WSIZE)


class Trainer(object):
    def __init__(self, config, seed = 0):
        self.seed = seed
        self.config = config

        self.model = MyModel(config=config)


        if config['dataset'] == 'train':
            self.dataset = MyDatasetTrain(config = config)
        elif config['dataset'] == 'test':
            self.dataset = MyDatasetTest(config = config)
        else:
            assert 0

    def setup(self):
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)

        self.model.setup()
        self.dataset.setup()

        # config = tf.ConfigProto()
        config = tf.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        # self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=0)

    def run_train_save(self, rank=0, wsize=1,verbose=1):

        if self.check_results_exists(rank, wsize) == (True, True):
            print('skipping experiment as results exists')
            return

        # run train for each img label
        # but start with just one first
        if self.config["img_indices"] is None:
            img_indices = range(self.dataset.num_data)
        else:
            img_indices = self.config["img_indices"]

        seeds = range(self.config['num_reruns'])

        img_indices = chunk(img_indices, rank ,wsize)
        self.model.load_nat_weight(self.sess, self.config['load_weight'])

        stats = []
        train_stats_tmp = []
        for idx, img_idx in enumerate(img_indices):
            # print(idx, img_idx)
            stats_seeds = []
            for sd in seeds:
                t0 = time.time()
                st = {}

                x, y = self.dataset.load_index(img_idx)
                # print(sd, 'eval')
                self.model.initialize_variable(self.sess, check_loss=(x,y), threshold = 0.0001)
                # self.model.initialize_variable(self.sess, check_loss=None)
                # print(sd, 'eval')

                st['loss_init'], st['corr_init'] = self.model.eval(self.sess, x, y)
                st1 = self.model.train(
                    self.sess,
                    x,
                    y,
                    self.config['lr'],
                    num_epoch = self.config['max_epoch'],
                    verbose = 0
                    )

                st.update(st1)

                stats_seeds.append(st)
                train_stats_tmp.append(st)

                t1 = time.time()
                if verbose:
                    print(" img %d sd %d init %.3f loss %.3f took %.3fsec (est total %.3fsec)"%(
                        img_idx,sd,st['loss_init'] ,st['loss'], (t1-t0), (t1-t0) * len(img_indices) * len(seeds) ))

            st = st_best = self._find_optimal_stat(stats_seeds, metric = 'loss', optimum='max')
            if verbose:
                print("img %d (%d/%d) best init %.3f loss %.3f"%(img_idx,idx,len(img_indices),st['loss_init'] ,st['loss']))

            st['exp_seed'] = stats_seeds

            stats.append(st_best)

        stats_zero = []

        self.model.set_v_zero(self.sess)
        for img_idx in img_indices:
            st = {}
            x, y = self.dataset.load_index(img_idx)
            st['loss'], st['corr'] = self.model.eval(
                self.sess,
                x,
                y
                )
            stats_zero.append(st)

        avg_loss = np.mean([st['loss'] for st in stats])
        print('avg_loss',avg_loss)

        import pickle
        str1 = "results/"+self.fname_prefix()

        pickle.dump( stats, open(str1+"stats_%dof%d.pkl"%(rank,wsize), "wb" ) )
        pickle.dump( stats_zero, open(str1+"stats_zero_%dof%d.pkl"%(rank,wsize), "wb" ) )
        print('saved results to ',str1+"stats_zero_%dof%d.pkl"%(rank,wsize))

        pickle.dump(train_stats_tmp, open('train_stats_tmp.pkl', "wb" ))
        print('saved train_stats')


    def check_results_exists(self, rank, wsize):
        str1 = "results/"+self.fname_prefix()
        fname1 = str1+"stats_%dof%d.pkl"%(rank,wsize)
        fname2 = str1+"stats_zero_%dof%d.pkl"%(rank,wsize)
        import os
        t1 = os.path.exists(fname1)
        t2 = os.path.exists(fname1)
        return t1, t2


    def merge_results(self, wsize):
        stats = []
        stats_zero = []

        def _load(str0, rank, wsize):
            import pickle
            str1 = "results/"+self.fname_prefix()
            return pickle.load( open( str1+"%s_%dof%d.pkl"%(str0,rank,wsize), "rb" ) )

        for rank in range(wsize):

            # print('reading %d /%d\r'%(rank, wsize), end='')

            stats += _load('stats', rank, wsize)
            stats_zero += _load('stats_zero', rank, wsize)

        avg_loss = np.mean([st['loss'] for st in stats])
        avg_loss_zero = np.mean([st['loss'] for st in stats_zero])
        avg_corrects = np.mean([st['corr'] for st in stats])
        avg_corrects_zero = np.mean([st['corr'] for st in stats_zero])

        # print([st['corr'] for st in stats])
        # print([st['corr'] for st in stats_zero])

        print('config:')
        print(self.config)

        print("trained avg_loss %f, avg_acc %f"%(avg_loss, avg_corrects ))
        print("zero avg_loss %f, avg_acc %f"%(avg_loss_zero, avg_corrects_zero ))

    def fname_prefix(self):
        c = self.config

        str0 = "%s_%s_%s_eps%.3f_lr_%.3f_me%d_nr%d" % \
            (
                c['load_weight'],
                c['norm'],
                c['dataset'],
                c['eps'],
                c['lr'],
                c['max_epoch'],
                c['num_reruns']
                )
        return str0

    def _find_optimal_stat(self,stats, metric = 'loss', optimum='min'):
        l1 = [st[metric] for st in stats]

        if optimum == 'max':
            idx = np.argmax(l1)
        elif optimum == 'min':
            idx = np.argmin(l1)
        else:
            assert 0

        return stats[idx]

class MyModel(object):
    def __init__(self, config=None, seed = 0):
        self.seed = seed
        self.config = config

    def setup(self):

        ##### layer sturucture ######
        self.x_pl = tf.placeholder(tf.float32, shape=(None, 28*28))
        self.y_pl = tf.placeholder(tf.int32, shape=(None,))

        self.weights_nat = weight_nat()

        self.v = weight_variable(stddev=1.0,shape = (1,28*28))


        self.input_image = self.x_pl + self.v

        self.output = model_nat(self.input_image, self.weights_nat)

        ##### losses ######

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = self.y_pl, logits = self.output))

        self.obj_func = - self.loss

        #### optimizer ####

        self.lr_pl = tf.placeholder(tf.float32, shape=())

        # self.momentum = self.config['momentum']
        self.momentum = 0.9

        # print('!!!Using Gradient descent optimizer!')
        # optimizer = tf.train.GradientDescentOptimizer(self.lr_pl)

        optimizer = tf.train.MomentumOptimizer(
            self.lr_pl,
            self.momentum,
            use_nesterov=True
            )
        print('!!!use use_nesterov')

        # optimizer = tf.train.AdamOptimizer()
        # print('!!!use AdamOptimizer')

        self.train_op = optimizer.minimize(self.obj_func, var_list=[self.v])

        if self.config['norm'] == 'l2':
            self.clip_op = tf.assign(self.v, tf.clip_by_norm(self.v, self.config["eps"]))
        elif self.config['norm'] == 'linf':
            v_normalized = self.v / tf.norm(self.v, ord = np.inf) * self.config["eps"]
            self.clip_op = tf.assign(self.v, v_normalized)
        else:
            assert 0

        self.initialize_op = tf.initializers.variables(var_list=[self.v], name='init')
        self.initialize_optimizer_op = tf.initializers.variables(var_list=optimizer.variables(), name='init_opt')

        self.zero_op = tf.assign(self.v, np.zeros(shape=(1,28*28)))

        ## metrics ##
        self.y_pred = tf.cast(tf.argmax(self.output, 1), tf.int32)
        self.correct_prediction = tf.equal(self.y_pred, self.y_pl)
        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def initialize_variable(self, sess, check_loss = None, threshold = 0.001, max_count = 100):
        verbose = 0
        if check_loss is None:
            sess.run([self.initialize_op, self.initialize_optimizer_op])
        else:
            (x,y) = check_loss
            # initialize until intial loss is more than 1.0
            loss = 0.0
            ct = 0
            while loss <= threshold and ct <max_count:
                sess.run([self.initialize_op, self.initialize_optimizer_op])

                sess.run([self.clip_op])

                loss , _ = self.eval(sess, x, y)
                if verbose: print('init_loss',loss)
                ct += 1

            if ct == max_count:
                print("initialize check_loss th %f failed"%threshold)

    def train(self, sess, x, y, learning_rate, num_epoch=1, verbose = 0):

        # verbose = 0

        ###### my training loop ########
        stats={}
        epoch_stats = []

        lr_decay = 1.0 # per epoch
        lr = learning_rate
        for epoch in range(num_epoch):
            st ={}
            feed_dict0 = {
                self.x_pl: x,
                self.y_pl: y,
                self.lr_pl: learning_rate}

            sess.run([self.train_op], feed_dict = feed_dict0)
            sess.run([self.clip_op], feed_dict = feed_dict0)

            st['loss'], st['corr'] = self.eval(sess, x, y)
            if verbose : print("epoch %d loss %f corr %d lr %.3f"%(epoch,st['loss'], st['corr'], lr))

            epoch_stats.append(st)

            lr = lr * lr_decay

        stats['loss'], stats['corr'] = self.eval(sess, x, y)

        # optional (turn off to save disk)
        stats['v'] = sess.run(self.v)
        stats['epoch_stats'] = epoch_stats

        return stats # v, loss, corr

    def eval(self, sess, x, y):
        feed_dict0 = {
            self.x_pl: x,
            self.y_pl: y}
        loss = sess.run(self.loss, feed_dict = feed_dict0)

        correct = sess.run(self.num_correct, feed_dict = feed_dict0) # will be 1 or 0 for single data

        return loss, correct

    def set_v_zero(self,sess):
        sess.run(self.zero_op)

    def load_nat_weight(self, sess, load_weight_from):
        fetch_checkpoint_hack(sess, self.weights_nat, model = load_weight_from)

class MyDatasetTrain(object):
    def __init__(self, config=None, seed = 0):
        self.seed = seed
        self.config = config

    def setup(self):
        from tensorflow.examples.tutorials.mnist import input_data
        data_sets = input_data.read_data_sets('MNIST_data', one_hot=False)
        self.xs = data_sets.train.images # (55000, 784)
        self.ys = data_sets.train.labels # (55000,)

        self.num_data = self.xs.shape[0]

    def load_index(self, idx):
        return self.xs[idx,:].reshape((-1,784)), np.array(self.ys[idx]).reshape((1))

class MyDatasetTest(object):
    def __init__(self, config=None, seed = 0):
        self.seed = seed
        self.config = config

    def setup(self):
        from tensorflow.examples.tutorials.mnist import input_data
        data_sets = input_data.read_data_sets('MNIST_data', one_hot=False)

        self.xs = data_sets.test.images # (10000, 784)
        self.ys = data_sets.test.labels # (10000,)

        self.num_data = self.xs.shape[0]

    def load_index(self, idx):
        return self.xs[idx,:].reshape((-1,784)), np.array(self.ys[idx]).reshape((1))


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



