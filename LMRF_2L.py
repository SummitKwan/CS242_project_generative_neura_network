from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import datetime

import numpy
import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano import function
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams

l_flag_model = ['bottom_up', 'top_down', 'energy']
flag_model = l_flag_model[0]

path_figs = './temp_figs/'


def image_downsample(X, r_sample=2, threshold=0.3):
    """
    Down sample the image
    :param X: np array [N*M], N items, M pixels (flattened)
    :param r_sample: ratio of dawn sampling, if 2, every other pixel
    :return: X_ds, down sampled, [N* (M/r_sample**2)]
    """
    [N, M] = X.shape
    index_linear = np.arange(M)
    X_ds = X[:, np.logical_and(index_linear%r_sample==1,
                               np.floor((index_linear/np.sqrt(M)))%r_sample==1 )]
    X_ds = 1.0*(X_ds>threshold)
    return X_ds

def gen_label_array(y, K=0):
    """
    Generate binary label array from label
    :param y: label, 1D int array [N,], eg. [0,1,2]
    :param K: number of unique labels
    :return: Y, 2D binary label array [N,K], eg. [ [1,0,0], [0,1,0], [0,0,1] ]
    """
    if K == 0:
        K = y.max()+1
    N = len(y)
    Y = np.zeros([N,K])
    Y[range(N), y] = 1
    return Y

def process_data(data_in):
    """
    down-sample image (X) and make label (y) a binary array
    :param data_in: [X,y]
    :return: [X,Y]
    """
    [X,y] = data_in
    X = image_downsample(X)
    Y = gen_label_array(y)
    return (X,Y)

def rc_plot(N_plot):
    """
    calculate the [n_rows, n_cols] for subplot
    :param N: number of pannels
    :return: [n_rows, n_cols]
    """
    Nc = np.ceil(np.sqrt(N_plot))
    Nr = np.ceil(N_plot / Nc)
    return [Nr, Nc]

def draw_processed_data_samples(X,Y,y):
    N_plot = 20
    [Nr, Nc] = rc_plot(N_plot)
    for i in range(N_plot):
        l = np.random.randint(0, N)
        plt.subplot(Nr, Nc, i)
        plt.imshow(X[l, :].reshape([14, 14]), cmap='Greys', interpolation='none')
        plt.title(y[l])
        plt.axis('off')

def draw_W(W, image_size=[14,14], clim_scale=1.0):
    N_plot = 10
    [Nr, Nc] = rc_plot(N_plot)
    for i in range(N_plot):
        plt.subplot(Nr, Nc, i+1)
        plt.imshow(W[:, i].reshape([14, 14]),  cmap='coolwarm', interpolation='none')
        plt.clim(-clim_scale, clim_scale)
        plt.title(i)
        plt.axis('off')


class FF_network(object):
    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros((n_out,),dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def cost(self, Y, lambda_reg=0.01):
        """cost function to optimize, L2 loss"""
        # return T.mean((self.p_y_given_x - Y) ** 2)
        # return T.mean((self.p_y_given_x-Y)**2) + lambda_reg* T.mean(self.W**2)
        return -T.mean(T.log(T.abs_(self.p_y_given_x - (1-Y))) ) + lambda_reg * T.mean(self.W ** 2)

    def errors(self, Y):
        y = T.argmax(Y, axis=1)
        return T.mean(T.neq(self.y_pred, y))



class MRF_network(object):
    def __init__(self, x0, x1, n0, n1):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.random.randn(n0, n1).astype(dtype=theano.config.floatX) /10,
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b0 = theano.shared(
            value=numpy.zeros((n0,),dtype=theano.config.floatX),
            name='b0',
            borrow=True
        )
        self.b1 = theano.shared(
            value=numpy.zeros((n1,),dtype=theano.config.floatX),
            name='b1',
            borrow=True
        )

        self.theano_rng = RandomStreams(0)

        self.p_x1_given_x0 = T.nnet.sigmoid(T.dot(x0, self.W) + self.b1)
        self.p_x0_given_x1 = T.nnet.sigmoid(T.dot(x1, self.W.T) + self.b0)

        # parameters of the model
        self.params = [self.b0, self.b1, self.W]

        # keep track of model input
        self.input = [x0, x1]

    def propup(self, x0):
        pre_sigmoid_activation = T.dot(x0, self.W) + self.b1
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def propdown(self, x1):
        pre_sigmoid_activation = T.dot(x1, self.W.T) + self.b0
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_x1_given_x0(self, x0_sample):
        pre_sigmoid_x1, x1_mean = self.propup(x0_sample)
        x1_sample = self.theano_rng.binomial(size=x1_mean.shape,
                                             n=1, p=x1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_x1, x1_mean, x1_sample]

    def sample_x0_given_x1(self, x1_sample):
        pre_sigmoid_x0, x0_mean = self.propdown(x1_sample)
        x0_sample = self.theano_rng.binomial(size=x0_mean.shape,
                                             n=1, p=x0_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_x0, x0_mean, x0_sample]

    def cost(self, Y, lambda_reg=0.01):
        """cost function to optimize, L2 loss"""
        # return T.mean((self.p_y_given_x - Y) ** 2)
        # return T.mean((self.p_y_given_x-Y)**2) + lambda_reg* T.mean(self.W**2)
        return -T.mean(T.log(T.abs_(self.p_y_given_x - (1-Y))) ) + lambda_reg * T.mean(self.W ** 2)

    def errors(self, Y):
        y = T.argmax(Y, axis=1)
        return T.mean(T.neq(self.y_pred, y))




""" ============================== """

""" get dataset and process them """
dataset='./data/mnist.pkl.gz'
with gzip.open(dataset, 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

[_,y]=train_set
[X,Y]=process_data(train_set)
[N,M] = X.shape
[_,K] = Y.shape

if False:
    draw_processed_data_samples(X,Y,y)


shared_x0 = theano.shared(numpy.asarray(X, dtype=theano.config.floatX), borrow=True)
shared_x1 = theano.shared(numpy.asarray(Y, dtype=theano.config.floatX), borrow=True)
shared_y1 = theano.shared(numpy.asarray(y, dtype=theano.config.floatX), borrow=True)


""" test theano function """
learning_rate=0.13
n_epochs = 10
batch_size = 100

index = T.lscalar()
x0 = T.matrix('x')  # data, presented as rasterized images
x1 = T.matrix('Y')  #
y1 = T.ivector('y')  # labels, presented as 1D vector of [int] labels


""" model and training """
if flag_model == l_flag_model[0]:
    model = FF_network(input=x0, n_in=14*14, n_out=10)
    cost = model.cost(x1)
    errors = model.errors(x1)
    g_W = T.grad(cost=cost, wrt=model.W)
    g_b = T.grad(cost=cost, wrt=model.b)
    updates = [(model.W, model.W - learning_rate * g_W),
               (model.b, model.b - learning_rate * g_b)]

    train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x0: shared_x0[index * batch_size: (index + 1) * batch_size, :],
                x1: shared_x1[index * batch_size: (index + 1) * batch_size, :]
            }
        )
    cal_errors = theano.function(
        inputs=[index],
        outputs=errors,
        givens={
            x0: shared_x0[index * batch_size: (index + 1) * batch_size, :],
            x1: shared_x1[index * batch_size: (index + 1) * batch_size, :]
        }
    )
    srng = RandomStreams(0)

    time_str = datetime.datetime.now().strftime('%Y-%m%d_%H%M%S')
    N_minibach = np.floor(N/batch_size).astype(int)
    N_ts_train = N_minibach*n_epochs
    cost_train = np.zeros(N_ts_train)
    errors_train = np.zeros(N_ts_train)
    index_ts_train = 0
    plt.ioff()
    for epoch_index in range(n_epochs):
        for minibatch_index in range(N_minibach):

            errors_cur = cal_errors(minibatch_index)
            cost_cur = train_model(minibatch_index)
            cost_train[index_ts_train] = cost_cur
            errors_train[index_ts_train] = errors_cur

            if True:
                if minibatch_index%100==0:
                    print(minibatch_index)
                    plt.figure()
                    W_cur = model.W.get_value()
                    draw_W(W_cur)
                    plt.savefig('{}{}_{}_{}_{}.png'.format(path_figs, flag_model, time_str,epoch_index, minibatch_index))
                    plt.close()
            # print(model.b.get_value())
            # print(model.W.get_value())
            # print(minibatch_index)
            index_ts_train = index_ts_train + 1
    plt.ion()
    plt.figure()
    plt.plot(range(N_ts_train), errors_train)
    plt.plot(range(N_ts_train), cost_train)
    plt.legend(['errors','cost'])
    plt.savefig('{}{}_{}_{}.png'.format(path_figs, flag_model, time_str, 'cost_error'))

elif flag_model == l_flag_model[1]:
    model = FF_network(input=x1, n_in=10, n_out=14*14)
    cost = model.cost(x0)
    errors = model.errors(x0)
    g_W = T.grad(cost=cost, wrt=model.W)
    g_b = T.grad(cost=cost, wrt=model.b)
    updates = [(model.W, model.W - learning_rate * g_W),
               (model.b, model.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x0: shared_x0[index * batch_size: (index + 1) * batch_size, :],
            x1: shared_x1[index * batch_size: (index + 1) * batch_size, :]
        }
    )
    cal_errors = theano.function(
        inputs=[index],
        outputs=errors,
        givens={
            x0: shared_x0[index * batch_size: (index + 1) * batch_size, :],
            x1: shared_x1[index * batch_size: (index + 1) * batch_size, :]
        }
    )
    srng = RandomStreams(0)

    time_str = datetime.datetime.now().strftime('%Y-%m%d_%H%M%S')
    N_minibach = np.floor(N/batch_size).astype(int)
    N_ts_train = N_minibach*n_epochs
    cost_train = np.zeros(N_ts_train)
    errors_train = np.zeros(N_ts_train)
    index_ts_train = 0
    plt.ioff()
    for epoch_index in range(n_epochs):
        for minibatch_index in range(N_minibach):

            errors_cur = cal_errors(minibatch_index)
            cost_cur = train_model(minibatch_index)
            cost_train[index_ts_train] = cost_cur
            errors_train[index_ts_train] = errors_cur

            if True:
                if minibatch_index%100==0:
                    print(minibatch_index)
                    plt.figure()
                    W_cur = model.W.get_value()
                    draw_W(W_cur.T, clim_scale=1.0/(14*14)*10)
                    plt.savefig('{}{}_{}_{}_{}.png'.format(path_figs, flag_model, time_str,epoch_index, minibatch_index))
                    plt.close()
            # print(model.b.get_value())
            # print(model.W.get_value())
            # print(minibatch_index)
            index_ts_train = index_ts_train + 1
    plt.ion()
    plt.figure()
    plt.plot(range(N_ts_train), cost_train)
    plt.legend(['cost'])
    plt.savefig('{}{}_{}_{}.png'.format(path_figs, flag_model, time_str, 'cost_error'))

elif flag_model == l_flag_model[2]:
    print('haha')
    model = MRF_network(x0=x0, x1=x1, n0=14*14, n1=10)
    temp = function([], model.p_x1_given_x0, givens={x0: shared_x0[0:4, :]})
    temp = function([], model.p_x0_given_x1, givens={x1: shared_x1[0:4, :]})
    model.sample_x1_given_x0(shared_x0[0:4, :])


# run LMRF_2L