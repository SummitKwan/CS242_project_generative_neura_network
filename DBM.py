from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import datetime
import copy

import numpy as np
import scipy as sp
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

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
        l = np.random.randint(0, N+1)
        plt.subplot(Nr, Nc, i+1)
        plt.imshow(X[l, :].reshape([14, 14]), cmap='Greys', interpolation='none')
        plt.title(y[l])
        plt.axis('off')

def draw_W(W, image_size=[14,14], clim_scale=1.0, N_plot=None):
    if N_plot is None:
        N_plot = W.shape[1]
    [Nr, Nc] = rc_plot(N_plot)
    for i in range(N_plot):
        plt.subplot(Nr, Nc, i+1)
        plt.imshow(W[:, i].reshape([14, 14]),  cmap='coolwarm', interpolation='none')
        plt.clim(-clim_scale, clim_scale)
        # plt.title(i)
        plt.axis('off')

def sample_bern(p):
    return ( np.random.rand(*p.shape)<p ).astype(float)


class DBM(object):
    def __init__(self, widths, L=1, r_clone=None, masks_h=None, masks=None, Xs_ini=None, bs_ini=None, Ws_ini=None):
        """
        Initialize network
        :param L_width: Layer width, i.e. number of units in every layer, e.g., [50, 100, 10]
        """
        # number of data points
        self.L = L
        # number of layers/hierarchies
        self.H = len(widths)
        # number of node at every layer e.g., a list [50,100,10])
        self.widths = tuple(widths)
        # shape of weight matrix at every neighboring layer, e.g. a list [ (100,50), (10,100) ]
        self.shape_Ws = tuple(zip(widths[1:], widths[:-1] ))

        # number of clones of the nodes in this layer, like importance score
        if r_clone is None:
            self.r_clone = np.ones(self.H)
        else:
            self.r_clone = r_clone

        # Node value, binary values of every layer
        if Xs_ini is None:
            self.Xs = [ sample_bern( np.random.rand(L, width_h) ) for width_h in widths]
        else:
            self.Xs = Xs_ini

        # tf (true/false) mask, whether the layer is hidden (1) or visible (0), or mixed (2)
        if masks_h is None:
            self.masks_h = tuple([0]*self.H)
        else:
            self.masks_h = masks_h

        # tf (true/false) mask, whether the node is hidden (1) or visible (0)
        # Note: this is only used when mask_h==2 for a layer
        if masks is None:
            self.masks = tuple([ self.masks_h[h] * np.ones(self.widths[h]) for h in range(self.H) ])
        else:
            self.masks = masks

        # bias term value, continuous value of every layer
        if bs_ini is None:
            self.bs = [sample_bern(np.random.randn(width_h))/1000 for width_h in widths]
        else:
            self.bs = bs_ini

        # Weight value, continuous values of every neighboring layer pairs, [width_out * width_in]
        if Ws_ini is None:
            self.Ws = [np.random.randn(*shape_W)/1000 for shape_W in self.shape_Ws]
        else:
            self.Ws = Ws_ini

    def cal_E(self):
        """
        calculate energy of Boltzmann model
        :return: E(X,b,W)= - sum_i( b_i^T * X_i  +  X_{i+1}^T * W * X_i )
        """
        E = np.zeros( self.L )
        for h in range(self.H):
            E = E- np.dot(self.Xs[h], self.bs[h]) * self.r_clone[h]
        for h in range(self.H-1):
            E = E-  np.sum(np.dot(self.Xs[h+1], self.Ws[h]) *  self.Xs[h], axis=1) * (self.r_clone[h]*self.r_clone[h+1])
        return E


    def cal_p_cdtn(self, h):
        """
        p(X_h=1 | X_(h-1), X(h+1) )
        conditional probability of the response of one layer conditioned on its neighboring layers
        :param h: layer index
        :return:
        """
        # calculate  ( W * X_neighbor )
        if h == 0:
            WX = np.dot(self.Xs[h+1], self.Ws[h]) * self.r_clone[h+1]
        elif h == self.H -1:
            WX = np.dot(self.Xs[h-1], self.Ws[h-1].T) * self.r_clone[h-1]
        else:
            WX = np.dot(self.Xs[h-1], self.Ws[h-1].T) * self.r_clone[h-1] + np.dot(self.Xs[h+1], self.Ws[h]) * self.r_clone[h+1]
        return sigmoid( WX + self.bs[h] )


    def cal_psudo_llh(self):
        """
        calculate psudo-log-likelihood
        :return:  psudo log-likelihood of the network: mean(log (p(X_h|X_{h-1}, X_{h+1}) ) )
        """
        sum_psedu_llh = 0
        node_count    = 0
        for h in range(self.H):
            sum_psedu_llh =  sum_psedu_llh + np.sum( np.log(np.abs( self.cal_p_cdtn(h) - (1-self.Xs[h]) ) ), axis=1) * self.r_clone[h]
            node_count    =  node_count    + self.widths[h] *  self.r_clone[h]
        return sum_psedu_llh/node_count



    def sample_layer(self, h, ignore_mask=False, tf_keep_p=False, tf_inplace=True, r_sample=1.0):
        """
        sample the response of one single layer h: X_h
        :param h: index of layer
        :return: Xh: a binary array, response of one layer
        """
        if ignore_mask == True:     ### assumes every node is visible
            ph_cdtn = self.cal_p_cdtn(h)                   # p( X_h=1 | X_(h-1), X(h+1) )
            if tf_keep_p:
                Xh = ph_cdtn
            else:
                Xh = sample_bern(ph_cdtn)              # sample of X_h
        else:                       ### use mask_h and mask to determine visible/hidden
            if self.masks_h[h] == 0:            ## if visible, do nothing
                Xh = self.Xs[h]
            elif self.masks_h[h] == 1:          ## if hidden, sample based on neighboring layers
                ph_cdtn = self.cal_p_cdtn(h)               # p( X_h=1 | X_(h-1), X(h+1) )
                if tf_keep_p:
                    Xh = ph_cdtn
                else:
                    Xh = sample_bern(ph_cdtn)          # sample of X_h
            elif self.masks_h[h] == 2:          ## if mixed, use mask to determine
                ph_cdtn = self.cal_p_cdtn(h)
                if tf_keep_p:
                    Xh_new = sample_bern(ph_cdtn)
                else:
                    Xh_new = ph_cdtn
                Xh = self.Xs[h]*(1-self.masks[h]) + Xh_new* self.masks[h]

        if r_sample != 1.0:
            mask_sample = np.random.rand(*(Xh.shape))
            Xh = Xh*mask_sample + self.Xs[h]*(1.0-mask_sample)

        if tf_inplace == True:
            self.Xs[h] = Xh

        return Xh


    def sample_layers(self, h_order=None, ignore_mask=False, tf_keep_p=False, r_sample=1.0):
        """
        sample the response of all layers of the network
        :param h_order: the order of sampling, if None, use randome orders
        :param h_given: the layers that is given/visible
        :return:
        """
        if h_order is None:       # random schedule
            h_order = np.random.permutation(self.H)
        if h_order is 'up_down':  # ordered schedule, simultaneous bottom-up and top down, e.g. [0,4, 1,3, 2,2, 3,1, 4,0]
            h_order = sum([[i, self.H-1-i] for i in range(self.H)], [])
        # sample layer-by-layer
        for h in h_order:
            self.sample_layer(h, ignore_mask=ignore_mask, tf_keep_p=tf_keep_p, r_sample=r_sample)
        return self.Xs


    def cal_suff_stat(self):
        """
        calculate the sufficient statistics of the model
        :return:
        """
        # first order stat: bias:
        theta_b = [ np.mean(self.Xs[h], axis=0)               for h in range(self.H)   ]
        theta_W = [ np.dot(self.Xs[h+1].T, self.Xs[h])/self.L for h in range(self.H-1) ]
        theta = theta_b + theta_W
        return theta


    def theta_grad_CD(self, num_pos=1, num_neg=3, method_pos='gibbs'):
        """
        Calculate the gradient of model parameters theta=[b,W], using the contrastive divergence algorithm
        :param h_order: number of Gibbs iterations for the positive phase (samples conditioned on observation)
        :param h_order: number of Gibbs iterations for the negative phase (samples unconditioned)
        :return:
        """
        # sample for positive phase and calculate suff_stat
        for i in range(num_pos):
            if method_pos is 'gibbs':
                self.sample_layers(h_order='up_down', ignore_mask=False, tf_keep_p=False)
            else:
                self.sample_layers(h_order='up_down', ignore_mask=False, tf_keep_p=True)
        suff_stat_pos = self.cal_suff_stat()

        # sample for negative phase and calculate suff_stat
        for i in range(num_neg):
            self.sample_layers(h_order='up_down', ignore_mask=True)
        suff_stat_neg = self.cal_suff_stat()

        # gradient
        theta_grad = [ suff_stat_pos[i]-suff_stat_neg[i] for i in range(len(suff_stat_pos))]
        return theta_grad


    def update_theta(self, theta_grad, lambda_grad=0.1, labmda_b=0.01, labmda_W=0.01):
        """
        Update model parameters theta=[b,W]
        :return:
        """
        self.bs = [self.bs[h]*(1-labmda_b) + lambda_grad*theta_grad[h] for h in range(self.H)]
        self.Ws = [self.Ws[h]*(1-labmda_W) + lambda_grad*theta_grad[h+self.H] for h in range(self.H-1)]



def mlp_eval(mlp, X, Y=None, tf_keep_p=False):
    """
    Evaluate the response of multi-layer perceptron
    :param mlp:
    :param X: input
    :param Y: output, if none, evalute the Y using network
    :param tf_keep_p: if use continuouse probability 'p' instead of binary variables '~Bernoulli(p)'
    :return: Xs: response at all layers
    """
    Xs = []
    Xs.append(X)
    for h in range(1,mlp.n_layers_):
        Xh = mlp.intercepts_[h-1] + np.dot( Xs[h-1], mlp.coefs_[h-1] )
        Xh = sigmoid(Xh)
        if h != mlp.n_layers_-1:
            if tf_keep_p:
                pass
            else:
                Xh = sample_bern(Xh)
        else:
            if Y is None:
                Xh = (Xh==Xh.max(axis=1, keepdims=True)).astype('float')
            else:
                Xh = Y
        Xs.append(Xh)
    return Xs


""" ============================== """

""" get dataset and process them """
tf_load_data = True
if tf_load_data:

    dataset='./data/mnist.pkl.gz'
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    [_,y]=train_set
    [X,Y]=process_data(train_set)
    [N,M] = X.shape
    [_,K] = Y.shape


""" train DBM parameters """
size_minibatch = 20
num_minibatch = X.shape[0] / size_minibatch
M = [14*14, 64]   # width of hidden layers

""" parameters to initialize DBM """
widths = copy.copy(M)
widths.insert(0, X.shape[1])
widths.append(Y.shape[1])
H = len(widths)
masks_h = [1]*H
masks_h[0] = 0
masks_h[-1] = 0
masks_h_predict = copy.copy(masks_h)
masks_h_predict[-1] = 1
# r_clone = [1.0*widths[0]/r for r in widths]
r_clone = [1.0, 1.0, 2.0, 10.0]

""" creaat dbm object """
dbm = DBM(widths, L=size_minibatch, masks_h=masks_h, r_clone=r_clone)
dbm_predict = DBM(widths, L=size_minibatch, masks_h=masks_h_predict, r_clone=r_clone)


""" conjugate multi-layer-perceptron """
tf_use_mlp = True
# tf_use_mlp = False

if tf_use_mlp:
    mlp = MLPClassifier(hidden_layer_sizes=(M), activation='logistic', max_iter=40, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    print('training mlp')
    mlp.fit(X, y)
    Xs_mlp = mlp_eval(mlp, X, tf_keep_p=False)
    print('mlp score: {}'.format(mlp.score(X, y)))
    dbm.masks_h = [0]*H

count = 0
E = []
psudo_llh = []
r_err = []
itrt_c = []
itrt_r_err = []
plt.ioff()
for idx_epoch in range(2):
    for indx_minibatch in range(num_minibatch):
        itrt_c.append(count)
        """ current mini-batch """
        indx_data = range(indx_minibatch * size_minibatch, (indx_minibatch + 1) * size_minibatch)

        if tf_use_mlp:
            Xs_cur = [ Xs_mlp[h][indx_data, :] for h in range(dbm.H) ]
        else:
            Xs_cur = [None] * H
            for h in range(H):
                if h == 0:
                    Xs_cur[h] = X[indx_data, :]
                elif h == H-1:
                    Xs_cur[h] = Y[indx_data, :]
                else:
                    Xs_cur[h] = np.zeros( [size_minibatch, widths[h]] )


        """ train """
        dbm.Xs = copy.copy(Xs_cur)
        # dbm.Xs = [Xs_mlp[0][indx_data, :], Xs_mlp[1][indx_data, :], Y[indx_data, :]]
        dbm.update_theta(dbm.theta_grad_CD(method_pos='gibbs'), lambda_grad=0.01, labmda_b=0.01, labmda_W=0.0005)

        """ training snapshot: W """
        if indx_minibatch % 500 == 0:
            print(indx_minibatch)
            draw_W(dbm.Ws[0].T, image_size=[14, 14], clim_scale=0.5)
            plt.savefig('./temp_figs/temp_{}'.format(count))

        """ training snapshot: E, psudo_llh """
        if indx_minibatch % 50 == 0:
            Xs_cur_predict = copy.copy(Xs_cur)
            for h in range(1, dbm_predict.H):
                Xs_cur_predict[h] = Xs_cur_predict[h] * 0
            dbm_predict.Xs = Xs_cur_predict
            dbm_predict.bs = dbm.bs
            dbm_predict.Ws = dbm.Ws
            num_gibbs = 40
            Y_hat_gibbs = np.zeros([size_minibatch, Y.shape[1], num_gibbs])
            for i in range(num_gibbs):
                for j in range(10):
                    dbm_predict.sample_layers(h_order='up_down')
                Y_hat_gibbs[:, :, i] = dbm_predict.cal_p_cdtn(h=dbm_predict.H-1)
            Y_hat = np.mean(Y_hat_gibbs[:, :, num_gibbs/2:num_gibbs], axis=2)
            y_hat = np.argmax(Y_hat + np.random.rand(*Y_hat.shape) / 10 ** 6, axis=1)
            r_err.append(np.mean(y_hat != y[indx_data]))
            itrt_r_err.append(count)

        E.append(np.mean(dbm.cal_E()))
        psudo_llh.append(np.mean(dbm.cal_psudo_llh()))
        count = count + 1

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(itrt_c,E)
plt.xlabel('training iteration')
plt.title('energy E(X)')
plt.subplot(2, 2, 3)
plt.plot(itrt_c,psudo_llh)
plt.xlabel('training iteration')
plt.title('psudo-log-likelihood log(p~(X))')
plt.subplot(2, 2, 2)
plt.plot(itrt_r_err, r_err)
plt.ylim([0, 1])
plt.xlabel('training iteration')
plt.title('classification error rate')
plt.tight_layout()
plt.savefig('./temp_figs/train_error')

""" draw samples """
masks_h_sample = [1]*H
masks_h_sample[-1] = 0
plt.figure()
for label in range(10):
    print('sampleing {}'.format(label))
    dbm_sample = DBM(dbm.widths, L=1, masks_h=masks_h_sample, Xs_ini=None, bs_ini=dbm.bs, Ws_ini=dbm.Ws, r_clone=r_clone)
    dbm_sample.Xs = [dbm_sample.Xs[h] * 0 for h in range(dbm.H)]
    dbm_sample.Xs[-1] = gen_label_array([label], K=10)

    N_sample = 10
    N_gap = 20
    samples = np.zeros([N_sample, X.shape[1]])
    for i_sample in range(N_sample):
        samples[i_sample, :] = dbm_sample.sample_layer(h=0, tf_keep_p=True, tf_inplace=False)
        plt.subplot2grid([10,N_sample], [label,i_sample])
        plt.imshow( np.reshape(samples[i_sample,:], [14,14]), cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.axis('off')
        for i_gap in range(N_gap):
            dbm_sample.sample_layers(r_sample=1.0)
plt.savefig('./temp_figs/sample')
# plt.close()

plt.ion()

""" ===== below are old codes ====="""

if False:
    draw_processed_data_samples(X,Y,y)

if False:
    dbm = DBM([2, 4, 1], L=5)
    # print(dbm.Ws)

if False:
    size_minibatch = 100
    dbm = DBM([X.shape[1], Y.shape[1]], L=size_minibatch)
    E = []
    r_err = []
    for indx_minibatch in range(500):
        if indx_minibatch%10==0:
            print(indx_minibatch)
        indx_data = range(indx_minibatch * size_minibatch, (indx_minibatch+1) * size_minibatch)
        dbm.Xs = [ X[indx_data, :], Y[indx_data, :] ]
        dbm.update_theta(dbm.theta_grad_CD(), labmda_b=0.01, labmda_W=0.001)


        dbm_predict = copy.deepcopy(dbm)
        dbm_predict.masks_h = [0, 1]
        dbm_predict.Xs = [ X[indx_data, :], dbm.Xs[1] ]
        num_gibbs = 10
        Y_hat_gibbs = np.zeros([size_minibatch, Y.shape[1], num_gibbs])
        for i in range(num_gibbs):
            Y_hat_gibbs[:, :, i] = dbm_predict.sample_layers()[-1]
        Y_hat = np.mean(Y_hat_gibbs, axis=2)
        y_hat = np.argmax(Y_hat + np.random.rand(*Y_hat.shape)/10**6, axis=1)
        r_err.append(np.mean(y_hat != y[indx_data]))

        E.append(np.mean(dbm.cal_E()))
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(E)
    plt.subplot(1, 2, 2)
    plt.plot(r_err)
    plt.figure()
    draw_W(dbm.Ws[0].T, image_size=[14, 14], clim_scale=1.0)


if False:
    size_minibatch = 100
    dbm = DBM([X.shape[1], 10, Y.shape[1]], masks_h=[0,1,0], L=size_minibatch)
    E=[]
    r_err = []
    for idx_epoch  in range(5):
        for indx_minibatch in range(500):
            if indx_minibatch%100==0:
                plt.figure()
                draw_W(dbm.Ws[0].T, image_size=[14, 14], clim_scale=0.1)
                print(indx_minibatch)
            indx_data = range(indx_minibatch * size_minibatch, (indx_minibatch+1) * size_minibatch)
            dbm.Xs = [ X[indx_data, :], dbm.Xs[1], Y[indx_data, :] ]
            dbm.update_theta(dbm.theta_grad_CD(num_pos=1,num_neg=3), lambda_grad=0.001, labmda_b=0.0001, labmda_W=0.0001)

            # dbm_predict = copy.deepcopy(dbm)
            # dbm_predict.masks_h = [0, 1, 1]
            # dbm_predict.Xs = [ X[indx_data, :], dbm.Xs[1], dbm.Xs[2] ]
            # num_gibbs = 20
            # Y_hat_gibbs = np.zeros( [size_minibatch, Y.shape[1], num_gibbs] )
            # for i in range(num_gibbs):
            #     Y_hat_gibbs[:,:,i] = dbm_predict.sample_layers(h_order='up_down')[-1]
            # Y_hat = np.mean(Y_hat_gibbs, axis=2)
            # y_hat = np.argmax( Y_hat + np.random.rand( *Y_hat.shape )/10**6,  axis=1 )
            # r_err.append(np.mean(y_hat != y[indx_data]))

            E.append(np.mean(dbm.cal_E()))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(E)
    plt.subplot(1, 2, 2)
    plt.plot(r_err)
    plt.figure()
    draw_W(dbm.Ws[0].T, image_size=[14, 14], clim_scale=0.1)


# % three layer
if False:
    size_minibatch = 20
    num_minibatch  = X.shape[0]/size_minibatch
    M = 14 * 14
    # dbm = DBM([X.shape[1], Y.shape[1]], L=size_minibatch, masks_h=[0,1])
    # dbm = DBM([X.shape[1], M], L=size_minibatch, masks_h=[0, 1])
    dbm = DBM([X.shape[1], M, Y.shape[1]], L=size_minibatch, masks_h=[0,1,0], r_clone=[1,1,10])
    # dbm = DBM([X.shape[1], M, Y.shape[1]], L=size_minibatch, masks_h=[0, 0, 0], r_clone=[1, 1, 10])
    dbm_predict = DBM([X.shape[1], M, Y.shape[1]], L=size_minibatch, masks_h=[0,1,1], r_clone=[1,1,10])
    count = 0
    E=[]
    r_err = []
    for idx_epoch  in range(3):
        for indx_minibatch in range(num_minibatch):
            count = count + 1

            indx_data = range(indx_minibatch * size_minibatch, (indx_minibatch+1) * size_minibatch)
            # dbm.Xs = [ X[indx_data, :],  np.zeros([size_minibatch, M])]
            dbm.Xs = [ X[indx_data, :],  np.zeros([size_minibatch, M]), Y[indx_data, :] ]
            # dbm.Xs = [Xs_mlp[0][indx_data, :], Xs_mlp[1][indx_data, :], Y[indx_data, :]]
            dbm.update_theta(dbm.theta_grad_CD(method_pos='gibbs'), lambda_grad=0.01, labmda_b=0.01, labmda_W=0.0005)

            if indx_minibatch%500==0:
                print(indx_minibatch)
                draw_W(dbm.Ws[0].T, image_size=[14, 14], clim_scale=0.5)
                plt.savefig('./temp_figs/temp_{}'.format(count))

            if indx_minibatch%50==0:
                dbm_predict.Xs = [ X[indx_data, : ],  np.zeros([size_minibatch, M]), np.zeros([size_minibatch, 10]) ]
                dbm_predict.bs = dbm.bs
                dbm_predict.Ws = dbm.Ws
                num_gibbs = 40
                Y_hat_gibbs = np.zeros( [size_minibatch, Y.shape[1], num_gibbs] )
                for i in range(num_gibbs):
                    Y_hat_gibbs[:,:,i] = dbm_predict.sample_layers()[-1]
                Y_hat = np.mean(Y_hat_gibbs[:,:,num_gibbs/2:num_gibbs], axis=2)
                y_hat = np.argmax( Y_hat + np.random.rand( *Y_hat.shape )/10**6,  axis=1 )
                r_err.append(np.mean(y_hat != y[indx_data]))

            # E.append(np.mean(dbm.cal_E()))
            E.append(np.mean(dbm.cal_psudo_llh()))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(E)
    plt.subplot(1, 2, 2)
    plt.plot(r_err)
    plt.ylabel([0,1])
    plt.savefig('./temp_figs/train_error')


# % four layer
if False:
    size_minibatch = 20
    num_minibatch  = X.shape[0]/size_minibatch
    M = 14 * 14
    M1 = 64
    # dbm = DBM([X.shape[1], Y.shape[1]], L=size_minibatch, masks_h=[0,1])
    # dbm = DBM([X.shape[1], M], L=size_minibatch, masks_h=[0, 1])
    dbm =         DBM([X.shape[1], M, M1, Y.shape[1]], L=size_minibatch, masks_h=[0,1,1,0], r_clone=[1,1,2,10])
    dbm_predict = DBM([X.shape[1], M, M1, Y.shape[1]], L=size_minibatch, masks_h=[0,1,1,1], r_clone=[1,1,2,10])
    count = 0
    E=[]
    r_err = []
    for idx_epoch  in range(3):
        for indx_minibatch in range(num_minibatch):
            count = count + 1

            indx_data = range(indx_minibatch * size_minibatch, (indx_minibatch+1) * size_minibatch)
            # dbm.Xs = [ X[indx_data, :],  np.zeros([size_minibatch, M])]
            dbm.Xs = [ X[indx_data, :],  np.zeros([size_minibatch, M]), np.zeros([size_minibatch, M1]), Y[indx_data, :] ]
            dbm.update_theta(dbm.theta_grad_CD(method_pos='gibbs'), lambda_grad=0.01, labmda_b=0.01, labmda_W=0.0005)

            if indx_minibatch%500==0:
                print(indx_minibatch)
                draw_W(dbm.Ws[0].T, image_size=[14, 14], clim_scale=0.5)
                plt.savefig('./temp_figs/temp_{}'.format(count))

            if indx_minibatch%50==0:
                dbm_predict.Xs = [ X[indx_data, : ],  np.zeros([size_minibatch, M]), np.zeros([size_minibatch, M1]), np.zeros([size_minibatch, 10]) ]
                dbm_predict.bs = dbm.bs
                dbm_predict.Ws = dbm.Ws
                num_gibbs = 20
                Y_hat_gibbs = np.zeros( [size_minibatch, Y.shape[1], num_gibbs] )
                for i in range(num_gibbs):
                    Y_hat_gibbs[:,:,i] = dbm_predict.sample_layers(h_order='up_down')[-1]
                Y_hat = np.mean(Y_hat_gibbs, axis=2)
                y_hat = np.argmax( Y_hat + np.random.rand( *Y_hat.shape )/10**6,  axis=1 )
                r_err.append(np.mean(y_hat != y[indx_data]))

            E.append(np.mean(dbm.cal_E()))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(E)
    plt.subplot(1, 2, 2)
    plt.plot(r_err)
    plt.savefig('./temp_figs/train_error')
    # plt.figure()
    # draw_W(dbm.Ws[0].T, image_size=[14, 14], clim_scale=0.5)


if False:
    # indx_minibatch=2
    # indx_data = range(indx_minibatch * size_minibatch, (indx_minibatch + 1) * size_minibatch)
    # dbm.Xs = [X[indx_data, :], np.zeros([size_minibatch, M])]
    dbm_sample = DBM(dbm.widths, L=1, masks_h=[1,1], Xs_ini=None, bs_ini=dbm.bs, Ws_ini=dbm.Ws)
    N_sample = 100
    N_gap    = 5
    samples = np.zeros([N_sample, X.shape[1]])
    for i_sample in range(N_sample):
        samples[i_sample, :] = dbm_sample.Xs[0][0, :]
        for i_gap in range(N_gap):
            dbm_sample.sample_layers()
        dbm_sample.sample_layer(h=0, tf_keep_p=True)
    draw_W(samples.T)

if False:   # three layers
    for label in range(10):
        dbm_sample = DBM(dbm.widths, L=1, masks_h=[1,1,0], Xs_ini=None, bs_ini=dbm.bs, Ws_ini=dbm.Ws, r_clone=[1,1,10])
        dbm_sample.Xs[-1] = gen_label_array([label], K=10)
        dbm_sample.Xs[0] = dbm_sample.Xs[0]*0
        dbm_sample.Xs[1] = dbm_sample.Xs[1]*0
        N_sample = 100
        N_gap    = 20
        samples = np.zeros([N_sample, X.shape[1]])
        for i_sample in range(N_sample):
            samples[i_sample, :] = dbm_sample.sample_layer(h=0, tf_keep_p=True, tf_inplace=False)
            for i_gap in range(N_gap):
                dbm_sample.sample_layers(r_sample=1.0)
        draw_W(samples.T)
        plt.savefig('./temp_figs/sample_{}'.format(label))
        plt.close()

if False:  # four layers
    for label in range(10):
        dbm_sample = DBM(dbm.widths, L=1, masks_h=[1, 1, 1, 0], Xs_ini=None, bs_ini=dbm.bs, Ws_ini=dbm.Ws,
                         r_clone=[1, 1, 2, 10])
        dbm_sample.Xs[-1] = gen_label_array([label], K=10)
        dbm_sample.Xs[0] = dbm_sample.Xs[0] * 0
        dbm_sample.Xs[1] = dbm_sample.Xs[1] * 0
        dbm_sample.Xs[2] = dbm_sample.Xs[2] * 0
        N_sample = 100
        N_gap = 20
        samples = np.zeros([N_sample, X.shape[1]])
        for i_sample in range(N_sample):
            samples[i_sample, :] = dbm_sample.sample_layer(h=0, tf_keep_p=True, tf_inplace=False)
            for i_gap in range(N_gap):
                dbm_sample.sample_layers(r_sample=1.0)
        draw_W(samples.T)
        plt.savefig('./temp_figs/sample_{}'.format(label))
    plt.close()


# test Multi-layer perceptron
if False:
    mlp = MLPClassifier(hidden_layer_sizes=(14*14,), activation='logistic', max_iter=20, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    mlp.fit(X, y)
    draw_W(mlp.coefs_[0])
    Xs_mlp = mlp_eval(mlp, X, tf_keep_p=False)
    print( 1- np.mean(np.argmax(Xs_mlp[-1], axis=1) == y) )


# run DBM
