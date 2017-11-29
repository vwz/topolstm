'''
Author: Jia Wang
'''
import numpy as np
import networkx as nx
import theano
# from theano import tensor
from theano import config
from collections import OrderedDict
import timeit
import six.moves.cPickle as pickle
import downhill
import metrics
# import pdb
import pprint

import data_utils
import tprnn_model


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def init_params(options):
    """
    Initializes values of shared variables.
    """
    params = OrderedDict()

    # word embedding, shape = (n_words, dim_proj)
    randn = np.random.randn(options['n_words'],
                            options['dim_proj'])
    params['Wemb'] = (0.1 * randn).astype(config.floatX)

    # shape = dim_proj * (4*dim_proj)
    lstm_W = np.concatenate([ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj'])], axis=1)
    params['lstm_W'] = lstm_W

    # shape = dim_proj * (4*dim_proj)
    lstm_U = np.concatenate([ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj'])], axis=1)
    params['lstm_U'] = lstm_U

    lstm_b = np.zeros((4 * options['dim_proj'],))
    params['lstm_b'] = lstm_b.astype(config.floatX)

    # decoding matrix for external influences
    randn = np.random.randn(options['dim_proj'],
                            options['n_words'])
    params['W_ext'] = (0.1 * randn).astype(config.floatX)
    dec_b = np.zeros(options['n_words'])
    params['b_ext'] = dec_b.astype(config.floatX)

    return params


def init_tparams(params):
    '''
    Set up Theano shared variables.
    '''
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def evaluate(f_prob, test_loader, k_list=[10, 50, 100]):
    '''
    Evaluates trained model.
    '''
    n_batches = len(test_loader)
    y = None
    y_prob = None
    for _ in range(n_batches):
        batch_data = test_loader()
        y_ = batch_data[-1]
        y_prob_ = f_prob(*batch_data[:-1])

        # excludes activated nodes when predicting.
        for i, p in enumerate(y_prob_):
            length = int(np.sum(batch_data[1][:, i]))
            sequence = batch_data[0][: length, i]
            assert y_[i] not in sequence, str(sequence) + str(y_[i])
            p[sequence] = 0.
            y_prob_[i, :] = p / float(np.sum(p))

        if y_prob is None:
            y_prob = y_prob_
            y = y_
        else:
            y = np.concatenate((y, y_), axis=0)
            y_prob = np.concatenate((y_prob, y_prob_), axis=0)

    return metrics.portfolio(y_prob, y, k_list=k_list)


def train(data_dir='data/memes/',
          dim_proj=512,
          maxlen=30,
          batch_size=256,
          keep_ratio=1.,
          shuffle_data=True,
          learning_rate=0.001,
          global_steps=50000,
          disp_freq=100,
          save_freq=1000,
          test_freq=1000,
          saveto_file='params.npz',
          weight_decay=0.0005,
          reload_model=False,
          train=True):
    """
    Topo-LSTM model training.
    """
    options = locals().copy()
    saveto = data_dir + saveto_file

    # loads graph
    G, node_index = data_utils.load_graph(data_dir)
    print nx.info(G)
    options['n_words'] = len(node_index)

    print options

    # creates and initializes shared variables.
    print 'Initializing variables...'
    params = init_params(options)
    if reload_model:
        print 'reusing saved model.'
        load_params(saveto, params)
    tparams = init_tparams(params)

    # builds Topo-LSTM model
    print 'Building model...'
    model = tprnn_model.build_model(tparams, options)

    print 'Loading test data...'
    test_examples = data_utils.load_examples(data_dir,
                                             dataset='test',
                                             node_index=node_index,
                                             maxlen=maxlen,
                                             G=G)
    test_loader = data_utils.Loader(test_examples, options=options)
    print 'Loaded %d test examples' % len(test_examples)

    if train:
        # prepares training data.
        print 'Loading train data...'
        train_examples = data_utils.load_examples(data_dir,
                                                  dataset='train',
                                                  keep_ratio=options[
                                                      'keep_ratio'],
                                                  node_index=node_index,
                                                  maxlen=maxlen,
                                                  G=G)
        train_loader = data_utils.Loader(train_examples, options=options)
        print 'Loaded %d training examples.' % len(train_examples)

        # compiles updates.
        optimizer = downhill.build(algo='adam',
                                   loss=model['cost'],
                                   params=tparams.values(),
                                   inputs=model['data'])

        updates = optimizer.get_updates(max_gradient_elem=5.,
                                        learning_rate=learning_rate)

        f_update = theano.function(model['data'],
                                   model['cost'],
                                   updates=list(updates))

        # training loop.
        start_time = timeit.default_timer()

        # downhill.minimize(
        #     loss=cost,
        #     algo='adam',
        #     train=train_loader,
        #     # inputs=input_list + [labels],
        #     # params=tparams.values(),
        #     # patience=0,
        #     max_gradient_clip=1,
        #     # max_gradient_norm=1,
        #     learning_rate=learning_rate,
        #     monitors=[('cost', cost)],
        #     monitor_gradients=False)

        n_examples = len(train_examples)
        batches_per_epoch = n_examples // options['batch_size'] + 1
        n_epochs = global_steps // batches_per_epoch + 1

        global_step = 0
        cost_history = []
        for _ in range(n_epochs):
            for _ in range(batches_per_epoch):
                cost = f_update(*train_loader())
                cost_history += [cost]

                if global_step % disp_freq == 0:
                    print 'global step %d, cost: %f' % (global_step, cost)

                # dump model parameters.
                if global_step % save_freq == 0:
                    params = unzip(tparams)
                    np.savez(saveto, **params)
                    pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)

                # evaluate on test data.
                if global_step % test_freq == 0:
                    scores = evaluate(model['f_prob'], test_loader)
                    print 'eval scores: ', scores
                    end_time = timeit.default_timer()
                    print 'time used: %d seconds.' % (end_time - start_time)

                global_step += 1

    scores = evaluate(model['f_prob'], test_loader)
    pprint.pprint(scores)


if __name__ == '__main__':
    train(data_dir='data/twitter', dim_proj=512, keep_ratio=1.)
