#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing training adversarial neural network for de-correlated jet tagging."""

# Basic import(s)
import glob
import pickle
import logging as log
import itertools

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Scientific import(s)
import numpy as np
import pandas as pd
import root_numpy
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Project import(s)
from adversarial.utils     import *
from adversarial.profile   import *
from adversarial.constants import *
from .common import *


# Global variable(s)
RNG = np.random.RandomState(21)  # For reproducibility

def DisCo(y_true_tmp, y_pred, alpha = 0., power = 1, normedweight=1):
    from keras import backend as K
    from keras.losses import binary_crossentropy
    import tensorflow as tf

    """
    var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """
    print 'y_true_tmp = ',y_true_tmp
    y_true = tf.gather(y_true_tmp, [0], axis=1)
    X_in = tf.gather(y_true_tmp, [1], axis=1)
    Y_in = y_pred
    W_in = tf.gather(y_true_tmp, [2], axis=1)
    print 'X_in shape: ', X_in.get_shape()
    print 'Y_in shape: ', Y_in.get_shape()
    print 'W_in shape: ', W_in.get_shape()

    mymaskX = tf.where(y_true<1,K.ones_like(X_in, dtype=bool),K.zeros_like(X_in, dtype=bool))
    mymaskY = tf.where(y_true<1,K.ones_like(Y_in, dtype=bool),K.zeros_like(Y_in, dtype=bool))
    var_1 = tf.boolean_mask(X_in, mymaskX)
    var_2 = tf.boolean_mask(Y_in, mymaskY)
    #normedweight = tf.boolean_mask(W_in, mymaskX)
    print 'var_1 shape: ', var_1.get_shape()
    print 'var_2 shape: ', var_2.get_shape()
    #print 'normedweight shape: ', normedweight.get_shape()

    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
 
    yy = tf.transpose(xx)
    #amat = tf.math.abs(xx-yy)
    amat = tf.abs(xx-yy)
    
    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
    
    yy = tf.transpose(xx)
    #bmat = tf.math.abs(xx-yy)
    bmat = tf.abs(xx-yy)
   
    amatavg = tf.reduce_mean(amat*normedweight, axis=1)
    bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)
 
    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)

    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)

    ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)
   
    if power==1:
        #dCorr = tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
        dCorr = tf.reduce_mean(ABavg*normedweight)/tf.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    else:
        #dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power
        dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power

    return binary_crossentropy(y_true,y_pred) + alpha*dCorr


# Main function definition
@profile
def main (args):

    # Initialisation
    # --------------------------------------------------------------------------
    with Profile("Initialisation"):

    # Initialising
    # ----------------------------------------------------------------------
        args, cfg = initialise(args)

        # Validate train/optimise flags
        if args.optimise_classifier:

            # Stand-alone classifier optimisation
            args.train_classifier  = True
            args.train_adversarial = False
            args.train = False
            cfg['classifier']['fit']['verbose'] = 2


        cfg['classifier']['fit']['verbose'] = 2  # @TEMP

        # Initialise Keras backend
        initialise_backend(args)

        import keras
        import keras.backend as K
        from keras.models import load_model
        from keras.callbacks import Callback, TensorBoard, EarlyStopping
        from keras.utils.vis_utils import plot_model
        from adversarial.models import classifier_model


        # Neural network-specific initialisation of the configuration dict
        initialise_config(args, cfg)

        # Setup TensorBoard, if applicable
        tensorboard_dir = initialise_tensorboard(args, cfg)

        # Print the current environment setup
        print_env(args, cfg)
        pass

    # Loading data
    # --------------------------------------------------------------------------
    #data, features, features_decorrelation = load_data(args.input + 'data_10000.h5')
    data, features, features_decorrelation = load_data(args.input + 'data.h5')
    #data, features, features_decorrelation = load_data(args.input + 'data.h5', train=True)
    '''
    kNN_var = 'D2-k#minusNN'
    with Profile("Add variables"):
        # Tau21DDT
        from run.ddt.common import add_ddt
        add_ddt(data, path='models/ddt/ddt.pkl.gz')

        # D2-CSS
        #from run.css.common import add_css
        #add_css("D2", data)

        # D2-kNN
        from run.knn.common import add_knn, VAR as kNN_basevar, EFF as kNN_eff
        print "k-NN base variable: {} (cp. {})".format(kNN_basevar, kNN_var)
        add_knn(data, newfeat=kNN_var, path='models/knn/knn_{}_{}.pkl.gz'.format(kNN_basevar, kNN_eff))
        pass
    features.remove('Tau21')
    features.remove('D2')
    features.insert(0,'Tau21DDT')
    #features.insert(0,'D2CSS')
    features.insert(0,'D2-k#minusNN')
    '''
    num_features = len(features)

    # Regulsarisation parameter
    lambda_reg = cfg['classifier']['lambda_reg']  
    digits = int(np.ceil(max(-np.log10(lambda_reg), 0)))
    lambda_str = '{l:.{d:d}f}'.format(d=digits,l=lambda_reg).replace('.', 'p')
    print 'lambda_reg = ',lambda_reg
    print 'lambda_str = ',lambda_str

    # Get standard-formatted decorrelation inputs
    decorrelation = get_decorrelation_variables(data)
    aux_vars = ['logpt']
    data['logpt'] = pd.Series(np.log(data['pt'].values), index=data.index)
    
    # Specify common weights
    # -- Classifier
    weight_var = 'weight_adv'  # 'weight_adv' / 'weight_train'
    data['weight_clf'] = pd.Series(data[weight_var].values, index=data.index)

    # -- Adversary
    data['weight_adv'] = pd.Series(np.multiply(data['weight_adv'].values,  1. - data['signal'].values), index=data.index)


    # Classifier-only fit, cross-validation
    # --------------------------------------------------------------------------
    with Profile("Classifier-only fit, cross-validation"):

        # Define variable(s)
        basename = 'crossval_classifier_lambda{}'.format(lambda_str)
        basedir  = 'models/disco/classifier/crossval/'

        # Get indices for each fold in stratified k-fold training
        # @NOTE: No shuffling is performed -- assuming that's already done.
        skf = StratifiedKFold(n_splits=args.folds).split(data[features].values[data['train']  == 1],
                                                         data['signal'].values[data['train']  == 1])

        # Collection of classifiers and their associated training histories
        classifiers = list()
        histories   = list()

        # Train or load classifiers
        if args.optimise_classifier:  # args.train or args.train_classifier:
            log.info("Training cross-validation classifiers")

            # Loop `k` folds
            for fold, (train, validation) in enumerate(skf):
                with Profile("Fold {}/{}".format(fold + 1, args.folds)):

                    # Define unique name for current classifier
                    name = '{}__{}of{}'.format(basename, fold + 1, args.folds)

                    # Get classifier
                    classifier = classifier_model(num_features, **cfg['classifier']['model'])

                    # Parallelise on GPUs
                    # @NOTE: Store reference to base model to allow for saving.
                    #        Cf. [https://github.com/keras-team/keras/issues/8446#issuecomment-343559454]
                    parallelised = parallelise_model(classifier, args)

                    # Create custom Disco loss 
                    cfg['classifier']['compile']['loss'] = lambda y_true_tmp, y_pred: DisCo(y_true_tmp, y_pred, alpha = lambda_reg)

                    # Compile model (necessary to save properly)
                    parallelised.compile(**cfg['classifier']['compile'])

                    # Prepare arrays
                    X = data[features].values[train]
                    Y = np.stack((data['signal'].values[train], data['m'].values[train]), axis=1)
                    W = data['weight_clf'].values[train]
                    validation_data = (
                        data[features].values[validation],
                        np.stack((data['signal'].values[validation], data['m'].values[validation]), axis=1),
                        data['weight_clf'].values[validation]
                    )

                    # Create callbacks
                    callbacks = []

                    # -- TensorBoard
                    if args.tensorboard:
                        callbacks += [TensorBoard(log_dir=tensorboard_dir + 'classifier/fold{}/'.format(fold))]
                        pass

                    # Fit classifier model
                    ret = parallelised.fit(X, Y, sample_weight=W, validation_data=validation_data, callbacks=callbacks, **cfg['classifier']['fit'])

                    # Add to list of cost histories
                    histories.append(ret.history)

                    # Save classifier model and training history to file, both
                    # in unique output directory and in the directory for pre-
                    # trained classifiers
                    save([args.output, basedir], name, classifier, ret.history)
                    pass
                pass # end: k-fold cross-validation
            pass

        pass


    # Early stopping in case of stand-alone classifier optimisation
    # --------------------------------------------------------------------------
    if args.optimise_classifier:

        # Compute average validation loss
        val_avg = np.mean([hist['val_loss'] for hist in histories], axis=0)
        val_std = np.std ([hist['val_loss'] for hist in histories], axis=0)
        return val_avg[-1] + val_std[-1]


    # Classifier-only fit, full
    # --------------------------------------------------------------------------
    with Profile("Classifier-only fit, full"):

        # Define variable(s)
        name    = 'classifier_lambda{}'.format(lambda_str)
        basedir = 'models/disco/classifier/full/'

        if args.train or args.train_classifier:
            log.info("Training full classifier")

            # Get classifier
            classifier = classifier_model(num_features, **cfg['classifier']['model'])

            # Save classifier model diagram to file
            plot_model(classifier, to_file=args.output + 'model_{}.eps'.format(name), show_shapes=True)

            # Parallelise on GPUs
            parallelised = parallelise_model(classifier, args)

            # Create custom Disco loss 
            cfg['classifier']['compile']['loss'] = lambda y_true_tmp, y_pred: DisCo(y_true_tmp, y_pred, alpha = lambda_reg)

            # Compile model (necessary to save properly)
            parallelised.compile(**cfg['classifier']['compile'])

            # Create callbacks
            callbacks = []

            # -- TensorBoard
            if args.tensorboard:
                callbacks += [TensorBoard(log_dir=tensorboard_dir + name + '/')]
                pass

            # Prepare arrays
            X = data[features].values[data['train']  == 1]
            #Y = np.stack((data['signal'].values[data['train']  == 1], data['m'].values[data['train']  == 1]), axis=1)
            Y = np.stack((data['signal'].values[data['train']  == 1], data['m'].values[data['train']  == 1], data['weight_clf'].values[data['train']  == 1]), axis=1)
            W = data['weight_clf'].values[data['train']  == 1]
            validation_data = (
                data[features].values[data['train']  == 0],
                #np.stack((data['signal'].values[data['train']  == 0], data['m'].values[data['train']  == 0]), axis=1),
                np.stack((data['signal'].values[data['train']  == 0], data['m'].values[data['train']  == 0], data['weight_clf'].values[data['train']  == 0]), axis=1),
                data['weight_clf'].values[data['train']  == 0]
             )

            print 'Main X shape:', X.shape
            print 'Main Y shape:', Y.shape
            print 'Main W shape:', W.shape
            #print 'Y[:,0] = ', Y[:,0]
            #print 'Y[:,1] = ', Y[:,1]

            # Fit classifier model
            ret = parallelised.fit(X, Y, sample_weight=W, validation_data=validation_data, callbacks=callbacks, **cfg['classifier']['fit'])

            # Save classifier model and training history to file, both in unique
            # output directory and in the directory for pre-trained classifiers.
            save([args.output, basedir], name, classifier, ret.history)

            # Saving classifier in lwtnn-friendly format.
            lwtnn_save(classifier, 'disco_lambda{}'.format(lambda_str), basedir='models/disco/lwtnn/')

        else:

            # Load pre-trained classifier
            log.info("Loading full classifier from file")
            classifier, history = load(basedir, name)
            pass # end: train/load
        pass




# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(adversarial=True)

    # Call main function
    main(args)
    pass
