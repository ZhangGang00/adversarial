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

def DisCo(y_true, y_pred, x_in, alpha = 0.):
    from keras import backend as K
    from keras.losses import binary_crossentropy
    import tensorflow as tf

    #alpha determines the amount of decorrelation; 0 means no decorrelation.
    #Note that the decorrelating feature is also used for learning.
   
    print 'x_in = ',x_in
    X = tf.gather(x_in, [0], axis=1) #decorrelate with the second element of the input (=mass)
    #X = K.variable(x_in[1]) #decorrelate with the second element of the input (=mass)
    #X = x_in #decorrelate with the second element of the input (=mass)
    Y = y_pred
    print 'X shape: ', X.get_shape()
    print 'Y shape: ', Y.get_shape()

    LX = K.shape(X)[0]
    LY = K.shape(Y)[0]
    print 'LX : ', LX
    print 'LY : ', LY
    
    X=K.reshape(X,shape=(LX,1))
    Y=K.reshape(Y,shape=(LY,1))
    print 'X shape: ', X.get_shape()
    print 'Y shape: ', Y.get_shape()
 
    ajk = K.abs(K.reshape(K.repeat(X,LX),shape=(LX,LX)) - K.transpose(X))
    bjk = K.abs(K.reshape(K.repeat(Y,LY),shape=(LY,LY)) - K.transpose(Y))
    print 'ajk shape: ', ajk.get_shape()
    print 'bjk shape: ', bjk.get_shape()

    Ajk = ajk - K.mean(ajk,axis=0)[None, :] - K.mean(ajk,axis=1)[:, None] + K.mean(ajk)
    Bjk = bjk - K.mean(bjk,axis=0)[None, :] - K.mean(bjk,axis=1)[:, None] + K.mean(bjk)
    print 'Ajk shape: ', Ajk.get_shape()
    print 'Bjk shape: ', Bjk.get_shape()

    dcor = K.sum(Ajk*Bjk) / K.sqrt(K.sum(Ajk*Ajk)*K.sum(Bjk*Bjk))
    
    return binary_crossentropy(y_true,y_pred) + alpha*dcor


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
    data, features, features_decorrelation = load_data(args.input + 'data_10000.h5', train=True)
    #data, features, features_decorrelation = load_data(args.input + 'data.h5', train=True)
    features.insert(0,'m')
    num_features = len(features)

    # Regulsarisation parameter
    lambda_reg = cfg['combined']['model']['lambda_reg']  # Use same `lambda` as the adversary
    digits = int(np.ceil(max(-np.log10(lambda_reg), 0)))
    lambda_str = '{l:.{d:d}f}'.format(d=digits,l=lambda_reg).replace('.', 'p')

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

    # Create custom Disco loss 
    #cfg['classifier']['compile']['loss'] = lambda y_true, y_pred: DisCo(y_true, y_pred, data['m'].values, alpha = 2.)


    # Classifier-only fit, cross-validation
    # --------------------------------------------------------------------------
    with Profile("Classifier-only fit, cross-validation"):

        # Define variable(s)
        basename = 'crossval_classifier'
        basedir  = 'models/disco/classifier/crossval/'

        # Get indices for each fold in stratified k-fold training
        # @NOTE: No shuffling is performed -- assuming that's already done.
        skf = StratifiedKFold(n_splits=args.folds).split(data[features].values,
                                                         data['signal'].values)

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
                    cfg['classifier']['compile']['loss'] = lambda y_true, y_pred: DisCo(y_true, y_pred, parallelised.input, alpha = 2.)

                    # Compile model (necessary to save properly)
                    parallelised.compile(**cfg['classifier']['compile'])

                    # Prepare arrays
                    X = data[features].values[train]
                    Y = data['signal'].values[train]
                    W = data['weight_clf'].values[train]
                    validation_data = (
                        data[features].values[validation],
                        data['signal'].values[validation],
                        data['weight_clf'].values[validation]
                    )

                    # Create callbacks
                    callbacks = []

                    # -- TensorBoard
                    if args.tensorboard:
                        callbacks += [TensorBoard(log_dir=tensorboard_dir + 'classifier/fold{}/'.format(fold))]
                        pass

                    # Compute initial losses
                    X_val, Y_val, W_val = validation_data
                    eval_opts = dict(batch_size=cfg['classifier']['fit']['batch_size'], verbose=0)
                    initial_losses = [[parallelised.evaluate(X,     Y,     sample_weight=W,     **eval_opts)],
                                      [parallelised.evaluate(X_val, Y_val, sample_weight=W_val, **eval_opts)]]

                    # Fit classifier model
                    ret = parallelised.fit(X, Y, sample_weight=W, validation_data=validation_data, callbacks=callbacks, **cfg['classifier']['fit'])

                    # Prepend initial losses
                    for metric, loss_train, loss_val in zip(parallelised.metrics_names, *initial_losses):
                        ret.history[metric]         .insert(0, loss_train)
                        ret.history['val_' + metric].insert(0, loss_val)
                        pass

                    # Add to list of cost histories
                    histories.append(ret.history)

                    # Add to list of classifiers
                    classifiers.append(classifier)

                    # Save classifier model and training history to file, both
                    # in unique output directory and in the directory for pre-
                    # trained classifiers
                    save([args.output, basedir], name, classifier, ret.history)
                    pass
                pass # end: k-fold cross-validation
            pass

        '''
        else:
            # Load pre-trained classifiers
            log.info("Loading cross-validation classifiers from file")
            try:
                for fold in range(args.folds):
                    name = '{}__{}of{}'.format(basename, fold + 1, args.folds)
                    classifier, history = load(basedir, name)
                    classifiers.append(classifier)
                    histories.append(history)
                    pass
            except IOError as err:
                log.error("{}".format(err))
                log.error("Not all files were loaded. Exiting.")
                #return 1  # @TEMP
                pass

            pass # end: train/load
        '''
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
        name    = 'classifier'
        basedir = 'models/disco/classifier/full/'

        if args.train or args.train_classifier:
            log.info("Training full classifier")

            # Get classifier
            classifier = classifier_model(num_features, **cfg['classifier']['model'])

            # Save classifier model diagram to file
            plot_model(classifier, to_file=args.output + 'model_{}.png'.format(name), show_shapes=True)

            # Parallelise on GPUs
            parallelised = parallelise_model(classifier, args)

            # Create custom Disco loss 
            cfg['classifier']['compile']['loss'] = lambda y_true, y_pred: DisCo(y_true, y_pred, parallelised.input, alpha = 2.)

            # Compile model (necessary to save properly)
            parallelised.compile(**cfg['classifier']['compile'])

            # Create callbacks
            callbacks = []

            # -- TensorBoard
            if args.tensorboard:
                callbacks += [TensorBoard(log_dir=tensorboard_dir + name + '/')]
                pass

            # Prepare arrays
            X = data[features].values
            Y = data['signal'].values
            W = data['weight_clf'].values
            M = data['m'].values
            print 'Main X shape:', X.shape
            print 'Main Y shape:', Y.shape
            print 'Main M shape:', M.shape
            #print data['m']

            # Fit classifier model
            ret = parallelised.fit(X, Y, sample_weight=W, callbacks=callbacks, **cfg['classifier']['fit'])

            # Save classifier model and training history to file, both in unique
            # output directory and in the directory for pre-trained classifiers.
            save([args.output, basedir], name, classifier, ret.history)

            # Saving classifier in lwtnn-friendly format.
            lwtnn_save(classifier, 'disco', basedir='models/disco/lwtnn/')

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
