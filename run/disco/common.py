# -*- coding: utf-8 -*-

"""Common methods for training and testing neural network classifiers."""

# Basic import(s)
import re
import sys
import json
import logging as log
from pprint import pprint
import subprocess
import datetime
import copy

# Scientific import(s)
import numpy as np
import pandas as pd

# Project import(s)
from adversarial.utils import INPUT_VARIABLES
from adversarial.profile import profile


@profile
def add_disco (data, clf, newfeat=None):
    """
    Add neural network tagger from `clf to `data`. Modifies `data` in-place.

    Arguments:
        data: Pandas DataFrame to which to add classifier variable.
        clf: Keras network model, from which to get classifier variable.
        newfeat: Name of output feature.
    """
    # Check(s)
    assert newfeat is not None, "Please specify an output feature name"

    # Add NN-classifier variable to DataFrame
    features = copy.deepcopy(INPUT_VARIABLES)
    #features.remove('Tau21')
    #features.remove('D2')
    #features.insert(0, 'Tau21DDT')
    #features.insert(0, 'D2CSS')
    #features.insert(0,'D2-k#minusNN')
    data[newfeat] = pd.Series(clf.predict(data[features].values, batch_size=2048).flatten(), index=data.index)
    return


def parallelise_model (model, args):
    """
    Parallelise model on GPUs. Requires TensorFlow backend, GPU-enabled running
    environment, and more than once devices requested/available.

    Arguments:
        model: Keras model to be parallelised.
        args: Namespace holding commandline arguments, for configuration.

    Returns:
        Parallelised, multi-GPU Keras model if possible; otherwise the unaltered
        input model.
    """

    # Import(s) -- done here to ensure that Keras background has been configured.
    from keras.utils import multi_gpu_model

    # Parallelise on GPUs
    if (not args.theano) and args.gpu and args.devices > 1:
        parallelised = multi_gpu_model(model, args.devices)
    else:
        parallelised = model
        pass

    return parallelised


def initialise_config (args, cfg):
    """
    Neural network-specific initialisation of the configuration dict. Modifies
    `cfg` in-place.

    Arguments:
        args: Namespace holding commandline arguments.
        cfg: Configuration dict.
    """

    # Import(s)
    import keras

    # If the `model/architecture` parameter is provided as an int, convert
    # to list of empty dicts
    for network in ['classifier']:
        if isinstance(cfg[network]['model']['architecture'], int):
            cfg[network]['model']['architecture'] = [{} for _ in range(cfg[network]['model']['architecture'])]
            pass
        pass


    # Multiply batch size by number of devices, to ensure equal splits.
    for model in ['classifier']:
        cfg[model]['fit']['batch_size'] *= args.devices
        pass

    # Validate learning rates and decays
    # @NOTE: If e.g. `lr = -3`, then let `lr -> 10^(lr) = 1E-03`
    transform = lambda v: np.power(10., v)
    for mdl in cfg.keys():
        for key in ['lr', 'decay']:
            if key in cfg[mdl]['compile'] and cfg[mdl]['compile'][key] < 0:
                log.info("Scaling {}/{} {} -> {}".format(mdl, key, cfg[mdl]['compile'][key], transform(cfg[mdl]['compile'][key])))
                cfg[mdl]['compile'][key] = transform(cfg[mdl]['compile'][key])
                pass
            pass
        pass

    # Validate architecture
    # @NOTE: If `0 < units < 10`, then let `units -> 2^(units)`
    do_transform_units = lambda d: 'units' in d and d['units'] > 0 and d['units'] < 10
    transform_units    = lambda d: int(np.power(2, d['units']))
    for mdl in cfg.keys():

        # Defaults + per-layer
        layers = list()
        if 'architecture' in cfg[mdl]['model']:
            layers += cfg[mdl]['model']['architecture']
            pass

        if 'default' in cfg[mdl]['model']:
            layers += [cfg[mdl]['model']['default']]
            pass

        for layer in layers:
            if do_transform_units(layer):
                log.info("Scaling {}/units {} -> {}".format(mdl, layer['units'], transform_units(layer)))
                layer['units'] = transform_units(layer)
                pass
            pass
        pass

    # Evaluate the 'optimizer' fields for each model, once and for all
    # @NOTE: This should be done _last_, to ensure that `lr` and `decay` have
    #        been properly transformed, if necessary.
    for model in ['classifier']:
        opts = cfg[model]['compile']
        opts['optimizer'] = eval("keras.optimizers.{}(lr={}, decay={})" \
                                 .format(opts['optimizer'],
                                         opts.pop('lr'),
                                         opts.pop('decay', 0)))
        pass

    return


def initialise_tensorboard (args, cfg):
    """
    Setup TensorBoard, if applicable.

    Arguments:
        args: Namespace holding commandline arguments.
        cfg: Configuration dict.

    Return:
        Directory to which TensorBoard logs are written; or `None`:

    Raises:
        `AssertionError` if TensorBoard is requested with Theano backend.
    """

    # Start TensorBoard instance
    tensorboard_dir = None
    if args.tensorboard:
        assert not args.theano, "TensorBoard requires TensorFlow backend."

        tensorboard_dir = 'logs/tensorboard/{}/'.format('-'.join(re.split('-|:| ', str(datetime.datetime.now()).replace('.', 'T'))) if args.jobname == "" else args.jobname)
        log.info("Writing TensorBoard logs to '{}'".format(tensorboard_dir))
        pass

    return tensorboard_dir


def print_env (args, cfg):
    """
    Print/log the current setup and environment.

    Arguments:
        args: Namespace holding commandline arguments.
        cfg: Configuration dict.
    """

    # Import(s)
    import keras
    import keras.backend as K

    # Print setup information
    log.info("Running '%s'" % __file__)
    log.info("Command-line arguments:")
    pprint(vars(args))

    log.info("Configuration file contents:")
    pprint(cfg)

    log.info("Python version: {}".format(sys.version.split()[0]))
    log.info("Numpy  version: {}".format(np.__version__))
    try:
        log.info("Keras  version: {}".format(keras.__version__))
        log.info("Using keras backend: '{}'".format(K.backend()))
        if K.backend() == 'tensorflow':
            import tensorflow
            print "  TensorFlow version: {}".format(tensorflow.__version__)
        else:
            import theano
            print "  Theano version: {}".format(theano.__version__)
            pass
    except NameError: log.info("Keras not imported")

    # Save command-line argument namespace and configuration dict in output
    # directory.
    for obj, mthd, name in zip([args, cfg], [vars, str], ['args', 'config']):
        with open(args.output + '{}.json'.format(name), 'wb') as f:
            json.dump(mthd(obj), f, indent=4, sort_keys=True)
            pass
        pass

    return
