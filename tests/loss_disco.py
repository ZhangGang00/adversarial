#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing loss study."""

# Basic import(s)
import os
import glob
import json
import itertools
from pprint import pprint

# Set Keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Scientific import(s)
import ROOT
import numpy as np
import root_numpy
import matplotlib.pyplot as plt

# Project import(s)
from adversarial.utils import parse_args, initialise, mkdir, load_data, get_decorrelation_variables
from adversarial.layers import PosteriorLayer
from adversarial.profile import profile, Profile
from adversarial.constants import *
from .studies.common import *

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Common definitions
    num_folds   = 3

    # Perform classifier loss study
    plot_classifier_training_loss(num_folds)

    return 0


@profile
def plot_classifier_training_loss (num_folds, basedir='models/disco/classifier/crossval/'):
    """
    Plot the classifier training loss.
    """

    # Check(s)
    if not basedir.endswith('/'):
        basedir += '/'
        pass

    # Get paths to classifier training losses
    #paths = sorted(glob.glob(basedir + '/history__crossval_classifier_lambda30__*of{}.json'.format(num_folds)))
    basedir = 'models/disco/classifier/full/'
    paths = sorted(glob.glob(basedir + '/history__classifier_lambda30.json'))
    #paths = sorted(glob.glob('/afs/cern.ch/work/g/gang/boosted_dijetISR/my_adversarial/history__classifier_lambda30.json'))

    if len(paths) == 0:
        print "No models found for classifier CV study."
        return

    index = [0]
    # Read losses from files
    losses = {'train': list(), 'val': list()}
    for path in paths:
        with open(path, 'r') as f:
            d = json.load(f)
            pass

        loss = np.array(d['val_loss'])
        #loss1 = np.delete(loss, index, axis=0)
        #print "Outliers:", loss[np.abs(loss - 0.72) < 0.02]
        #loss[np.abs(loss - 0.72) < 0.02] = np.nan  # @FIXME: This probably isn't completely kosher
        losses['val'].append(loss)
        loss = np.array(d['loss'])
        #loss1 = np.delete(loss, index, axis=0)
        losses['train'].append(loss)
        pass

    #print 'losses[val]:',losses['val']
    #print 'losses[train]:',losses['train']

    # Define variable(s)
    bins     = np.arange(len(loss))
    histbins = np.arange(len(loss) + 1) + 0.5

    # Canvas
    c = rp.canvas(batch=True)

    # Plots
    categories = list()

    for name, key, colour, linestyle in zip(['Validation', 'Training'], ['val', 'train'], [rp.colours[4], rp.colours[1]], [1,2]):

        # Histograms
        loss_mean = np.nanmean(losses[key], axis=0)
        loss_std  = np.nanstd (losses[key], axis=0)
        #print 'loss_mean:',loss_mean
        #print 'loss_std:',loss_std
        hist = ROOT.TH1F(key + '_loss', "", len(histbins) - 1, histbins)
        for idx in range(len(loss_mean)):
            hist.SetBinContent(idx + 1, loss_mean[idx])
            hist.SetBinError  (idx + 1, loss_std [idx])
            pass

        c.hist([0], bins=[0, max(bins)], linewidth=0, linestyle=0)  # Force correct x-axis
        #c.hist(hist, fillcolor=colour, alpha=0.3, option='LE3')
        c.hist(hist, linecolor=colour, linewidth=3, linestyle=linestyle, option='HISTL')

        categories += [(name,
                        {'linestyle': linestyle, 'linewidth': 3,
                         'linecolor': colour, 'fillcolor': colour,
                         'alpha': 0.3, 'option': 'FL'})]
        pass

    # Decorations
    c.pads()[0]._yaxis().SetNdivisions(505)
    c.xlabel("Training epoch")
    #c.ylabel("Cross-validation classifier loss, L_{clf}")
    c.ylabel("Classifier loss, L_{clf}")
    c.xlim(0, max(bins))
    #c.ylim(0.3, 2)
    c.ylim(0., 5.)
    c.legend(categories=categories, width=0.25)  # ..., xmin=0.475
    c.text(TEXT + ["#it{W} jet tagging", "Neural network (NN) + DisCo classifier"],
           qualifier=QUALIFIER)
    # Save
    mkdir('figures/')
    c.save('figures/loss_disco.pdf')
    return



# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
