###############################################################################
############### Generate Data Set of Persistence Diagram Vectors ##############
###############################################################################

import os

import numpy as np
from ripser import ripser
from persim import plot_diagrams, PersImage
from persim import wasserstein, wasserstein_matching
from persim import bottleneck, bottleneck_matching

import matplotlib.pyplot as plt

from TDA_helper_fcns import load_data, sublevel_set_time_series_dist

def bottleneck_dist_mat(gdat, verbose=True):
    """
    Generate distance matrix for persistence diagrams of images

    INPUTS
    gdat - gestures data matrix; use output of load_data

    OUTPUTS
    distance matrix using bottleneck metric
    """
    ## code below taken from gen_all_pds.py - look there for os cmds and saving
    # iterate through all subjects
    pd_dict = dict()

    for sbj, sdict in gdat.items():
        # Dictionary of each subject with all gestures
        for gnum, garray in sdict.items():
            # loop through each signal in the gesture
            t_axis = garray[:, 0] # time data
            for s in range(1, garray.shape[1]-1):
                # sublevel set filtraton
                sls = sublevel_set_time_series_dist(garray[:,s])
                # generate persistence diagram
                pd = ripser(sls, distance_matrix=True)
                # remove inf persistence point
                pd["dgms"][0] = pd["dgms"][0][np.isfinite(pd["dgms"][0][:,1]),:]
                pd_dict[sbj+"_"+gnum] = pd

    # ordered list of keys
    klist = [k for k in pd_dict.keys()]
    klist.sort() # ordered ascending by subject, gesture

    # initialize bottleneck distance matrix
    bd_mat = np.zeros(len(klist)**2).reshape(len(klist), len(klist))

    for n, k in enumerate(klist):
        if verbose:
            ### progress bar ###
            pb = "~"*(int(n/len(klist)*100))+" "*(int((1-n/len(klist))*100))+"|"
            print(pb, end="\r")
            ####################
        for m, j in enumerate(klist):
            if n == m: bd_mat[n, m] = 0.0
            else: bd_mat[n, m] = bottleneck(
                pd_dict[k]["dgms"][0],
                pd_dict[j]["dgms"][0]
                )

    return bd_mat


if __name__ == "__main__":

    gdat = load_data(subjects=["02", "09"]) # gestures data
    nvects = len(gdat.keys()) * 24 # each subject performs 24 total gestures
    pimdim = 20 # persistence image dims (square)
    pimsd = 0.00001 # persistence image st. dev.
    # vects have equal # persim pix + 2 cols for subj & gest labels
    matsize = pimdim**2*nvects + 2*nvects

    pim_mat = np.zeros(matsize).reshape(nvects, -1)

    # instantiate persistence image object
    pim = PersImage(pixels=[pimdim,pimdim], spread=pimsd)

    ### ONLY LOOKING AT ONE CHANNEL - FIX IMMEDIATELY!!!

    for sbj, sdict in gdat.items():
        # Dictionary of each subject with all gestures
        for gnum, garray in sdict.items():
            # loop through each channel in the gesture
            t_axis = garray[:, 0] # time data
            for s in range(1, garray.shape[1]-1):
                sls = sublevel_set_time_series_dist(garray[:,s]) # sublvl sets
                pd = ripser(sls, distance_matrix=True) # generate pers diag
                # remove inf persistence point
                pd["dgms"][0] = pd["dgms"][0][np.isfinite(pd["dgms"][0][:,1]),:]
                pd_dict[sbj+"_"+gnum] = pd


    #bd_mat = bottleneck_dist_mat(gdat)

    #plt.matshow(bd_mat)
    #plt.show()
