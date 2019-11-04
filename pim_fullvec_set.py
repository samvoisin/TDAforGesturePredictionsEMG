################################################################################
############### Generate Data Set of Persistence Diagram Vectors ###############
############# Calculating Rips complex on all channels w/ out time #############
################################################################################

import os

import numpy as np
import pandas as pd

from ripser import ripser, Rips
from persim import plot_diagrams, PersImage
from persim import wasserstein, wasserstein_matching
from persim import bottleneck, bottleneck_matching

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
    pimsd = 1e-5 # persistence image st. dev.
    # vects have equal # persim pix + 2 cols for subj & gest labels
    matsize = pimdim**2*nvects + 2*nvects

    pim_mat = np.zeros(matsize).reshape(nvects, -1) # empty matrix for pim vects
    # instantiate persistence image generator & vietoris-rips complex generator
    pim = PersImage(pixels=[pimdim,pimdim], spread=pimsd)
    rips = Rips(maxdim=1, verbose=False)

    vct = 0 # which vector we are on
    for sbj, sdict in gdat.items():
        # Dictionary of each subject with all gestures
        for gnum, garray in sdict.items():
            ### progress bar ###
            pb = "~"*int(vct/nvects*100)+" "*int((1-vct/nvects)*100)+"|"
            print(pb, end="\r")
            ####################
            dgms = rips.fit_transform(garray[:, 1:-1]) # generate rips complex
            img = pim.transform(dgms[1]) # persistence image of 1 cycles
            pim_mat[vct, :pimdim**2] = img.flatten()
            pim_mat[vct, -1] = int(gnum[0]) # gesture number
            pim_mat[vct, -2] = int(sbj) # subject number
            vct += 1

    # save matrix as DataFrame
    pim_df = pd.DataFrame(pim_mat)
    cnames = ["px"+str(i) for i in pim_df.columns]
    cnames[-2:] = ["gest", "subj"]
    pim_df.columns = cnames
    pim_df.to_csv("./Data/pim_vectors.csv", index=False)
