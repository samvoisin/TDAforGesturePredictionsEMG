################################################################################
############### Generate Data Set of Persistence Diagram Vectors ###############
############# Calculating Rips complex on all channels w/ out time #############
################################################################################

import os

import numpy as np
import pandas as pd

import multiprocessing as mp

from ripser import ripser, Rips
from persim import plot_diagrams, PersImage

from TDA_helper_fcns import load_data, sublevel_set_time_series_dist


########################## parallel helper functions ###########################


def gest_to_pim(garray, px, sd):
    """
    Convert a gesture to a persistence image. The garray input should not
    include time series, subject number, gesture number columns.

    INPUTS
    garray - gesture array w/o time series & w/o subject and gesture number
    px - pixel dimension/ resolution; e.g. px=20 gives 20x20 persistence image
    sd - persistence image concentration parameter (gaussian)

    OUTPUTS
    flattened persistence image vector of length px**2
    """
    # instantiate persistence image generator & vietoris-rips complex generators
    rips = Rips(maxdim=1, verbose=False) # 1-D homology rips complex
    pim = PersImage(pixels=[px,px], spread=sd)

    dgms = rips.fit_transform(garray) # generate rips complex on points
    img = pim.transform(dgms[1]) # persistence image of 1 cycles
    return img.flatten()


def subj_to_pims(sbj, sdict, px, sd):
    """
    generate persistence images for all gestures for a given subject
    INPUTS
    sbj - subject number
    sdict - dict of all gestures performed by a subject; 24 per subject
    px - pixel dimension/ resolution; e.g. px=20 gives 20x20 persistence image
    sd - persistence image concentration parameter (gaussian)

    OUTPUTS
    """
    for gnum, garray in sdict.items():
        # flattened pim vector, gesture number, subject number
        return np.r_[
            gest_to_pim(garray[:,1:-2], pim_px, pim_sd),int(gnum[0]),int(sbj)
            ]

################################################################################


if __name__ == "__main__":

    gdat = load_data(subjects=["01", "21"]) # gestures data (test w/ 2)
    nvects = len(gdat.keys()) * 24 # each subject performs 24 total gestures
    pim_px = 20 # persistence image dims (square)
    pim_sd = 1e-5 # persistence image st. dev.

    # vects have equal # persim pix + 2 cols for subj & gest labels
    matsize = pim_px**2*nvects + 2*nvects
    pim_mat = np.zeros(matsize).reshape(nvects, -1) # init array for pim vects

    pool = mp.Pool(4) # use 4 CPU cores


    ##### here down has problems #############

    par_res = [pool.apply_async(
        subj_to_pims,
        args=(sbj, sdict, pim_px, pim_sd)
            ) for sbj, sdict in gdat.items()]


    # save matrix as DataFrame
    pim_df = pd.DataFrame(pim_mat)
    cnames = ["px"+str(i) for i in pim_df.columns]
    cnames[-2:] = ["gest", "subj"]
    pim_df.columns = cnames
    pim_df.to_csv("./Data/pim_vectors.csv", index=False)
