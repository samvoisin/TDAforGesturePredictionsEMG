################################################################################
############### Generate Data Set of Persistence Diagram Vectors ###############
############# Calculating Rips complex on all channels w/ out time #############
################################################################################

import numpy as np
import pandas as pd

import multiprocessing as mp

from ripser import ripser, Rips
from persim import plot_diagrams, PersImage

from data_cube import DataCube
#from TDA_helper_fcns import load_data, sublevel_set_time_series_dist



########################## parallel helper functions ###########################

def subj_to_pims(sbj, sdict, px, sd):
    """
    generate persistence images for all gestures for a given subject
    INPUTS
    sbj - subject number
    sdict - dict of all gestures performed by a subject; 24 per subject
    px - pixel dimension/ resolution; e.g. px=20 gives 20x20 persistence image
    sd - persistence image concentration parameter (gaussian)

    OUTPUTS
    array of gestures made by subject
    """
    # instantiate persistence image generator & vietoris-rips complex generators
    rips = Rips(maxdim=1, verbose=False) # 1-D homology rips complex
    pim = PersImage(pixels=[px,px], spread=sd)
    nobs = 24 # each subject performs 24 gestures
    # each vector have equal # persim pix + 2 cols for subj & gest labels
    res_mat = np.zeros(px**2*nobs + 2*nobs).reshape(nobs, -1)

    v = 0
    for gnum, garray in sdict.items():
        # generate rips complex on points; slice out time col and gesture label
        dgms = rips.fit_transform(garray[:, 1:-1])
        img = pim.transform(dgms[1]) # persistence image of 1 cycles
        obs_vec = np.r_[img.flatten(), int(gnum[0]), int(sbj)]
        res_mat[v, :] = obs_vec
        v += 1

    return res_mat

################################################################################

if __name__ == "__main__":

    dc = DataCube(
        subjects="all",
        gestures=["1", "2", "3", "4"],
        channels=["2", "4", "5", "6", "8"],
        data_grp="parsed"
        )
    dc.load_data()

    nvects = len(dc.data_set.keys()) * 24 # each subj performs 24 total gests
    pim_px = 20 # persistence image dims (square)
    pim_sd = 1e-5 # persistence image st. dev.

    # vects have equal # persim pix + 2 cols for subj & gest labels
    matsize = pim_px**2*nvects + 2*nvects
    pim_mat = np.zeros(matsize).reshape(nvects, -1) # init array for pim vects

    pool = mp.Pool(6) # specify number of CPU cores

    par_res = [
    pool.apply_async(subj_to_pims, args=(sbj, sdict, pim_px, pim_sd)
    ) for sbj, sdict in dc.data_set.items()
    ]

    pool.close()
    pool.join()

    # stack persistence image vectors
    r = 0
    for i in par_res:
        pim_mat[r:r+24, :] = i.get()
        r += 24

    # save matrix as DataFrame
    pim_df = pd.DataFrame(pim_mat)
    cnames = ["px"+str(i) for i in pim_df.columns]
    cnames[-2:] = ["gest", "subj"]
    pim_df.columns = cnames
    pim_df.to_csv("./pim_vectors_mp20_sbst.csv", index=False)
