import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt

import multiprocessing as mp
import json

from TDA_helper_fcns import load_data


def build_SSM(A, norm_ord=2):
    """
    construct self-similarity matrix from array A with dim (n, m)
    norm_ord - numpy vector norm; default is L2/ euclidian norm

    OUTPUTS
    dictionary with gesture number, self-similarity matrix, array of time points
    """
    tvec = A[:, 0] # time vector
    gnum = A[0, -1] # gesture number
    (r, c) = A[:, 1:-1].shape
    SSM = np.zeros(r**2).reshape(r, r) # self-similarity matrix
    for i in range(r):
        for j in range(r):
            SSM[i, j] = la.norm(A[i, 1:-1]-A[j, 1:-1], norm_ord)

    return {"gnum" : gnum, "SSM" : SSM, "time" : tvec}


def subj_SSM_mp(sdict, norm_ord=2):
    """
    multiprocessing helper function for creating self similarity matrices
    sdict is dictionary for each subject (e.g. '01', '12', '25')
    norm_ord - numpy vector norm to be passed to `build_SSM`
    """
    return { gnum : build_SSM(gest) for gnum, gest in sdict.items() }


################################################################################


if __name__ == "__main__":

    gdat = load_data() # gestures data

    pool = mp.Pool(6) # specify number of CPU cores

    par_res = {
    sbj : pool.apply_async(subj_SSM_mp, args=(sdict, 2)
    ) for sbj, sdict in gdat.items()
    }

    pool.close() # end multiprocessing
    pool.join()

    gdat_ssm = {sbj : i.get() for sbj, i in par_res.items()}

    with open("./ssm_data_mp.json", w) as fh:
        json.dump(gdat_ssm, fh)

    ### Read JSON from file ###
    #with open("./ssm_data_mp.json", r) as fh:
    #    gdat_ssm = json.load(fh)
