import os
import math
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt

import multiprocessing as mp

from TDA_helper_fcns import load_data



def rbf_scalar_weight(u, v, scale):
    """
    radial basis weighting function for rbf kernel
    used to construct SSM
    u and v are scalars
    """
    return math.exp((u**2 + v**2) / -scale)


def rbf_vector_weight(u, v, scale, norm_ord=2):
    """
    radial basis weighting function for rbf kernel
    used to construct SSM
    u and v are vectors (1D numpy arrays)
    """
    return math.exp(la.norm(u-v, norm_ord)**2 / -scale)


def build_vec_SSM(A, scale, norm_ord=2):
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
            SSM[i, j] = rbf_vector_weight(
                A[i, 1:-1],
                A[j, 1:-1],
                scale,
                norm_ord
                )

    return {"SSM" : SSM, "time" : tvec}


def build_1D_SSM(A, wgt_fcn, scale=1):
    """
    construct self-similarity matrix from array A with dim (n x 1)
    This version of the SSM function uses rbf weighting function
    scaling parameter defaults to 1

    OUTPUTS
    dictionary with gesture number, self-similarity matrix, array of time points
    """
    n = len(A)
    SSM = np.zeros(n**2).reshape(n, n) # self-similarity matrix
    for i in range(n):
        for j in range(n):
            SSM[i, j] = wgt_fcn(A[i], A[j], scale)

    return SSM


def subj_SSM_mp(inp_lst):
    """
    multiprocessing helper function for creating self similarity matrices
    inp_list - single list of inputs; format needed for multiprocessing
    sbj - subject number (e.g. '01', '12', '25')
    sdict - dictionary for each subject
    file_path - primary directory for saving data
    norm_ord - numpy vector norm to be passed to `build_SSM`
    """
    sbj, sdict, file_path, norm_ord = inp_lst
    # create and save SSMs for all gestures performed by subject
    for gnum, gest in sdict.items():
        sbj_path = file_path + "/" + sbj + "/"
        os.makedirs(sbj_path, exist_ok=True)
        ssm_dict = build_SSM(gest, norm_ord)
        ssm_frame = pd.DataFrame(ssm_dict["SSM"])
        ssm_frame.index = ssm_dict["time"]
        file_ref = sbj_path + gnum + ".csv"
        ssm_frame.to_csv(file_ref, index=True, sep=",")


################################################################################


if __name__ == "__main__":

    to_dir = "./Data/EMG_data_for_gestures-SSMs/"

    print("Loading Data...\n")
    gdat = load_data() # gestures data
    finps = [[sbj, sdict, to_dir, 2] for sbj, sdict in gdat.items()] # inp list

    print("Performing Multiprocessing Loop...\n")
    with mp.Pool(processes=6) as pool:
        pool.map(subj_SSM_mp, finps)

    print("Multiprocessing Complete.\n")
