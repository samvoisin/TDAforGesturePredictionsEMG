################################################################################
############### Generate Data Set of Persistence Diagram Vectors ###############
############# Calculating Rips complex on all channels w/ out time #############
################################################################################

import numpy as np
import numpy.linalg as la

from pandas import DataFrame
from scipy.ndimage import gaussian_filter
from sklearn.manifold import Isomap

from data_cube import DataCube
from similarity_network_fusion import SNF, cumulated_euc_ts


############################### helper functions ###############################


def cumulated_ts_2(a1, a2):
    """
    cumulated version of the time series w/ euclidean distance
    in which we take the sum values over time as time increases
    and then apply the chosen metric.
    i, j - arrays of data points
    """
    return la.norm(a1.sum(axis=0)-a2.sum(axis=0))


################################################################################

if __name__ == "__main__":

    dc = DataCube(
        subjects="all",
        gestures=["3", "4", "5", "6"],
        channels=["2", "4", "6", "8"],
        data_grp="parsed")
    dc.load_data()
    dc.normalize_modalities()
    dc.rms_smooth(200, 100)

    pim_px = 40 # persistence image dims (square)
    pim_sd = 0.5 # persistence image st. dev.

    ##### organize labels and arrays #####
    subj_lab = []
    gest_lab = []
    arrays = []

    for s, gdict in dc.data_set_smooth.items():
        for g, a in gdict.items():
            subj_lab.append(s)
            gest_lab.append(int(g[0]))
            arrays.append(a[:, 1:-1])

    ############################################################################
    ########################### Raw Data Vector SSMs ###########################
    ############################################################################

    print("Generating Raw Data Matrix Images")

    # generate SSMs for each gesture
    max_sz = 0 # track size to determine largest
    raw_ssm_lst = [np.zeros(shape=(a.shape[0], a.shape[0])) for a in arrays]
    for n, a in enumerate(arrays):
        for i in range(a.shape[0]):
            for j in range(a.shape[0]):
                raw_ssm_lst[n][i,j] = cumulated_ts_2(a[i,:],a[j,:])
                if a.shape[0] > max_sz: max_sz = a.shape[0]

    # smooth SSM images
    for r, s in enumerate(raw_ssm_lst):
        raw_ssm_lst[r] = gaussian_filter(s, sigma=1)

    # zero pad images
    shape = (max_sz,max_sz)
    pad_img = [
        np.pad(a, np.subtract(
            shape, a.shape),
            'constant',
            constant_values=0) for a in raw_ssm_lst
            ]

    raw_mat = np.zeros(shape=(len(arrays),max_sz**2+2)) # store pim vecs

    for n, p in enumerate(pad_img):
        raw_mat[n,:-2] = p.reshape(1, -1)
        raw_mat[n,-2] = subj_lab[n]
        raw_mat[n,-1] = gest_lab[n]

    # save matrix as DataFrame
    raw_df = DataFrame(raw_mat)
    cnames = ["px"+str(i) for i in raw_df.columns]
    cnames[-2:] = ["subj", "gest"]
    raw_df.columns = cnames
    raw_df.to_csv("./Data/raw_mat_vectors.csv", index=False)

    ############################################################################
    ################################ ISOMAP SSMs ###############################
    ############################################################################

    print("Generating ISOMAP Matrices")


    # initialize embedding
    iso = Isomap(n_neighbors=3, n_components=1)

    # generate SSMs for each gesture
    max_sz = 0 # track size to determine largest
    iso_ssm_lst = [np.zeros(shape=(a.shape[0], a.shape[0])) for a in arrays]
    for n, a in enumerate(arrays):
        embed = iso.fit_transform(a)
        for i in range(embed.size):
            for j in range(embed.size):
                iso_ssm_lst[n][i,j] = cumulated_ts_2(embed[i,:], embed[j,:])
                if embed.shape[0] > max_sz: max_sz = embed.shape[0]

    # smooth SSM images
    for r, s in enumerate(iso_ssm_lst):
        iso_ssm_lst[r] = gaussian_filter(s, sigma=1)

    # zero pad images
    shape = (max_sz,max_sz)
    pad_img = [
        np.pad(a, np.subtract(
            shape, a.shape),
            'constant',
            constant_values=0) for a in iso_ssm_lst
            ]

    iso_mat = np.zeros(shape=(len(arrays),max_sz**2+2)) # store pim vecs

    for n, p in enumerate(pad_img):
        iso_mat[n,:-2] = p.reshape(1, -1)
        iso_mat[n,-2] = subj_lab[n]
        iso_mat[n,-1] = gest_lab[n]


    # save matrix as DataFrame
    iso_df = DataFrame(iso_mat)
    cnames = ["px"+str(i) for i in iso_df.columns]
    cnames[-2:] = ["subj", "gest"]
    iso_df.columns = cnames
    iso_df.to_csv("./Data/iso_mat_vectors.csv", index=False)

    ############################################################################
    #################################### SNF ###################################
    ############################################################################

    print("Generating SNF Matrices")

    subj_lab = []
    gest_lab = []
    snf_lst = []
    max_sz = 0 # track size to determine largest
    for a in arrays:
        snf = SNF(a, k=0.5, metric=cumulated_euc_ts)
        # calculate graph weights to find knn
        snf.calc_weights()
        snf.normalize_weights()
        # generate and normalize knn graphs
        snf.calc_knn_weights()
        snf.normalize_knn_weights()
        # fuse graphs
        snf.network_fusion(eta=1, iters=10)
        # save template to dict
        snf_lst.append(snf.fused_similarity_template)
        if snf.fused_similarity_template.shape[0] > max_sz:
            max_sz = snf.fused_similarity_template.shape[0]

    # smooth SNF images
    for r, s in enumerate(snf_lst):
        snf_lst[r] = gaussian_filter(s, sigma=1)


    # zero pad images
    shape = (max_sz,max_sz)
    pad_img = [
        np.pad(a, np.subtract(
            shape, a.shape),
            'constant',
            constant_values=0) for a in iso_ssm_lst
            ]

    snf_mat = np.zeros(shape=(len(arrays),max_sz**2+2)) # store pim vecs

    for n, p in enumerate(pad_img):
        snf_mat[n,:-2] = p.reshape(1, -1)
        snf_mat[n,-2] = subj_lab[n]
        snf_mat[n,-1] = gest_lab[n]


    # save matrix as DataFrame
    snf_df = DataFrame(snf_mat)
    cnames = ["px"+str(i) for i in snf_df.columns]
    cnames[-2:] = ["subj", "gest"]
    snf_df.columns = cnames
    snf_df.to_csv("./Data/snf_mat_vectors.csv", index=False)
