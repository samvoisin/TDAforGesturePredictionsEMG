################################################################################
############### Generate Data Set of Persistence Diagram Vectors ###############
############# Calculating Rips complex on all channels w/ out time #############
################################################################################

import numpy as np
import numpy.linalg as la

from pandas import DataFrame
from scipy.ndimage import gaussian_filter
from sklearn.manifold import Isomap
from ripser import lower_star_img
from persim import PersImage

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
    dc.rms_smooth(100, 50)

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

    print("Generating Raw Data Persistence Images")

    # vects have equal num persim pix + 2 cols for subj & gest labels
    raw_pim_mat = np.zeros(shape=(len(arrays),pim_px**2+2)) # store pim vecs

    # initialize peristence image object
    pim = PersImage(spread=pim_sd, pixels=[pim_px,pim_px], verbose=False)

    # generate SSMs for each gesture
    raw_ssm_lst = [np.zeros(shape=(a.shape[0], a.shape[0])) for a in arrays]
    for n, a in enumerate(arrays):
        for i in range(a.shape[0]):
            for j in range(a.shape[0]):
                raw_ssm_lst[n][i,j] = cumulated_ts_2(a[i,:],a[j,:])

    # smooth SSM images
    for r, s in enumerate(raw_ssm_lst):
        raw_ssm_lst[r] = gaussian_filter(s, sigma=1)

    # generate persistence images
    for n, s in enumerate(raw_ssm_lst):
        pd = lower_star_img(s)
        img = pim.transform(pd[:-1,:]) # remove 'inf' persistence
        raw_pim_mat[n,:-2] = img.reshape(1,-1)
        raw_pim_mat[n,-2] = subj_lab[n]
        raw_pim_mat[n,-1] = gest_lab[n]

    # save matrix as DataFrame
    raw_pim_df = DataFrame(raw_pim_mat)
    cnames = ["px"+str(i) for i in raw_pim_df.columns]
    cnames[-2:] = ["subj", "gest"]
    raw_pim_df.columns = cnames
    raw_pim_df.to_csv("./Data/raw_pim_vectors.csv", index=False)

    ############################################################################
    ################################ ISOMAP SSMs ###############################
    ############################################################################

    print("Generating ISOMAP Persistence Images")

    # vects have equal num persim pix + 2 cols for subj & gest labels
    iso_pim_mat = np.zeros(shape=(len(arrays),pim_px**2+2)) # store pim vecs

    # initialize peristence image object
    pim = PersImage(spread=pim_sd, pixels=[pim_px,pim_px], verbose=False)

    # initialize embedding
    iso = Isomap(n_neighbors=3, n_components=1)

    # generate SSMs for each gesture
    iso_ssm_lst = [np.zeros(shape=(a.shape[0], a.shape[0])) for a in arrays]
    for n, a in enumerate(arrays):
        embed = iso.fit_transform(a)
        for i in range(embed.size):
            for j in range(embed.size):
                iso_ssm_lst[n][i,j] = cumulated_ts_2(embed[i,:], embed[j,:])

    # smooth SSM images
    for r, s in enumerate(iso_ssm_lst):
        iso_ssm_lst[r] = gaussian_filter(s, sigma=1)

    # generate persistence images
    for n, s in enumerate(iso_ssm_lst):
        pd = lower_star_img(s)
        img = pim.transform(pd[:-1,:]) # remove 'inf' persistence
        iso_pim_mat[n,:-2] = img.reshape(1,-1)
        iso_pim_mat[n,-2] = subj_lab[n]
        iso_pim_mat[n,-1] = gest_lab[n]

    # save matrix as DataFrame
    iso_pim_df = DataFrame(iso_pim_mat)
    cnames = ["px"+str(i) for i in iso_pim_df.columns]
    cnames[-2:] = ["subj", "gest"]
    iso_pim_df.columns = cnames
    iso_pim_df.to_csv("./Data/iso_pim_vectors.csv", index=False)

    ############################################################################
    #################################### SNF ###################################
    ############################################################################

    print("Generating SNF Persistence Images")

    # vects have equal num persim pix + 2 cols for subj & gest labels
    snf_pim_mat = np.zeros(shape=(len(arrays),pim_px**2+2)) # store pim vecs


    snf_lst = []
    for a in arrays:
        snf = SNF(a, k=0.2, metric=cumulated_euc_ts)
        # calculate graph weights to find knn
        snf.calc_weights()
        snf.normalize_weights()
        # generate and normalize knn graphs
        snf.calc_knn_weights()
        snf.normalize_knn_weights()
        # fuse graphs
        snf.network_fusion(eta=1, iters=20)
        # save template to dict
        snf_lst.append(snf.fused_similarity_template)

    # smooth SNF images
    for r, s in enumerate(snf_lst):
        snf_lst[r] = gaussian_filter(s, sigma=1)

    # generate persistence images
    for n, s in enumerate(snf_lst):
        pd = lower_star_img(s)
        img = pim.transform(pd[:-1,:]) # remove 'inf' persistence
        snf_pim_mat[n,:-2] = img.reshape(1,-1)
        snf_pim_mat[n,-2] = subj_lab[n]
        snf_pim_mat[n,-1] = gest_lab[n]

    # save matrix as DataFrame
    snf_pim_df = DataFrame(snf_pim_mat)
    cnames = ["px"+str(i) for i in snf_pim_df.columns]
    cnames[-2:] = ["subj", "gest"]
    snf_pim_df.columns = cnames
    snf_pim_df.to_csv("./Data/snf_pim_vectors.csv", index=False)
