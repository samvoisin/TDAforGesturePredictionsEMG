################################################################################
############### Spectral Clustering for Viewing Gesture Clusters ###############
############ Data set is persistence images - see pim_fullvec_set.py ###########
################################################################################

import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import SpectralClustering, KMeans

def rbf_kern(v1, v2, scale):
    """
    radial-basis function
    """
    return np.exp(la.norm(v1-v2, 2)**2 / -scale)


def form_wgt_mat(A, kern, gamma, tol=1e-10):
    """
    generate weight matrix
    INPUT:
    pim vectors A
    kernel function
    tolerance - any weight below this value will be set to zero
    """
    (r, c) = A.shape
    wgt_mat = np.zeros(r * r).reshape(r, r)

    for i in range(r):
        ### progress bar ###
        pb = "~"*int(i/r*100)+" "*int((1-i/r)*100)+"|"
        print(pb, end="\r")
        ####################
        for j in range(r):
            if i == j: wgt_mat[i, j] = 0
            else: wgt_mat[i, j] = rbf_kern(A[i, :], A[j, :], gamma)

    for i in range(r):
        for j in range(r):
            if wgt_mat[i, j] < tol and wgt_mat[i, j] != 0:
                wgt_mat[i, j] = 0

    return wgt_mat

def cluster_composition(clabs, df, idcol):
    """
    return percentage of each gesture composing each cluster
    INPUT:
    clabs - cluster labels (numpy array)
    df - data frame of observations including column of IDs (pandas DataFrame)
    idcol - specific column which includs ID values
    """
    pass

################################################################################

if __name__ == "__main__":
    pim_df = pd.read_csv("./pim_vectors_mp40.csv")
    pim_vecs = pim_df.values[:, :-2]

    W = form_wgt_mat(pim_vecs, rbf_kern, 15)
    D = np.diag(W.sum(axis=1))
    L = D - W # laplacian

    evals, evecs = la.eig(L)
    eidx = np.argsort(evals.real)
    evecs = evecs.real[:, eidx]
    evals = evals.real[eidx]

    # plot first 20 eigenvalues 0 = lambda_1 <= lambda_2 <= ... <= lambda_20
    sns.scatterplot(range(20), evals[:20])
    plt.show()

    X = evecs[:4]

    kmeans = KMeans(n_clusters=4, precompute_distances=True)
    kmeans.fit(X)

    kmeans.labels_
