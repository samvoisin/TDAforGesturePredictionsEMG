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


def form_wgt_mat(A, kern, gamma, self_loop=False):
    """
    generate weight matrix
    INPUT:
    pim vectors A
    kernel function
    """
    (r, c) = A.shape
    wgt_mat = np.zeros(r * r).reshape(r, r)

    for i in range(r):
        for j in range(r):
            if i == j: wgt_mat[i, j] = 0
            else: wgt_mat[i, j] = rbf_kern(A[i, :], A[j, :], gamma)

    return wgt_mat


def cut(A, G):
    """
    perform 'cut' operation on a set of edges
    this operation calculates the cost of severing some weighted edges

    INPUTS
    A - set of vertices in a graph (i.e. a subgraph)
    G - the graph

    OUTPUT
    double
    """
    pass


def ratio_cut(L):
    pass


################################################################################

if __name__ == "__main__":
    pim_df = pd.read_csv("./pim_vectors_mp40.csv")
    pim_vecs = pim_df.values[:, :-2]

    # gamma=8 seems to provide ~8 clusts
    W = form_wgt_mat(pim_vecs, rbf_kern, 8)
    D = np.diag(W.sum(axis=1))
    L = D - W # laplacian

    evals, evecs = la.eig(L)
    eidx = np.argsort(evals.real)
    evecs = evecs.real[:, eidx]
    evals = evals.real[eidx]

    sns.scatterplot(range(20), evals[:20])
    plt.show()
