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

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score


def rbf_kern(v1, v2, scale):
    """
    radial-basis function
    """
    return np.exp(la.norm(v1-v2, 2)**2 / -2*scale)


def unweighted_kern(v1, v2, scale=1):
    """
    kernel function for unweighted graph
    """
    return 1


def euclidian_kern(v1, v2, scale=1):
    """
    euclidian/ L2 norm between two vectors
    """
    return la.norm(v1-v2,ord=2)


def form_wgt_mat(A, kern, scale):
    """
    generate weight matrix

    INPUT:
    pim vectors A
    kernel function
    gamma - scale parameter
    """
    (r, c) = A.shape
    wgt_mat = np.zeros(r * r).reshape(r, r)

    for i in range(r):
        ### progress bar ###
        pb = "~"*int(i/r*100)+" "*int((1-i/r)*100)+"|"
        print(pb, end="\r")
        ####################
        for j in range(r):
            if i != j:
                wgt_mat[i, j] = kern(A[i, :], A[j, :], scale)

    return wgt_mat


def cluster_composition(clabs, df, idcol):
    """
    return percentage of each gesture composing each cluster
    INPUT:
    clabs - cluster labels (numpy array)
    df - data frame of observations including column of IDs (pandas DataFrame)
    idcol - specific column which includs ID values
    """
    df["cluster_ID"] = clabs
    unq_cats = df[idcol].unique() # unique category variables in idcol
    unq_cats = unq_cats.astype("int")
    unq_clusts = df["cluster_ID"].unique() # unique cluster IDs

    # store cluster composition percentages
    cpdf = np.zeros(len(unq_cats)*len(unq_clusts)).reshape(len(unq_clusts), -1)

    clust_ct = np.zeros(len(unq_clusts))
    for n, i in enumerate(unq_clusts):
        clust_tot = sum(df.cluster_ID==i)
        clust_ct[n] = clust_tot # number of pts in cluster i
        temp = df[idcol][df.cluster_ID==i]
        for m, j in enumerate(unq_cats):
            cpdf[n, m] = (sum(temp == j) / clust_tot)

    cpdf = pd.DataFrame(cpdf)
    cpdf.columns = unq_cats
    cpdf.index = unq_clusts
    cpdf["MemberCount"] = clust_ct

    return cpdf


################################################################################

if __name__ == "__main__":
    pim_df = pd.read_csv("./pim_vectors_mp40.csv")
    #pim_df = pim_df.loc[pim_df.gest != 1.0, :]
    pim_vecs = pim_df.values[:, :-2]
    pim_df.gest = pim_df.gest.astype("int")

    print("Generating weight matrix: ")
    W = form_wgt_mat(pim_vecs, rbf_kern, scale=1)
    D = np.diag(W.sum(axis=1))
    L = D - W # graph laplacian

    # http://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm.pdf
    #D_hlf = la.inv(np.diag([(n)**0.5 for n in np.diag(D)]))
    #L_sym = D_hlf @ W @ D_hlf # symmetric normalized laplacian (w/o I - L)

    #evals, evecs = la.eig(L) # code for graph laplacian
    evals, evecs = la.eig(L) # code for symmetric normalized laplacian
    eidx = np.argsort(evals.real)
    evecs = evecs.real[:, eidx]
    evals = evals.real[eidx]

    # plot first 20 eigenvalues: 0 = lambda_1 <= lambda_2 <= ... <= lambda_20
    sns.scatterplot(range(20), evals[:20])
    plt.plot([0, 20], [0, 0], color="black", linestyle="--")
    plt.show()

    X = evecs[:, :7]

    # normalize rows of X
    for n, i in enumerate(X):
        X[n, :] = i / la.norm(i, 2)

    kmeans = KMeans(n_clusters=6, precompute_distances=True)
    kmeans.fit_predict(X)

    print(homogeneity_score(kmeans.labels_, pim_df["gest"]))

    c_comp = cluster_composition(kmeans.labels_, pim_df, "gest")
    print(c_comp)
