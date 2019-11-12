################################################################################
############### Spectral Clustering for Viewing Gesture Clusters ###############
############ Data set is persistence images - see pim_fullvec_set.py ###########
################################################################################

import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx # draw graphs
import plotly.express as px

from sklearn.cluster import KMeans


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


def form_wgt_mat(A, kern, scale, tol=1e-10):
    """
    generate weight matrix
    INPUT:
    pim vectors A
    kernel function
    gamma - scale parameter
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
    cpdf["ClusterCount"] = clust_ct

    return cpdf


################################################################################

### IS LAPLACIAN BLOCK DIAGONAL???? NEED TO CHECK ######


if __name__ == "__main__":
    pim_df = pd.read_csv("./pim_vectors_mp40.csv")
    #pim_df = pim_df.loc[pim_df.gest != 1.0, :]
    pim_vecs = pim_df.values[:, :-2]
    pim_df.gest = pim_df.gest.astype("int")

    W = form_wgt_mat(pim_vecs, rbf_kern, scale=1)
    D = np.diag(W.sum(axis=1))
    L = D - W # graph laplacian

    # http://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm.pdf
    D_hlf = la.inv(np.diag([(n)**0.5 for n in np.diag(D)]))
    L_sym = D_hlf @ W @ D_hlf # symmetric normalized laplacian (w/o I - L)

    #evals, evecs = la.eig(L) # code for graph laplacian
    evals, evecs = la.eig(L_sym) # code for symmetric normalized laplacian
    eidx = np.argsort(evals.real)
    evecs = evecs.real[:, eidx]
    evals = evals.real[eidx]

    # plot first 20 eigenvalues: 0 = lambda_1 <= lambda_2 <= ... <= lambda_20
    sns.scatterplot(range(20), evals[-1:-21:-1])
    plt.plot([0, 20], [0, 0], color="black", linestyle="--")
    plt.show()

    #for i in range(1,7):
    #    plt.subplot(2, 3, i)
    #    sns.scatterplot(evecs[:,i],
    #                    evecs[:,i+1],
    #                    hue=pim_df.gest,
    #                    palette="Set1")
    #    plt.xlabel("EigVector " + str(i))
    #    plt.ylabel("EigVector " + str(i+1))
    #plt.show()

    print(evecs)

    X = evecs[:, -1:-7:-1]

    # normalize rows of X
    for n, i in enumerate(X):
        X[n, :] = i / la.norm(i, 2)

    kmeans = KMeans(n_clusters=6, precompute_distances=True)
    kmeans.fit_predict(X)

    c_comp = cluster_composition(kmeans.labels_, pim_df, "gest")
    print(c_comp)
