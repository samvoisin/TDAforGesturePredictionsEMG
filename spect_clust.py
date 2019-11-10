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
    return np.exp(la.norm(v1-v2, 2)**2 / -scale)


def form_wgt_mat(A, kern, gamma, tol=1e-10):
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
            if i != j and kern(A[i, :], A[j, :], gamma) > tol:
                wgt_mat[i, j] = kern(A[i, :], A[j, :], gamma)

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
    unq_cids = df["cluster_ID"].unique() # unique cluster IDs

    # store cluster composition percentages
    cpdf = np.zeros(len(unq_cats)*len(unq_cids)).reshape(len(unq_cids), -1)

    for n, i in enumerate(unq_cids):
        clust_tot = sum(df.cluster_ID==i)
        temp = df[idcol][df.cluster_ID==i]
        for m, j in enumerate(unq_cats):
            cpdf[n, m] = (sum(temp == j) / clust_tot)*100

    cpdf = pd.DataFrame(cpdf)
    cpdf.columns = unq_cats
    cpdf.index = unq_cids

    return cpdf


def draw_graph(G):
    """
    draw graph G
    """
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)


################################################################################

if __name__ == "__main__":
    pim_df = pd.read_csv("./pim_vectors_mp40.csv")
    pim_vecs = pim_df.values[:, :-2]
    pim_df.gest = pim_df.gest.astype("int")

    W = form_wgt_mat(pim_vecs, rbf_kern, 10, tol=1e-6)
    D = np.diag(W.sum(axis=1))
    L = D - W # graph laplacian

    evals, evecs = la.eig(L)
    eidx = np.argsort(evals.real)
    evecs = evecs.real[:, eidx]
    evals = evals.real[eidx]

    # plot first 20 eigenvalues: 0 = lambda_1 <= lambda_2 <= ... <= lambda_20
    sns.scatterplot(range(20), evals[:20])
    plt.plot([0, 20], [0, 0], color="black", linestyle="--")
    plt.show()


    # draw the graph
    node_colors = [None,"red","blue","green","orange","purple","pink"]
    node_color_map = []
    V = W.shape[0] # cardinality of V
    G = nx.Graph()
    G.add_nodes_from(range(V))
    for i in range(V):
        node_color_map.append(node_colors[pim_df.gest[i]])
        for j in range(V):
            if j == i: break # only do upper triangle of matrix
            G.add_edge(i, j)
            G[i][j]["weight"] = W[i, j]

    nx.draw(G, node_color=node_color_map, with_labels=True)
    plt.show()

    for i in range(1,7):
        plt.subplot(2, 3, i)
        sns.scatterplot(evecs[:,i+3], evecs[:,i+4], hue=pim_df.gest, palette="Set1")
        plt.xlabel("EigVector" + str(i+3))
        plt.ylabel("EigVector" + str(i+4))
    plt.show()

    X = evecs[:, 3:8]

    kmeans = KMeans(n_clusters=5, precompute_distances=True)
    kmeans.fit_predict(X)

    c_comp = cluster_composition(kmeans.labels_, pim_df, "gest")
    print(c_comp)
