################################################################################
################ Create a Fused Similarity Template Class - SNF ################
###################### SNF class inherits from SSM parent ######################
################################################################################

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

from ssm import SSM


def euclidian_dist(i, j):
    """
    euclidian distance for scalar values
    """
    # abs equivalent to ((i-j)**2)**0.5 in scalar case
    return abs(i-j)


def gaussian(i, j, metric, s=1):
    """
    gaussian similarity kernel function
    i, j - data points between which similarity is measured
    metric - a metric
    s - variance parameter
    """
    return np.exp(-(metric(i,j)**2)/s)


def autotune_sigma(i, j, metric, inn, jnn, b):
    """
    i, j - data points between which similarity is measured
    meas - the similarity measure generating the SSM
    b - tuning parameter
    inn - numpy array object; nearest neighbors to point i
    jnn - numpy array object; nearest neighbors to point j

    Ref: Multiscale Geometric Summaries Similarity-based Sensor Fusion (p.3)
    """
    kN = inn.size
    kN_frac = 1/kN
    return b/3*( (kN_frac*inn.sum()) + (kN_frac*jnn.sum()) + metric(i,j) )


def tensor_sum_except(tensor, n, axs=0):
    """
    sum all matrices in a tensor excluding matrix `n`
    axs - axis over which sum is calculated; default axis 0
    """
    return tensor.sum(axis=axs) - tensor[n,:,:]


class SNF(SSM):
    """
    similarity network fusion (SNF) fused similarity template class inheriting
    from the `SSM` parent class. SSM Docstring is below

    k - float in (0,1]; proportion of nearest neighbors to
    vertex i (i.e. those js with the k largest similarity values in the
    ith row of the SSM)

    SSM DOCSTRING:
    CURRENTLY SUPPORTS SCALARS
    time_series - a (t x 1+m) numpy array object; t is time series
        1+m is time index + number of modalities
    metric - a function to be used in generating distance matrix SSMs
    """
    def __init__(self, time_series, k, metric=euclidian_dist):
        # inherit methods and properties from parent
        super().__init__(time_series, metric)
        self.k = k
        self.kN = int(self.k * self.n_obs) # set kN to int; this truncates


    def calc_sim_matrix(self, kern=gaussian, autotune=True, s=None, b=0.5):
        """
        calculate self-similarity matrix (SSM) using similarity kernel
        instead of distance metric

        kern - similarity kernel; a function
        autotune - Boolean; if True variance parameter will be autotuned based
            on average distance to nearest neighbors of xi and xj; default=True
        s - variace/ decay of similarity kernel; set to a specific float if
            autotune=False
        b - tunable parameter for autotuned variance [0.3, 0.8]

        """
        self.kern = kern
        self.reset_array()
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over observations in m
                for j in range(self.n_obs):
                    if autotune:
                        # find kN nearest neighbors of data point i
                        inn = np.array([
                            self.metric(self.mods[i,m], v) \
                            for v in self.mods[:,m]
                            ])
                        inn.sort()
                        # find kN nearest neighbors of data point j
                        jnn = np.array([
                            self.metric(self.mods[j,m], v) \
                            for v in self.mods[:,m]
                            ])
                        jnn.sort()
                        s = autotune_sigma(
                                self.mods[i, m],
                                self.mods[j, m],
                                metric=self.metric,
                                inn=inn[1:self.kN], # exclude self
                                jnn=jnn[1:self.kN], # exclude self
                                b=b
                                )
                    if i < j: # fill lower triangle only
                        self.array[m, i, j] = self.kern(
                            self.mods[i, m],
                            self.mods[j, m],
                            metric=self.metric,
                            s=s
                            )
            self.array[m, :, :] = self.array[m, :, :] + self.array[m, :, :].T
            self.array[m, :, :] += np.diag(np.ones(self.n_obs)) # main diagonal


    def calc_transition_matrix(self):
        """
        calculate a transition probability matrix from the SSM
        each row in the new array must sum to 1.

        Ref: Multiscale Geometric Summaries Similarity-based Sensor Fusion (p.4)
        """
        self.trans_mtrx = np.zeros(shape=(self.n_mods, self.n_obs, self.n_obs))
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over rows in SSM
                for j in range(self.n_obs): # loop over columns in SSM
                    if i != j:
                        self.trans_mtrx[m,i,j] = (
                            self.array[m,i,j] /
                            (2*(self.array[m,i,:].sum()-self.array[m,i,i]))
                            )
                    else:
                        self.trans_mtrx[m,i,j] = 0.5


    def calc_similarity_graph(self):
        """
        calculate a similarity graph (i.e."masked" transition probability
        matrix) from the SSM. This matrix represents a graph where vertices are
        data points and edges are the distances between vertices.

        k - float in (0,1]; proportion of nearest neighbors to
        vertex i (i.e. those js with the k largest similarity values in the
        ith row of the SSM)

        NOTE: trans_mtrx must exist; run `calc_transition_matrix()` method

        Ref: Multiscale Geometric Summaries Similarity-based Sensor Fusion (p.4)
        """
        self.similarity_graph = np.zeros(
            shape=(self.n_mods, self.n_obs, self.n_obs)
            )
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over rows of SSM
                knn = np.argsort(-self.array[m,i,:])[:self.kN] # knn index nums
                for j in range(self.n_obs): # loop over columns in SSM
                    if self.array[m,i,j] in self.array[m,i,:][knn]:
                        self.similarity_graph[m,i,j] = (
                            self.array[m,i,j] /
                            (2*self.array[m,i,:][knn].sum())
                            )


    def network_fusion(self, iters=50):
        """
        perform random walk over similarity graph

        OUTPUT:
        similarity_templates attribute - a 3D numpy array (i.e. tensor) of
        individual similarity matrices

        fused_similarity_template - A 2D numpy array representing the fusion
        of all modalities' similarity templates
        """
        self.similarity_templates = self.trans_mtrx.copy()
        for i in range(iters):
            for m in range(self.n_mods):
                self.similarity_templates[m,:,:] = (
                    self.similarity_graph[m,:,:] @
                    (
                        tensor_sum_except(self.similarity_templates, m) /
                        (self.n_mods-1)
                    ) @ self.similarity_graph[m,:,:].T
                )

        self.fused_similarity_template = (
            self.similarity_templates.sum(axis=0) / self.n_mods
            )


    def plot_template(
        self,
        m=0,
        fused=True,
        interp='nearest',
        cmap='afmhot',
        figsize=(12,8),
        save=False,
        path=None):
        """
        display fused similarity template matrix - this is the default behavior.
        if fused = False, then display self-similarity matrix for modality 'm'
        """
        plt.figure(figsize=figsize)
        if fused:
            plt.imshow(
                self.fused_similarity_template,
                interpolation=interp,
                cmap=cmap)
            plt.title("Fused similarity templates")
        else:
            plt.imshow(
                self.similarity_templates[m, :, :],
                interpolation=interp,
                cmap=cmap)
            plt.title("Fused transition matrix for modality " + str(m))
        if save:
            if path == None: raise ValueError("Must provide save path!")
            plt.savefig(path)
        else:
            plt.show()





if __name__ == "__main__":
    pass