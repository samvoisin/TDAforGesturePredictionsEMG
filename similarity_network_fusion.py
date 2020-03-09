################################################################################
################ Create a Fused Similarity Template Class - SNF ################
###################### SNF class inherits from SSM parent ######################
################################################################################

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

from ssm import SSM

############# metrics and kernels #############

def euclidian_dist(i, j):
    """
    euclidian distance for scalar values
    """
    # abs equivalent to ((i-j)**2)**0.5 in scalar case
    # end-of-array index included for compatibility
    return abs(i[-1] - j[-1])


def cumulated_euc_ts(i, j):
    """
    cumulated version of the time series w/ euclidean distance
    in which we take the sum values over time as time increases
    and then apply the chosen metric.
    i, j - arrays of data points
    """
    # abs equivalent to ((i-j)**2)**0.5 in scalar case
    return abs(i.sum() - j.sum())


def gaussian(i, j, metric, s=1):
    """
    gaussian similarity kernel function
    i, j - data points between which similarity is measured
    metric - a metric
    s - variance parameter
    """
    return np.exp(-(metric(i,j)**2)/s)

###############################################


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
    def __init__(
        self,
        time_series,
        k,
        metric=euclidian_dist,
        autotune=True,
        s=None):
        # inherit methods and properties from parent
        super().__init__(time_series, metric)
        self.k = k
        self.kN = int(self.k * self.n_obs) # set kN to int; this truncates


    def calc_weights(self, kern=gaussian, autotune=True, s=None, b=0.5):
        """
        calculate weight matrix W on for graph G using similarity kernel
        instead of distance metric

        kern - similarity kernel; a function
        autotune - Boolean; if True variance parameter will be autotuned based
            on average distance to nearest neighbors of xi and xj; default=True
        s - variace/ decay of similarity kernel; set to a specific float if
            autotune=False
        b - tunable parameter for autotuned variance [0.3, 0.8]

        """
        self.kern = kern
        self.W = np.zeros(shape=(self.n_mods, self.n_obs, self.n_obs))
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over observations in m
                for j in range(self.n_obs):
                    if i < j: # fill lower triangle only
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
                                    inn=inn[0:self.kN], # include self
                                    jnn=jnn[0:self.kN], # include self
                                    b=b
                                    )
                        self.W[m, i, j] = self.kern(
                            self.mods[:i, m],
                            self.mods[:j, m],
                            metric=self.metric,
                            s=s
                            )
            self.W[m,:,:] = self.W[m,:,:] + self.W[m,:,:].T
            self.W[m,:,:] += np.diag(np.ones(self.n_obs)) # main diagonal


    def normalize_weights(self):
        """
        normalize weight matrix W such that rows sum to 1
        this instantiates attribute array P
        """
        self.P = np.zeros(shape=(self.n_mods, self.n_obs, self.n_obs))
        for m in range(self.n_mods): # loop over modalities
            self.P[m,:,:] = (
                self.W[m,:,:] / self.W[m,:,:].sum(axis=1).reshape(-1,1)
                )


    def calc_knn_weights(self, kern=gaussian, autotune=True, s=None, b=0.5):
        """
        calculate weight matrix associated a k-nearest neightbors graph
        That is, knn_W(i,j) = kernel(i,j) if j is nearest neighbor of i, else 0
        """
        self.W_knn = np.zeros(shape=(self.n_mods, self.n_obs, self.n_obs))
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over rows of SSM
                knn = np.argsort(self.P[m,i,:])[:self.kN] # knn index nums
                for j in range(self.n_obs): # loop over columns in SSM
                    if autotune:
                        # find kN nearest neighbors of data point i
                        inn = np.array([
                            self.metric(self.mods[i,m], v) \
                            for v in self.mods[:,m]
                            ])
                        inn.sort() # sort ascending; low values are close points
                        # find kN nearest neighbors of data point j
                        jnn = np.array([
                            self.metric(self.mods[j,m], v) \
                            for v in self.mods[:,m]
                            ])
                        jnn.sort() # sort ascending; low values are close points
                        s = autotune_sigma(
                                self.mods[i, m],
                                self.mods[j, m],
                                metric=self.metric,
                                inn=inn[1:self.kN], # exclude self
                                jnn=jnn[1:self.kN], # exclude self
                                b=b
                                )
                    if j in knn:
                        self.W_knn[m,i,j] = self.kern(
                            self.mods[i, m],
                            self.mods[j, m],
                            metric=self.metric,
                            s=s
                            )


    def normalize_knn_weights(self):
        """
        normalize weight matrix associated a k-nearest neightbors graph
        """
        self.P_knn = np.zeros(shape=(self.n_mods, self.n_obs, self.n_obs))
        for m in range(self.n_mods): # loop over modalities
            for m in range(self.n_mods): # loop over modalities
                self.P_knn[m,:,:] = (
                    self.W_knn[m,:,:] /
                    self.W_knn[m,:,:].sum(axis=1).reshape(-1,1)
                    )


    def network_fusion(self, eta=1, iters=50):
        """
        perform random walk over similarity graph

        INPUT:
        eta - regularization term used to avoid the loss of
            self-similarity through the diffusion process and to ensure more
            robust mass is distributed
        iters - number of iterations to perform

        OUTPUT:
        similarity_templates attribute - a 3D numpy array (i.e. tensor) of
        individual similarity matrices

        fused_similarity_template - A 2D numpy array representing the fusion
        of all modalities' similarity templates
        """
        for i in range(iters):
            for m in range(self.n_mods):
                self.P[m,:,:] = (
                    self.P_knn[m,:,:] @
                    ((1/(self.n_mods-1))*tensor_sum_except(self.P[:,:,:], m)) @
                    self.P_knn[m,:,:].T
                    ) + eta*np.eye(self.n_obs)

        self.fused_similarity_template = (
            self.P.sum(axis=0) / self.n_mods
            )# - m * eta*np.eye(self.n_obs) # back out regularization constant


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
                self.P[m, :, :],
                interpolation=interp,
                cmap=cmap)
            plt.title("Fused transition matrix for modality " + str(m))
        if save:
            if path == None: raise ValueError("Must provide save path!")
            plt.savefig(path)
        else:
            plt.show()





if __name__ == "__main__":

    from data_cube import DataCube

    dc = DataCube(
        subjects="all",
        gestures=["1", "2", "3", "4"],
        channels=["2", "4", "5", "6", "8"],
        data_grp="parsed"
        )

    dc.load_data()
    dc.rms_smooth(300, 20)
