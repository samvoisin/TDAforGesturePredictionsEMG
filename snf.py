################################################################################
################ Create a Fused Similarity Template Class - SNF ################
###################### SNF class inherits from SSM parent ######################
################################################################################

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

from ssm import SSM, euclidian_dist


def tensor_sum_except(tensor, n, axs=0):
    """
    sum all matrices in a tensor excluding matrix `n`
    axs argument determines axis over which sum is calculated; default axis 0
    """
    return tensor.sum(axis=axs) - tensor[n,:,:]


class SNF(SSM):
    """
    similarity network fusion (SNF) fused similarity template class inheriting
    from the `SSM` parent class
    """
    def __init__(self, time_series, metric=euclidian_dist):
        # inherit methods and properties from parent
        super().__init__(time_series, metric)


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


    def calc_similarity_graph(self, k):
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
        k = int(k * self.n_obs) # set k to an integer; this truncates
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over rows of SSM
                knn = np.argsort(-self.array[m,i,:])[:k] # k nrst neighbor index
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
            plt.title("Transition matrix for modality " + str(m))
        if save:
            if path == None: raise ValueError("Must provide save path!")
            plt.savefig(path)
        else:
            plt.show()





if __name__ == "__main__":
    pass
