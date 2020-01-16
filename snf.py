################################################################################
################ Create a Fused Similarity Template Class - SNF ################
###################### SNF class inherits from SSM parent ######################
################################################################################

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

from ssm import SSM, euclidian_dist


def tensor_sum_except(tensor, n):
    """
    sum all matrices in a tensor excluding matrix `n`
    """
    return tensor.sum(axis=0) - tensor[n,:,:]


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


    def calc_masked_transition_matrix(self, k):
        """
        calculate a "masked" transition probability matrix from the SSM. This
        matrix represents a graph where vertices are data points and edges are
        this distances between vertices

        k - int; number of nearest neighbors to a given vertex i (i.e. those js
        with the k largest similarity values in the ith row of the SSM)

        NOTE: trans_mtrx must exist; run `calc_transition_matrix()` method

        Ref: Multiscale Geometric Summaries Similarity-based Sensor Fusion (p.4)
        """
        self.masked_mtrx = np.zeros(shape=(self.n_mods, self.n_obs, self.n_obs))
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over rows in SSM
                knn = np.argsort(-self.array[m,i,:])[:k] # k nrst neighbor index
                for j in range(self.n_obs): # loop over columns in SSM
                    if self.array[m,i,j] in self.array[m,i,:][knn]:
                        self.masked_mtrx[m,i,j] = (
                            self.array[m,i,j] /
                            (2*self.array[m,i,:][knn].sum())
                            )


    def plot_template(
        self,
        m,
        interp='nearest',
        cmap='afmhot',
        figsize=(12,8),
        save=False,
        path=None):
        """
        display self-similarity matrix for modality 'm'
        """
        plt.figure(figsize=figsize)
        plt.imshow(
            self.trans_mtrx[m, :, :],
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
