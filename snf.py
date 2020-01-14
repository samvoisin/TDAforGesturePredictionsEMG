################################################################################
################ Create a Fused Similarity Template Class - SNF ################
###################### SNF class inherits from SSM parent ######################
################################################################################

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

from ssm import SSM, euclidian_dist, sim_kern

class SNF(SSM):
    """
    Similarity network fusion (SNF) fused similarity template class
    SSM is the parent class; and SSM object
    """
    def __init__(self, time_series, metric=euclidian_dist):
        # inherit methods and properties from parent
        super().__init__(time_series, metric)

    def calc_sim_matrix(self, s=1, kern=sim_kern):
        """
        convert self-similarity matrix (SSM)
        SSM is a distance matrix under self.metric
        """
        self.reset_array()
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over observations in m
                for j in range(self.n_obs):
                    if i < j: # fill lower triangle only
                        self.array[m, i, j] = kern(
                            self.mods[i, m],
                            self.mods[j, m],
                            m=self.metric,
                            s=s
                            )
            self.array[m, :, :] = self.array[m, :, :] + self.array[m, :, :].T
            self.array[m, :, :] += np.diag(np.ones(self.n_obs)) # main diagonal


if __name__ == "__main__":
    pass
