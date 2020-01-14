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


    def calc_transition_matrix(self):
        """
        calculate a transition probability matrix from the SSM
        each row in the new array must sum to 1.
        """
        trans_mat = np.zeros(shape=(self.n_mods, self.n_obs, self.n_obs))
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over observations in m
                for j in range(self.n_obs):
                    if i != j:



if __name__ == "__main__":
    pass
