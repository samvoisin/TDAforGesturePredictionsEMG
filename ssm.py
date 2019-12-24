import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.interpolate import griddata






# Create SSMs

class SSM:

    def __init__(self, time_series, metric):
        """
        CURRENTLY SUPPORTS SCALARS
        time_series - (t x 1+m) numpy array; t is time series
            1+m is time index + number of modalities
        metric - metric to be used in generating SSMS (can be similarity kernel)
        """
        self.tidx = time_series[:, 0].astype("int32") # time index
        self.mods = time_series[:, 1:] # modalities
        self.metric = metric
        self.n_obs = self.tidx.size # number of obs
        self.n_mods = self.mods.shape[1] # number of modalitites
        self.array = np.zeros(shape=(self.n_mods, self.n_obs, self.n_obs))


    def calc_SSM(self):
        """
        calculate SSM
        """
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over observations in m
                for j in range(self.n_obs):
                    if i < j: # fill lower triangle only
                        self.array[m, i, j] = self.metric(
                            self.mods[i, m],
                            self.mods[j, m]
                            )
            self.array[m, :, :] = self.array[m, :, :] + self.array[m, :, :].T


if __name__ == "__main__":
    pass
