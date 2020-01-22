import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


def euclidian_dist(i, j):
    """
    euclidian distance for scalar values
    """
    return abs(i-j) # abs equivalent to ((i-j)**2)**0.5 in scalar case


def gaussian(i, j, metric, s=1):
    """
    gaussian similarity kernel function
    i, j - data points between which similarity is measured
    metric - a metric
    s - variance parameter
    """
    return np.exp(-(m(i,j)**2)/s)


# Create SSMs
class SSM:

    def __init__(self, time_series, metric=euclidian_dist):
        """
        CURRENTLY SUPPORTS SCALARS
        time_series - a (t x 1+m) numpy array object; t is time series
            1+m is time index + number of modalities
        metric - function to be used in generating distance matrix SSMs
        """
        self.tidx = time_series[:, 0].astype("int32") # time index
        self.mods = time_series[:, 1:] # modalities
        self.metric = metric
        self.n_obs = self.tidx.size # number of obs
        self.n_mods = self.mods.shape[1] # number of modalitites
        self.array = np.zeros(shape=(self.n_mods, self.n_obs, self.n_obs))


    def normalize_modalities(self):
        """
        Normalize and scale modalities in array
        """
        self.mods = scale(self.mods, axis=0)


    def reset_array(self):
        """
        reset array containing SSM or similarity matrix (i.e. self.array)
        to a tensor of zeros
        """
        self.array = np.zeros(shape=(self.n_mods, self.n_obs, self.n_obs))


    def calc_SSM(self):
        """
        calculate self-similarity matrix (SSM)
        SSM is a distance matrix under self.metric
        """
        self.reset_array()
        for m in range(self.n_mods): # loop over modalities
            for i in range(self.n_obs): # loop over observations in m
                for j in range(self.n_obs):
                    if i < j: # fill lower triangle only
                        self.array[m, i, j] = self.metric(
                            self.mods[i, m],
                            self.mods[j, m]
                            )
            self.array[m, :, :] = self.array[m, :, :] + self.array[m, :, :].T


    def plot_SSM(
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
        plt.subplot(121)
        plt.plot(self.tidx, self.mods[:, m])
        plt.title("Modality " + str(m))
        plt.subplot(122)
        plt.imshow(
            self.array[m, :, :],
            interpolation=interp,
            cmap=cmap)
        plt.title("SSM for modality " + str(m))
        if save:
            if path == None: raise ValueError("Must provide save path!")
            plt.savefig(path)
        else:
            plt.show()


if __name__ == "__main__":
    pass
