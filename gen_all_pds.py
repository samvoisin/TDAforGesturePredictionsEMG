################################################################################
######## Generate persistence diagrams for all subjects & all gestures #########
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

from ripser import ripser
from persim import plot_diagrams

from TDA_helper_fcns import load_data


### helper functions ###
def plot_ts_pd(ts, pd, figsize=(12, 6), save_img=False, path=None):
    """
    plot a time series and associated persistence diagram side by side
    ts - time series to be plotted; numpy array - col 0 is time; col 1 is data
    pd - a persistence diagram represented by an array; ripser object
    figsize - default size of plot figure
    save_img - bool; indicate whether or not to save image file
    path - where to save image file; defaults to working dir if None
    ----
    citation:
    https://ripser.scikit-tda.org/notebooks/Lower%20Star%20Time%20Series.html
    """

    # create persistence diagram axis markers (dashed lines)
    allgrid = np.unique(pd["dgms"][0].flatten()) # unique persistence pts
    allgrid = allgrid[allgrid < np.inf] # do not mark final cycle
    births = np.unique(pd["dgms"][0][:, 0]) # unique birth times
    deaths = np.unique(pd["dgms"][0][:, 1]) # unique death times
    deaths = deaths[deaths < np.inf] # do not mark final cycle

    plt.figure(figsize=figsize)
    # Plot the time series
    plt.subplot(121)
    plt.plot(ts[:, 0], ts[:, 1])
    ax = plt.gca()
    ax.set_yticks(allgrid)
    #ax.set_xticks([])
    plt.grid(linewidth=1, linestyle='--')
    plt.xlabel("time (ms)")
    plt.ylabel("kHz")

    # Plot the persistence diagram
    plt.subplot(122)
    ax = plt.gca()
    ax.set_xticks(births)
    ax.set_yticks(deaths)
    plt.grid(linewidth=1, linestyle='--')
    plot_diagrams(pd["dgms"][0], size=50)
    plt.title("Persistence Diagram")

    if not save_img:
        plt.show()
    elif path is not None:
        plt.savefig(path)
    else:
        plt.savefig(os.curdir + "/ts_pd.png")



if __name__ == "__main__":
    # load all data into a single dictionary
    all_subjs = load_data()

    # iterate through all subjects
    #for s, gdict in all_subjs.items():
