################################################################################
######## Generate persistence diagrams for all subjects & all gestures #########
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

from ripser import ripser
from TDA_helper_fcns import load_data


### helper functions ###
def plot_ts_pd(ts, pd):
    """
    plot a time series and associated persistence diagram side by side
    ts - time series to be plotted; a numpy array
    pd - a persistence diagram represented by an array; should be ripser object
    ----
    citation:
    https://ripser.scikit-tda.org/notebooks/Lower%20Star%20Time%20Series.html
    """

    # create persistence diagram axis markers (dashed lines)
    allgrid = np.unique(dgm0.flatten()) # unique persistence markers
    allgrid = allgrid[allgrid < np.inf] # do not mark final cycle
    births = np.unique(dgm0[:, 0]) # unique birth times
    deaths = np.unique(dgm0[:, 1]) # unique death times
    deaths = deaths[deaths < np.inf] # do not mark final cycle

    #Plot the time series and the persistence diagram
    plt.subplot(121)
    plt.plot(t, y)
    ax = plt.gca()
    ax.set_yticks(allgrid)
    ax.set_xticks([])
    #plt.ylim(ylims)
    plt.grid(linewidth=1, linestyle='--')
    plt.title("Subject 5; Gesture 3; Signal 1")
    plt.xlabel("time (ms)")

    plt.subplot(122)
    ax = plt.gca()
    ax.set_xticks(births)
    ax.set_yticks(deaths)
    plt.grid(linewidth=1, linestyle='--')
    plot_diagrams(pd, size=50)
    plt.title("Persistence Diagram")

    plt.show()



# load all data into a single dictionary
all_subjs = load_data()

# iterate through all subjects
for s, gdict in all_subjs.items():
