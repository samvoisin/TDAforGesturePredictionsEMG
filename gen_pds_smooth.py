####### Generate all persistence diagrams for smoothed gesture modalities
####### One for each channel, within each gesture, within each subject

### import libraries

import numpy as np
import matplotlib.pyplot as plt
import os

from ripser import ripser
from persim import plot_diagrams

from TDA_helper_fcns import sublevel_set_time_series_dist
from data_cube import DataCube
from ssm import SSM

import warnings
warnings.filterwarnings('ignore')

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
    # plot the time series
    plt.subplot(121)
    plt.plot(ts[:, 0], ts[:, 1])
    ax = plt.gca()
    ax.set_yticks(allgrid)
    ax.set_xticks([])
    plt.grid(linewidth=1, linestyle='--')
    plt.xlabel("time (ms)")
    plt.ylabel("Amplitude")

    # plot the persistence diagram
    plt.subplot(122)
    ax = plt.gca()
    ax.set_xticks(births)
    ax.set_yticks(deaths)
    ax.tick_params(labelrotation=45)
    plt.grid(linewidth=1, linestyle='--')
    plot_diagrams(pd["dgms"][0], size=50)
    plt.title("Persistence Diagram")

    if not save_img:
        plt.show()
    elif path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.savefig(os.curdir + "/ts_pd.png")
        plt.close()

### program body ###

if __name__ == "__main__":

    ### generate directory structure ###
    subjs = [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12",
        "13", "14", "15", "16", "17", "18",
        "19", "20", "21", "22", "23", "24",
        "25", "26", "27", "28", "29", "30",
        "31", "32", "33", "34", "35", "36"
        ]
    gests = [
        "1_0_1", "1_1_1", "1_0_2", "1_1_2",
        "2_0_1", "2_1_1", "2_0_2", "2_1_2",
        "3_0_1", "3_1_1", "3_0_2", "3_1_2",
        "4_0_1", "4_1_1", "4_0_2", "4_1_2",
        "5_0_1", "5_1_1", "5_0_2", "5_1_2",
        "6_0_1", "6_1_1", "6_0_2", "6_1_2"
        ]
    to_dir = "./figures/pd_smoothed/"
    for s in subjs:
        for g in gests:
            os.makedirs(to_dir+s+"/"+g, exist_ok=True)

    ### load data ###
    dc = DataCube(
        subjects="all",
        gestures=["1", "2", "3", "4", "5", "6"],
        data_grp="parsed"
    )
    dc.load_data()

    ### smooth modalities ###
    dc.rms_smooth(300, 90)

    for s, gdict in dc.data_set_smooth.items():
        print(f"Subject number {s}")
        for g, a in gdict.items():
            print(f"    Gesture ID {g}")
            r, c = a.shape
            tidx = a[:, 0]
            for m in range(1, c-1):
                # file path for saving image
                fig_pth = to_dir+"/"+s+"/"+g+"/"+"channel_"+str(m+1)+".png"
                sls = sublevel_set_time_series_dist(a[:,m])
                # calculate persistent homology
                pers_diag = ripser(sls, distance_matrix=True)
                # plot and save persistence diagram
                plot_ts_pd(
                    np.c_[tidx, a[:,m]],
                    pers_diag,
                    save_img=True,
                    path=fig_pth
                    )
