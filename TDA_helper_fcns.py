###############################################################################
####### Helper functions for data manipulation, access, plotting, etc. ########
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import os

from scipy import sparse

from ripser import ripser, Rips
from persim import bottleneck, bottleneck_matching


def load_data(subjects="all", gestures="all", dataset="parsed"):
    """
    load data set from master (i.e. raw) or parsed set
    if subject number is specified [list type] load only that subject(s)
    if gesture is specified [list type] load only that (those) gesture(s)
    """

    subj_lvl_dir = "./Data/EMG_data_for_gestures-"+dataset+"/"

    # specificy subjects
    if subjects == "all":
        subjs = os.listdir(subj_lvl_dir)
    else:
        subjs = subjects

    # specificy gestures
    if gestures == "all":
        # does not include 0 gesture; must specify
        gests = ["1", "2", "3", "4", "5", "6"]
    else:
        gests = gestures

    dat = {}
    # generate data dict subject : {gesture : array}
    for s in subjs:
        dat[s] = {}
        dir_root = subj_lvl_dir+s+"/"
        for f in os.listdir(dir_root):
            if f[0] in gests:
                with open(dir_root+f, "r") as fh:
                    # f[0:5] designates gest_performance(0 or 1)_file(1 or 2)
                    dat[s][f[0:5]] = np.loadtxt(fh, delimiter=",", skiprows=1)

    return dat


def get_max_perf_time(gdat):
    """
    find the maximum time required to perform each gesture for a
    given set of subjects

    INPUTS
    gdat is gestures data set as returned by `load_data` function

    OUTPUTS
    dictionary w/ gesture number (int) : maximum performance time (int)
    """
    sdict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0} # dictionary to be returned w/ keys
    for k, v in gdat.items(): # subject, gesture set
        for g, u in v.items(): # gesture ("1_0_1", "4_1_2"), array
            # last value in array denotes gesture being performed (i.e. [1, 6])
            # this is key in sdict
            gid = int(u[-1, -1])
            # update if current nrows in current u is greater than stored value
            if u.shape[0] > sdict[gid]:
                sdict[gid] = u.shape[0]
    return sdict


def plot_gests(subj, g, subj_dict, signals=range(1,9), save=False, path=None):
    """
    Example input: plot_gests("30", "3_1_2", thrty, signals=[1, 2, 3, 5, 8])
    create plots of data for a given subject (subj) - type == str
    and gesture (g) - array in subject dict (e.g. (3_0, 3_1, 6_1. etc.))
    subj_dict - dict containing data from one more more subj (key == subj #)
    signals - specify which signals; default is all
    save gestures to file path tbd
    """
    # fix list of available colors
    colors = ('blue','green','crimson',
              'purple', 'black', 'orange',
              'firebrick', 'gold','forestgreen')

    ### single plot code ###
    # if one signal specified no subplots necessary
    if type(signals) == int or len(signals) == 1:
        plt.plot(
        subj_dict[subj][g][:, 0],
        subj_dict[subj][g][:, signals]
        )
        plt.title("Subject "+subj+"; Gesture "+g+"; Channel "+str(signals))
        plt.xlabel("ms")
        plt.ylabel("Amplitude")
        return

    ### subplots code ###
    n_sig = len(signals)
    # 4 or fewer signals needs 1 col only
    if n_sig <= 4:
        fig, ax = plt.subplots(ncols=1, nrows=n_sig, sharex=True)

        clr = 0 # color and signal selector
        for n, i in enumerate(signals):
            ax[n].set_title("Channel "+str(i))
            ax[n].plot(
                subj_dict[subj][g][:, 0],
                subj_dict[subj][g][:, i],
                color=colors[clr]
                )
            clr += 1
            if clr == n_sig:
                fig.suptitle("Subject "+subj+"; Gesture "+g)
                # return subplots for <= 4 signals
                return

	# 5 or more signals gets 2 columns
    n_sbplts = n_sig
    if n_sbplts%2 != 0: n_sbplts += 1
    n_r = n_sbplts//2
    n_c = 2 # always 2 cols

    fig, ax = plt.subplots(ncols=n_c, nrows=n_r, sharex=True)

    clr = 0
    for i in range(n_r):
        for j in range(n_c):
            ax[i, j].set_title("Channel "+str(signals[clr]))
            ax[i, j].plot(
                subj_dict[subj][g][:, 0],
                subj_dict[subj][g][:, signals[clr]],
                color=colors[clr]
                )
            clr += 1
            if clr >= n_sig:
                fig.suptitle("Subject "+subj+"; Gesture "+g)
                # return subplots for > 4 signals
                return


def sublevel_set_time_series_dist(x):
    """
    Get sublevel set filtration for a time series
    x - a numpy array s.t. np.ndim(x) = 1
    returns (n x n) scipy.sparse distance matrix
    function adapted from scikit-tda tutorials:
    ----
    code adapted from citation:
    https://ripser.scikit-tda.org/notebooks/Lower%20Star%20Time%20Series.html
    """
    n = x.size
    # Add edges between adjacent points in the time series, with the "distance"
    # along the edge equal to the max value of the points it connects
    i = np.arange(n-1)
    j = np.arange(1, n)
    # find max(i, i+1) over entire array
    max_compare = np.maximum(x[0:-1], x[1::])
    # Add vertex birth times along the diagonal of the distance matrix
    i = np.concatenate((i, np.arange(n)))
    j = np.concatenate((j, np.arange(n)))
    max_compare = np.concatenate((max_compare, x))
    #Create the sparse distance matrix
    dist_mat = sparse.coo_matrix((max_compare, (i, j)), shape=(n, n)).tocsr()
    return dist_mat


def bottleneck_dist_mat(gdat, verbose=True):
    """
    Generate distance matrix for persistence diagrams of images

    INPUTS
    gdat - gestures data matrix; use output of load_data

    OUTPUTS
    distance matrix using bottleneck metric
    """
    ## code below taken from gen_all_pds.py - look there for os cmds and saving
    # iterate through all subjects
    pd_dict = dict()

    for sbj, sdict in gdat.items():
        # Dictionary of each subject with all gestures
        for gnum, garray in sdict.items():
            # loop through each signal in the gesture
            t_axis = garray[:, 0] # time data
            for s in range(1, garray.shape[1]-1):
                # sublevel set filtraton
                sls = sublevel_set_time_series_dist(garray[:,s])
                # generate persistence diagram
                pd = ripser(sls, distance_matrix=True)
                # remove inf persistence point
                pd["dgms"][0] = pd["dgms"][0][np.isfinite(pd["dgms"][0][:,1]),:]
                pd_dict[sbj+"_"+gnum] = pd

    # ordered list of keys
    klist = [k for k in pd_dict.keys()]
    klist.sort() # ordered ascending by subject, gesture

    # initialize bottleneck distance matrix
    bd_mat = np.zeros(len(klist)**2).reshape(len(klist), len(klist))

    for n, k in enumerate(klist):
        if verbose:
            ### progress bar ###
            pb = "~"*(int(n/len(klist)*100))+" "*(int((1-n/len(klist))*100))+"|"
            print(pb, end="\r")
            ####################
        for m, j in enumerate(klist):
            if n == m: bd_mat[n, m] = 0.0
            else: bd_mat[n, m] = bottleneck(
                pd_dict[k]["dgms"][0],
                pd_dict[j]["dgms"][0]
                )

    return bd_mat





if __name__ == "__main__":
    thrty=load_data(subjects=["30"])
    plot_gests("30", "3_1_2", thrty)

    plt.show()
