###############################################################################
####### Helper functions for data manipulation, access, plotting, etc. ########
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import os


def load_data(subjects="all", gestures="all", dataset="parsed"):
    """load data set from master (i.e. raw) or parsed set
    if subject number is specified [list type] load only that subject(s)
    if gesture is specified [list type] load only that (those) gesture(s)"""

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





def plot_gests(subj, g, subj_dict, signals=range(1,9), save=False, path=None):
    """
    create plots of data for a given subject (subj) - type == str
    and gesture (g) - array in subject dict (e.g. (3_0, 3_1, 6_1. etc.))
    subj_dict is dictionary containing data from one more more subj (key == subj #)
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
        plt.plot(subj_dict[subj][g][:, 0], subj_dict[subj][g][:, signals])
        plt.title("Subject "+subj+"; Gesture "+g+"; Signal "+str(signals))
        return

    ### subplots code ###

    n_sig = len(signals)

    # 4 or fewer signals needs 1 col only
    if n_sig <= 4:
        fig, ax = plt.subplots(ncols=1, nrows=n_sig, sharex=True)
        clr = 0 # color and signal selector
        for n, i in enumerate(signals):
            ax[n].set_title("Signal Number "+str(i))
            ax[n].plot(
                subj_dict[subj][g][:, 0],
                subj_dict[subj][g][:, i],
                color=colors[clr]
                )
            clr += 1
            if clr == n_sig:
                fig.suptitle("Subject "+subj+"; Gesture "+g)
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
            ax[i, j].set_title("Signal Number "+str(signals[clr]))
            ax[i, j].plot(
                subj_dict[subj][g][:, 0],
                subj_dict[subj][g][:, signals[clr]],
                color=colors[clr]
                )
            clr += 1
            if clr >= n_sig:
                fig.suptitle("Subject "+subj+"; Gesture "+g)
                return





if __name__ == "__main__":
    thrty=load_data(subjects=["30"])
    plot_gests("30", "3_1_2", thrty, signals=[5, 3, 1])


plt.show()
