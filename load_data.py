
###############################################################################
###############################################################################
###### Helper function to load specified data sets as nested dictionary #######
###############################################################################
###############################################################################


import numpy as np
import os


def load_data(subjects = "all", gestures = "all", dataset = "parsed"):
    """load data set from master (i.e. raw) or parsed set
    if subject number is specified [list type] load just that (those) subject(s)
    if gesture is specified [list type] load just that (those) gesture number(s)"""

    subj_lvl_dir = "./Data/EMG_data_for_gestures-" + dataset + "/"

    if subjects == "all":
        subjs = os.listdir(subj_lvl_dir)
    else:
        subjs = subjects

    if gestures == "all":
        # does not include 0 gesture; must specify
        gests = ["1", "2", "3", "4", "5", "6"]
    else:
        gests = gestures

    dat = {}
    # generate data sict subject : {gesture : array}
    for s in subjs:
        dat[s] = {}
        dir_root = subj_lvl_dir + s + "/"
        for f in os.listdir(dir_root):
            if f[0] in gests:
                with open(dir_root + f, "r") as fh:
                    dat[s][f[0:3]] = np.loadtxt(fh, delimiter = ",", skiprows = 1)


    return dat

