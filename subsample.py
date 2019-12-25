################################################################################
######### take subset of data points representing changes in amplitude #########
################################################################################

import numpy as np
import pandas as pd
import os
from data_cube import DataCube


def grow_array(a, axis=0):
    """
    Increase size of array by doubling on specified axis
    doubling this way gives O(log n) amortized time

    INPUTS
    a - input array of type np.array; must be 2 dimensional
    axis - is axis on which to double
    """
    r, c = a.shape
    blnk = np.zeros(shape=(r,c)) # blank array
    if axis == 0:
        res = np.r_[a, blnk]
        return res
    else:
        res = np.c_[a, blnk]
        return res


def trim_array(a, axis=0):
    """
    Trim zero rows from the bottom of an array

    INPUTS
    a - input array of type np.array; must be 2 dimensional
    axis - is axis on which to double
    """
    r, c = a.shape
    for n, i in enumerate(a):
        if all(i == np.zeros(c)):
            return a[:n, :]
        else:
            return a


def subsample(a):
    """
    subsample array removing rows with entries identical to the previous row
    """
    sf = a[0, :].reshape(1, -1) # set first row to be reference row
    c = 0
    for n, r in enumerate(a):
        # if all entries in a row are equal to reference row, skip that row
        if all(sf[c, 1:] == r[1:]):
            continue
        else: # else insert row into new array
            c += 1
            if sf.shape[0] <= c: # dynamically allocate space where needed
                sf = grow_array(sf)
            sf[c, :] = r # set new reference row

    sf = trim_array(sf) # trim hanging zeros

    return sf


if __name__ == "__main__":
    dc = DataCube(
        subjects="all",
        gestures="all",
        channels=["2", "4", "5", "6", "8"],
        data_grp="parsed"
        )
    dc.load_data()

    to_dir = "./Data/EMG_data_for_gestures-subsample/"

    for s, gdict in dc.data_set.items():
        os.makedirs(to_dir + s, exist_ok=True) # create new dir
        for g, a in gdict.items():
            adf = pd.DataFrame(subsample(a))
            ss_ref = to_dir + s + "/" + g + ".csv"
            adf.to_csv(ss_ref, index=False, sep=",")
