################################################################################
############ Manipulate structure of data files to rejoin gestures #############
#################### based on gesture and subject numbers ######################
################################################################################

### import modules
import os
import numpy as np
import pandas as pd

from data_cube import DataCube

pars_dir = "./Data/EMG_data_for_gestures-parsed/"
subj_nums = os.listdir(pars_dir)
subj_data = {s : os.listdir(pars_dir+s) for s in subj_nums}

to_dir = "./Data/EMG_data_for_gestures-no_zero/"
for s in subj_nums:
    os.makedirs(to_dir+s, exist_ok=True) # create new dir

### load parsed data set into DataCube
dc = DataCube(subjects="all",
              gestures="all",
              channels="all",
              data_grp="parsed")
dc.load_data()

cols = ["time", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "gest"]

r = len(dc.data_set.keys())
i = 0
for s, gdict in dc.data_set.items():
    ### progress bar ###
    pb = "~"*int(i/r*100)+" "*int((1-i/r)*100)+"|"
    print(pb, end="\r")
    ####################
    strt1 = np.zeros(10).reshape(1, 10) # file 1 starter array
    strt2 = np.zeros(10).reshape(1, 10) # file 2 starter array
    z = np.zeros(10).reshape(1, 10) # boundary array
    for g, a in gdict.items(): # gesture and array
        if g[-1] == "1":
            a[:, 0] = a[:, 0]+strt1[-1, 0] # update time index
            strt1 = np.r_[z]
            strt1 = np.r_[a]
        else:
            a[:, 0] = a[:, 0]+strt2[-1, 0] # update time index
            strt2 = np.r_[z]
            strt2 = np.r_[a]
        f1 = pd.DataFrame(strt1)
        f2 = pd.DataFrame(strt2)
        f1.columns = cols
        f2.columns = cols
        f1.to_csv(to_dir+s+"/"+"f1.csv", index=False)
        f2.to_csv(to_dir+s+"/"+"f2.csv", index=False)
    i += 1
