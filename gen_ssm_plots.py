####### Generate all SSMs for gestures data set
####### One for each channel, within each gesture, within each subject

### import libraries

import numpy as np
import os

from data_cube import DataCube
from ssm import SSM

import matplotlib.pyplot as plt

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
    to_dir = "./figures/ssm_imgs/"
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
    dc.rms_smooth(300, 50)

    ### generate SSM and images ###
    ed = lambda i, j: (i-j)**2 # use euclidian distance

    for s, gdict in dc.data_set_smooth.items():
        print(f"Subject number {s}")
        for g, a in gdict.items():
            print(f"    Gesture ID {g}")
            smtrx = SSM(a[:, :-1], metric=ed) # -1 for removing label column
            smtrx.calc_SSM()
            for m in range(smtrx.n_mods):
                fig_pth = to_dir+"/"+s+"/"+g+"/"+"channel_"+str(m+1)+".png"
                smtrx.plot_SSM(m, save=True, path=fig_pth)
                plt.close("all")
