################################################################################
################################ DataCube class ################################
################## designed for loading and managing data set ##################
################################################################################

import numpy as np
import os


class DataCube:
    """
    DataCube class designed for loading and managing data set
    """

    def __init__(self, subjects="all", gestures="all", data_grp="parsed"):
        """
        load data group from master (i.e. raw) or parsed set
        if subject number is specified [list type] load only that subject(s)
        if gesture is specified [list type] load only that (those) gesture(s)
        """
        self.data_grp = data_grp
        self.subj_lvl_dir = "./Data/EMG_data_for_gestures-"+data_grp+"/"
        self.loaded_flg = False # boolean flag for whether data is loaded
        self.data_set = {}
        # specify subjects
        if subjects == "all":
            self.subjects = os.listdir(self.subj_lvl_dir)
        else:
            self.subjects = subjects
        # specify gestures
        if gestures == "all": # no gesture 0; must specify
            self.gestures = ["1", "2", "3", "4", "5", "6"]
        else:
            self.gestures = gestures


    def load_data(self):
        """
        load data set from master (i.e. raw) or parsed set
        if subject number is specified [list type] load only that subject(s)
        if gesture is specified [list type] load only that (those) gesture(s)
        """
        if self.data_grp == "parsed":
            # generate data dict {subject : {gesture : array}}
            for s in self.subjects:
                self.data_set[s] = {}
                dir_root = self.subj_lvl_dir+s+"/" # directory root
                for f in os.listdir(dir_root):
                    if f[0] in self.gestures: # if a file exists in directory
                        with open(dir_root+f, "r") as fh:
                            # f[0:5] is gest_performance(0 or 1)_file(1 or 2)
                            self.data_set[s][f[0:5]] = np.loadtxt(
                                fh,
                                delimiter=",",
                                skiprows=1)
            self.gest_aliases = self.data_set[s].keys()

        elif self.data_grp == "master":
            for s in self.subjects:
                self.data_set[s] = {}
                dir_root = self.subj_lvl_dir+s+"/" # directory root
                for f in os.listdir(dir_root):
                    with open(dir_root+f, "r") as fh:
                        self.data_set[s][f] = np.loadtxt(
                        fh,
                        skiprows=1
                        )


        self.loaded_flg = True # set loaded flag to True







if __name__ == "__main__":
    dc = DataCube(subjects=["10", "20"], gestures="all", data_grp="master")
    dc.load_data()
    for k, v in dc.data_set.items():
        print(k)
        for f, d in v.items():
            print(f)
            print(d)
