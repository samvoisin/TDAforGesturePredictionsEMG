################################################################################
################################ DataCube class ################################
################## designed for loading and managing data set ##################
################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


def root_mean_sq(a):
    """
    calculate root mean squared of array a
    """
    return (sum(a**2)/a.size)**(0.5)


class DataCube:
    """
    DataCube class designed for loading and managing data set
    """

    def __init__(
        self,
        subjects="all",
        gestures="all",
        channels="all",
        data_grp="parsed"):
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
        # specify channels
        if channels == "all":
            self.channels = [i for i in range(1,9)]
        else:
            self.channels = [int(i) for i in channels]


    def load_data(self):
        """
        load data set from master (i.e. raw) or parsed set
        if subject number is specified [list type] load only that subject(s)
        if gesture is specified [list type] load only that (those) gesture(s)
        """
        if self.data_grp == "parsed" or self.data_grp == "subsample":
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
                            # remove unspecified channels
                            rm_ch = [
                                i for i in range(1,9) if i not in self.channels
                                ]
                            rm_ch.reverse() # rev removal channels for del order
                            for c in rm_ch:
                                # delete c^th column of array
                                self.data_set[s][f[0:5]] = (
                                    np.delete(self.data_set[s][f[0:5]], c, 1)
                                    )
            #self.gest_aliases = self.data_set[s].keys()

        elif self.data_grp == "master":
            for s in self.subjects:
                self.data_set[s] = {}
                dir_root = self.subj_lvl_dir+s+"/" # directory root
                for f in os.listdir(dir_root):
                    with open(dir_root+f, "r") as fh:
                        # f[0] is file num (1 or 2)
                        self.data_set[s][f[0]] = np.loadtxt(fh, skiprows=1)
                        # remove unspecified channels
                        rm_ch = [i for i in range(1,9) if i not in self.channels]
                        rm_ch.reverse() # rev removal channels for del order
                        for c in rm_ch:
                            # delete c^th column of array
                            self.data_set[s][f[0]] = (
                                np.delete(self.data_set[s][f[0]], c, 1)
                                )
        # set loaded flag to True
        self.loaded_flg = True



    def rms_smooth(self, N, stp):
        """
        Perform root-mean-squares smoothing on the data set
        This creates data_set_smooth attribute
        N - number of samples in time window
        stp - step size
        """
        self.data_set_smooth = {} # initialize empty data set attribute
        for subj, gdict in self.data_set.items():
            self.data_set_smooth[subj] = {}
            for g, a in gdict.items(): # for each array in the gesture g
                nr, nc = a.shape
                n_slides = (a[:, 1].size - (N - stp)) / stp # num windows
                ### initialize sliding window variables ###
                res_sz = int(n_slides) # truncate num of slides for result size
                res = np.zeros(shape=(res_sz, nc))
                s = 0 # window start
                e = N # window end
                for n, v in enumerate(res):
                    v[0] = a[int(e-s/2), 0]
                    v[1:] = np.apply_along_axis(root_mean_sq, 0, a[s:e, 1:])
                    s += stp
                    e += stp
                self.data_set_smooth[subj][g] = res


    def normalize_modalities(self, smooth=False):
        """
        Normalize and scale all modalities

        smooth - boolean; used self.data_set_smooth if True
        """
        if smooth:
            set = self.data_set_smooth
        else:
            set = self.data_set

        for s, gdict in set.items():
            for g, array in gdict.items():
                set[s][g] = np.c_[
                    array[:, 0],
                    scale(array[:, 1:-1], axis=0),
                    array[:, -1]
                    ]

        if smooth:
            self.data_set_smooth = set
        else:
            self.data_set = set


    def get_max_obs(self, smooth=False):
        """
        find maximum number of observations in loaded data set
        to be used for interpolation
        """
        self.max_obs = 0
        if smooth:
            for s, gdict in self.data_set_smooth.items():
                for g, a in gdict.items():
                    if self.max_obs < a[:, 0].size: # if current max < nrow
                        self.max_obs = a[:, 0].size # update current max
        else:
            for s, gdict in self.data_set.items():
                for g, a in gdict.items():
                    if self.max_obs < a[:, 0].size: # if current max < nrow
                        self.max_obs = a[:, 0].size # update current max


    def get_min_obs(self, smoothed=False):
        """
        find maximum number of observations in loaded data set
        to be used for interpolation
        """
        self.min_obs = np.inf
        for s, gdict in self.data_set.items():
            for g, a in gdict.items():
                if self.min_obs > a[:, 0].size: # if current min > nrow
                    self.min_obs = a[:, 0].size # update current min


    def plot_gests(self, subj, g, chans="all", save=False, path=None):
        """
        Example input: plot_gests("30", "3_1_2", thrty, chans=[1, 2, 3, 5, 8])
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

        if chans == "all":
            chans = self.channels
        else:
            chans = [int(i) for i in chans]

        time_ser = self.data_set[subj][g][:, 0]

        ### single plot code ###
        # if one signal specified no subplots necessary
        if type(chans) == int or len(chans) == 1:
            plt.plot(
            time_ser,
            self.data_set[subj][g][:, chans]
            )
            plt.title("Subject "+subj+"; Gesture "+g+"; Channel "+str(chans))
            plt.xlabel("ms")
            plt.ylabel("Amplitude")
            plt.show()

        ### subplots code ###
        n_chan = len(chans)
        # 4 or fewer signals needs 1 col only
        if n_chan <= 4:
            fig, ax = plt.subplots(ncols=1, nrows=n_sig, sharex=True)

            clr = 0 # color and signal selector
            for n, i in enumerate(chans):
                ax[n].set_title("Channel "+str(i))
                ax[n].plot(
                    time_ser,
                    self.data_set[subj][g][:, i],
                    color=colors[clr]
                    )
                clr += 1
                if clr == n_chan:
                    fig.suptitle("Subject "+subj+"; Gesture "+g)
                    # return subplots for <= 4 signals
                    plt.show()

    	# 5 or more signals gets 2 columns
        n_sbplts = n_chan
        if n_sbplts%2 != 0: n_sbplts += 1
        n_r = n_sbplts//2
        n_c = 2 # always 2 cols

        fig, ax = plt.subplots(ncols=n_c, nrows=n_r, sharex=True)

        clr = 0
        for i in range(n_r):
            for j in range(n_c):
                ax[i, j].set_title("Channel "+str(chans[clr]))
                ax[i, j].plot(
                    time_ser,
                    self.data_set[subj][g][:, clr+1],
                    color=colors[clr]
                    )
                clr += 1
                if clr >= n_chan:
                    break
        fig.suptitle("Subject "+subj+"; Gesture "+g)
        # return subplots for > 4 signals
        if save:
            plt.savefig(path)
        else:
            plt.show()


if __name__ == "__main__":
    dc = DataCube(subjects=["10", "20"], gestures="all", data_grp="parsed")
    dc.load_data()
