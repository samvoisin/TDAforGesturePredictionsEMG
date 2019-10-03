###############################################################################
########### Manipulate structure of data files to seperate gestures ###########
#################### based on gesture and subject numbers #####################
###############################################################################

### import modules

import os
import pandas as pd

### define helper functions

def mod_col_names(full_frame):
    """modify column names of full_frame prior to any other manipulation"""
    # rename "time" variable to "time_ms"
    # rename "class" variable to "gesture"
    colnames = list(full_frame.columns)
    colnames[0] = "time_ms"
    colnames[-1] = "gesture"
    full_frame.columns = colnames
    return full_frame

def find_break_pts(gesture, t = 1000):
    """find and return index for moments in df w/ > t ms between pt & nxt pt"""
    nrow = gesture.shape[0]
    brk_pts = [
        i-1 for i in range(2, nrow)
        if gesture.iloc[i].time_ms - gesture.iloc[i-1].time_ms > t
        ]
    return brk_pts

def zero_time(seg_df):
    """take prev segmented DataFrame by gest; set time_ms to start at 0"""
    end_time = min(seg_df.time_ms)
    seg_df.time_ms = seg_df.time_ms - end_time
    return seg_df


### global variables

# path to current/ raw data
data_dir = "./Data/EMG_data_for_gestures-master/"
subj_nums = os.listdir(data_dir)
subj_data = {n : os.listdir(data_dir + n) for n in subj_nums}


### script body

# define new file path and create new directory structure
to_dir = "./Data/EMG_data_for_gestures-cleaned/"
for n in subj_nums:
    os.makedirs(to_dir + n)