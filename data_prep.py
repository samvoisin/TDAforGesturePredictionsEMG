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

def gest_dict(filepath):
    """return dict where key is gesture # and
    value is indiv DataFrames for that gesture"""
    df = pd.read_table(filepath)
    df = mod_col_names(df)
    gests = df.gesture.unique()
    all_gst = dict()
    # seperate gestures in file
    for g in gests:
        # split single gesture df into iterations of gesture g
        brk_pts = find_break_pts(df[df.gesture == g])
        gd = dict() # store sub-dfs created by breaks
        trail = 0 
        for n, b in enumerate(brk_pts):
            if g == 0:
                # treat gest == 0 seperately
                # due to many cases btwn non-0 gests
                gd[n] = df[df.gesture == 0].iloc[trail:b]
                gd[n] = zero_time(gd[n])
                train = b + 1
            else:
                # this will handle all other cases (gesture != 0)
                gd[0] = zero_time(df[df.gesture == g].iloc[:b])
                gd[1] = zero_time(df[df.gesture == g].iloc[b+1:])
            all_gst[g] = gd
    
    return all_gst

def build_new_ref(to_dir, sn, orig_file, g, ct):
    """build a new file path based on attributes of data to be saved; naming convention:
    to_dir - top level dir for cleaned data
    sn - subject number
    orig_file - file name with raw data; 2 per subject
    g - gesture number
    ct - counter for files w/ similar names (e.g. gesture 2, 0 & 1)
    """
    if orig_file.startswith("1"):
        new_ref = to_dir + sn + "/1/" + g + "_" + ct
    else:
        new_ref = to_dir + sn + "/2/" + g + "_" + ct
    return new_ref

### script body

# path to current/ raw data
raw_dir = "./Data/EMG_data_for_gestures-master/"
subj_nums = os.listdir(raw_dir)
subj_data = {n : os.listdir(raw_dir + n) for n in subj_nums}

# define new file path and create new directory structure
to_dir = "./Data/EMG_data_for_gestures-cleaned/"
for n in subj_nums:
    os.makedirs(to_dir + n) # create new dir

# generate new data files
for n, v in subj_data.items():
    # origin file 1 or 2 for each subject
    for o_f in (0, 1):
        cln_gests = gest_dict(raw_dir + n + "/" + v[o_f])
        # parse data file seperating frames and saving them to new dir
        for g, d in cln_gests.items():
            ct = 0 # counter for new file names
            ### need another level here for d - see jupyter notebook
            clean_ref = build_new_ref(to_dir, n, v[o_f], g, ct)


