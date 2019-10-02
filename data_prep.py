###############################################################################
###############################################################################
########### Manipulate structure of data files to seperate gestures ###########
#################### based on gesture and subject numbers #####################
###############################################################################
###############################################################################

def find_break_pts(gesture, t = 1000):
    """find and return index for moments in df w/ > t ms between pt & nxt pt"""
    nrow = gesture.shape[0]
    brk_pts = [i-1 for i in range(2, nrow) if gesture.iloc[i].time_ms - gesture.iloc[i-1].time_ms > t]
    return brk_pts