import numpy as np
import pandas as pd
import pickle

from persim import PersImage

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


pimsd = 1e-5
px = 20
pim = PersImage(pixels=[px,px], spread=pimsd)


# code to load model
with open("log_reg_skl.sav", "rb") as fh:
   log_reg = pickle.load(fh)


inverse_image = np.copy(log_reg.coef_).reshape(-1, px)
for i in range(6):
    pim.show(inverse_image[i*px:(i+1)*px, :])
    plt.title("Inverse Persistence Image for Gesture: " + str(i+1))
    plt.show()
