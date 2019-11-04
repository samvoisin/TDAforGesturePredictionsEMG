################################################################################
####### Support Vector Machine for Testing Gesture/ Subject Separability #######
############ Data set is persistence images - see pim_fullvec_set.py ###########
################################################################################

import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm

pim_df = pd.read_csv("./Data/pim_vectors.csv")

pims = pim_df.values[:, :-2] # subj @ -2
print(pims.shape)
pimcov = pims.T @ pims

# PCA
spect = la.eig(pimcov)
print(sum(spect[0][:2]) / sum(spect[0]))
eigbase = spect[1][:, :2].real

PCApims = pims @ eigbase

pca_df = pd.DataFrame(np.c_[PCApims, pim_df.values[:, -2]])
pca_df.columns = ["V1", "V2", "subj"]

pca_df.plot.scatter(x = "V1", y = "V2", c = "subj")
sns.scatterplot("V1", "V2", hue="subj", data=pca_df)
plt.show()
