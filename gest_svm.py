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
from sklearn.model_selection import StratifiedKFold

pim_df = pd.read_csv("./Data/pim_vectors.csv")

pims = pim_df.values[:, :-2] # subj @ -2; gests @ -1
print(pims.shape)
pimcov = pims.T @ pims

# PCA
spect = la.eig(pimcov)
print(sum(spect[0][:2]) / sum(spect[0]))
eigbase = spect[1][:, :2].real

PCApims = pims @ eigbase

pca_df = pd.DataFrame(np.c_[PCApims, pim_df.values[:, -1]])
pca_df.columns = ["V1", "V2", "subj"]
pca_df.subj = pca_df.subj.astype("category")

sns.scatterplot("V1", "V2", hue="subj", data=pca_df)
plt.show()

################################# Fitting SVM ##################################

X = pim_df.values[:, :-2]
y = pim_df.values[:, -1]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
skf.get_n_splits(X, y)

clf = svm.SVC()

acc_scores = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    acc_scores.append(clf.score(X_test, y_test))
