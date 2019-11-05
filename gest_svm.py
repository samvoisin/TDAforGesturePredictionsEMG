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

pim_df = pd.read_csv("./pim_vectors.csv")
pim_df.columns = [i for i in pim_df.columns][:-2] + ["subj", "gest"] # oops

####################### Visualizing Priciple Components ########################

pims = pim_df.iloc[:, :-2].values
pimcov = pims.T @ pims

spect = la.eig(pimcov)
eigbase = spect[1][:, :2].real

PCApims = pims @ eigbase

pca_df = pd.DataFrame(np.c_[PCApims, pim_df.values[:, -1]])
pca_df.columns = ["V1", "V2", "gest"]
pca_df.gest = pca_df.gest.astype("int32")
#pca_df.gest = pca_df.gest.astype("category")

sns.scatterplot("V1", "V2", hue="gest", data=pca_df)
plt.show()

################################# Fitting SVM ##################################

X = pim_df.values[:, :-2]
y = pim_df.values[:, -1]

folds=5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=50)
skf.get_n_splits(X, y)

clf = svm.SVC(gamma="auto")

acc_scores = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    acc_scores.append(clf.score(X_test, y_test))

print(sum(acc_scores) / folds)
