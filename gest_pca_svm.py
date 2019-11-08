################################################################################
####### Support Vector Machine for Testing Gesture/ Subject Separability #######
############ Data set is persistence images - see pim_fullvec_set.py ###########
################################################################################

import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import SpectralClustering

pim_df = pd.read_csv("./pim_vectors_mp40.csv")
#pim_df = pim_df[pim_df.gest != 1.0]

####################### Visualizing Priciple Components ########################

pims = pim_df.values[:, :-2] # persistence image vectors
pimcov = pims.T @ pims

evals, evecs = la.eig(pimcov)
# sort eigenvalues and eigenvectors in descending order
eidx = np.argsort(-evals.real)
evecs = evecs.real[:, eidx]
evals = evals.real[eidx]

# skree plot
skree = [sum(evals[:n+1]) for n, i in enumerate(evals)]
sns.scatterplot(range(10), skree[:10])
plt.show()

########

eval = 2

# PCA dim redux
eigbase = evecs[:, :eval]
PCApims = pims @ eigbase

pca_df = pd.DataFrame(np.c_[PCApims, pim_df.values[:, -2]])
pca_df.columns = ["V1", "V2", "gest"]
pca_df.gest = pca_df.gest.astype("int32")

sns.scatterplot("V1", "V2", hue="gest", data=pca_df)
plt.show()

eigbase = evecs[:, eval:eval+3]
PCApims = pims @ eigbase

pca_df3 = pd.DataFrame(np.c_[PCApims, pim_df.values[:, -2]])
pca_df3.columns = ["V1", "V2", "V3", "gest"]
pca_df3.gest = pca_df3.gest.astype("category")

fig = px.scatter_3d(pca_df3, x='V1', y='V2', z='V3', color='gest')
fig.show()

########################### Fitting SVM to PCA Pims ############################

eigbase = evecs[:, :9]

X = pim_df.values[:, :-2] @ eigbase
y = pim_df.values[:, -2]

folds=5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
skf.get_n_splits(X, y)

clf = svm.SVC(gamma="auto")

acc_scores = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    acc_scores.append(clf.score(X_test, y_test))

print(sum(acc_scores) / folds)
